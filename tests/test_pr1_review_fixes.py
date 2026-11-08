from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from crypto_forecaster import service
from crypto_forecaster.config import Settings
from crypto_forecaster.long_2pct import (
    _load_state,
    run_long_2pct_experiment,
    select_candidates,
)
from crypto_forecaster.scalping import (
    BullRegime,
    ScalpObservation,
    ScalpScanReport,
    ScalpSetupAssessment,
    apply_scalp_notification_safety_gates,
    current_bracket_rows,
    filter_scalp_notification_report,
    format_scalp_observation_digest,
    load_pending_scalp_brackets,
    record_scalp_bracket_setups,
)
from crypto_forecaster.telegram import TelegramDelivery

ROOT = Path(__file__).resolve().parents[1]
STEP = 300_000
CLOSE = 1_790_000_099_999 - (1_790_000_099_999 + 1) % STEP
ASSESSMENT = ScalpSetupAssessment(
    "DIRECTIONAL_LONG", "LONG", "YUKARI", 60, 0.6, 30.0, 40, 5, 0.8, "ORTA"
)


def observation(symbol: str, family: str, close: int = CLOSE) -> ScalpObservation:
    return ScalpObservation(
        universe_version="v",
        spot_symbol=symbol,
        perpetual_symbol=symbol,
        universe_group="core",
        family=family,
        score=3.0,
        price=100.0,
        bar_open_time_ms=close - STEP + 1,
        bar_close_time_ms=close,
        details=(),
        regime_state="BULL",
        execution_eligible=True,
        alert_tier="KURULUM",
    )


def report(*items: ScalpObservation, newest: int = CLOSE) -> ScalpScanReport:
    return ScalpScanReport(
        "v", 89, 89, 0, (), tuple(items), newest + 1, BullRegime("BULL", 0, 0, 0, False, 89),
        newest_close_time_ms=newest,
    )


class ShadowFamilyTests(unittest.TestCase):
    def run_filter(self, *items: ScalpObservation) -> tuple[ScalpScanReport, dict[str, int]]:
        diagnostics: dict[str, int] = {}
        with patch("crypto_forecaster.scalping.scalp_setup_assessment", return_value=ASSESSMENT):
            result = filter_scalp_notification_report(
                report(*items), minimum_score=1.0, live_families=("F1", "B1"), diagnostics=diagnostics
            )
        return result, diagnostics

    def test_shadow_family_does_not_veto_a_confirmed_setup(self) -> None:
        result, diagnostics = self.run_filter(
            observation("A", "F1"), observation("A", "B1"), observation("A", "B4")
        )
        self.assertEqual(diagnostics, {"eligible": 1})
        self.assertEqual({item.family for item in result.observations}, {"F1", "B1"})

    def test_shadow_family_does_not_count_as_confirmation(self) -> None:
        result, diagnostics = self.run_filter(observation("A", "F1"), observation("A", "B4"))
        self.assertEqual(result.observations, ())
        self.assertEqual(diagnostics, {"no_setup": 1})

    def test_only_shadow_families_are_reported_as_such(self) -> None:
        _, diagnostics = self.run_filter(observation("A", "B4"))
        self.assertEqual(diagnostics, {"shadow_only_family": 1})


class DeploymentTemplateTests(unittest.TestCase):
    def test_template_matches_code_defaults_for_changed_settings(self) -> None:
        values = dict(
            line.split("=", 1)
            for line in (ROOT / "deploy/env.example").read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.startswith("#")
        )
        with patch.dict("os.environ", {}, clear=True):
            defaults = Settings()
        self.assertEqual(
            tuple(values["CRYPTO_SCALP_LIVE_FAMILIES"].split(",")), defaults.scalp_live_families
        )
        self.assertEqual(values["CRYPTO_SCALP_DIRECTION_POLICY"], defaults.scalp_direction_policy)
        self.assertEqual(
            float(values["CRYPTO_SCALP_TRANSITION_MIN_EXPECTED_NET_BPS"]),
            defaults.scalp_transition_minimum_expected_net_bps,
        )
        self.assertEqual(
            int(values["CRYPTO_SCALP_BRACKET_HORIZON_MINUTES"]),
            defaults.scalp_bracket_horizon_minutes,
        )


class CliPolicyTests(unittest.TestCase):
    def test_every_policy_sensitive_cli_call_passes_the_setting(self) -> None:
        names = {
            "record_scalp_target_setups",
            "record_scalp_bracket_setups",
            "filter_scalp_notification_report",
        }
        tree = ast.parse((ROOT / "src/crypto_forecaster/cli.py").read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") in names
        ]
        self.assertEqual(len(calls), 5)
        for call in calls:
            self.assertIn("direction_policy", {keyword.arg for keyword in call.keywords})


class ConcurrencyWindowTests(unittest.TestCase):
    def blocked(self, age_minutes: int) -> str | None:
        now = CLOSE + age_minutes * 60_000
        pending = [{"notification_sent": True, "bar_close_time_ms": CLOSE, "horizon_minutes": 240}]
        with tempfile.TemporaryDirectory() as tmp, patch(
            "crypto_forecaster.scalping.load_pending_scalp_brackets", return_value=pending
        ):
            settings = Settings(
                scalp_alert_kill_switch=False,
                scalp_maximum_concurrent_research_alerts=1,
                scalp_state_dir=Path(tmp),
            )
            _, reason = apply_scalp_notification_safety_gates(
                report(observation("A", "F1"), observation("A", "B1")),
                settings, ledger=(), now_ms=now,
            )
        return reason

    def test_long_bracket_only_occupies_a_slot_for_the_first_hour(self) -> None:
        self.assertEqual(self.blocked(59), "max_concurrent_research_alerts")
        self.assertIsNone(self.blocked(61))


class BracketPolicyTests(unittest.TestCase):
    def test_new_policy_rows_exclude_older_results(self) -> None:
        rows = [
            {"setup_id": "old"},
            {"setup_id": "v2-60", "bracket_policy": "v2-60m", "bar_close_time_ms": 1},
            {"setup_id": "v2-240", "bracket_policy": "v2-240m", "bar_close_time_ms": 2},
        ]
        self.assertEqual([row["setup_id"] for row in current_bracket_rows(rows)], ["v2-240"])

    def test_unversioned_ledger_is_unchanged(self) -> None:
        rows = [{"setup_id": "a"}, {"setup_id": "b"}]
        self.assertEqual(current_bracket_rows(rows), rows)

    def test_recorded_setup_carries_the_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "crypto_forecaster.scalping.scalp_setup_assessment", return_value=ASSESSMENT
        ):
            state = Path(tmp)
            record_scalp_bracket_setups(
                state, report(observation("A", "F1"), observation("A", "B1")),
                manifest=SimpleNamespace(version="v"), top_k=2, horizon_minutes=240,
            )
            pending = load_pending_scalp_brackets(state)
        self.assertEqual(pending[0]["bracket_policy"], "dynamic-bracket-v2-240m")


class DigestLabelTests(unittest.TestCase):
    def test_digest_separates_close_statistics_from_the_bracket(self) -> None:
        with patch("crypto_forecaster.scalping.scalp_setup_assessment", return_value=ASSESSMENT):
            text = format_scalp_observation_digest(
                report(observation("A", "F1"), observation("A", "B1")),
                manifest=SimpleNamespace(version="v", size=89),
                top_k=2,
                settings=Settings(scalp_bracket_horizon_minutes=240),
            )
        self.assertIn("Net beklenti (60 dk)", text)
        self.assertIn("hedef/stop (240 dk)", text)
        self.assertIn("hedef/stop başarı olasılığı değildir", text)


class Notifier:
    def __init__(self, *statuses: str) -> None:
        self.statuses = list(statuses)
        self.sent: list[str] = []

    def deliver_owner_once(self, *, signal_id: str, text: str, state_dir: Path) -> TelegramDelivery:
        self.sent.append(text)
        return TelegramDelivery(status=self.statuses.pop(0), message_id=1)


class ExperimentReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.settings = Settings(
            scalp_state_dir=root / "state",
            scalp_data_dir=root / "data",
            telegram_state_dir=root / "telegram",
            long_2pct_experiment_enabled=True,
            scalp_alert_kill_switch=False,
        )
        self.report = report(observation("AUSDT", "F1"), observation("AUSDT", "B1"))

    def at(self, offset_ms: int) -> datetime:
        return datetime.fromtimestamp((CLOSE + offset_ms) / 1000, UTC)

    def test_refused_first_alert_is_retried_inside_the_window(self) -> None:
        notifier = Notifier("REDDEDILDI", "SENT")
        run_long_2pct_experiment(self.settings, self.report, now=self.at(10_000), notifier=notifier)
        run_long_2pct_experiment(self.settings, report(), now=self.at(70_000), notifier=notifier)
        self.assertEqual(len(notifier.sent), 2)

    def test_uncertain_first_alert_is_never_resent(self) -> None:
        notifier = Notifier("UNCERTAIN")
        run_long_2pct_experiment(self.settings, self.report, now=self.at(10_000), notifier=notifier)
        run_long_2pct_experiment(self.settings, report(), now=self.at(70_000), notifier=notifier)
        self.assertEqual(len(notifier.sent), 1)

    def test_older_bar_is_not_a_new_signal(self) -> None:
        stale = report(
            observation("A", "F1", CLOSE - STEP), observation("A", "B1", CLOSE - STEP), newest=CLOSE
        )
        self.assertEqual(select_candidates(stale, [], daily_limit=5), [])

    def test_corrupt_state_stops_instead_of_restarting_the_experiment(self) -> None:
        path = self.settings.scalp_state_dir / "experiments/long-2pct-v1/state.json"
        path.parent.mkdir(parents=True)
        path.write_text("{bad", encoding="utf-8")
        with self.assertRaises(ValueError):
            _load_state(self.settings)
        self.assertEqual(path.read_text(encoding="utf-8"), "{bad")


class ServiceWiringTests(unittest.TestCase):
    def test_serve_loop_passes_the_unfiltered_report(self) -> None:
        source = inspect.getsource(service.serve_forever)
        self.assertIn("_run_long_2pct_step(settings, scalp_report", source)

    def step(self, settings: Settings, primary: bool = True, **run) -> tuple[list[str], object]:
        logs: list[str] = []
        with patch.object(service, "is_primary", return_value=primary), patch.object(
            service, "run_long_2pct_experiment", **run
        ) as runner:
            service._run_long_2pct_step(
                settings, report(), now=datetime(2026, 11, 8, tzinfo=UTC), progress=logs.append
            )
        return logs, runner

    def test_step_requires_primary_and_the_flag(self) -> None:
        on = Settings(long_2pct_experiment_enabled=True)
        self.assertFalse(self.step(on, primary=False, return_value=[])[1].called)
        self.assertFalse(
            self.step(Settings(long_2pct_experiment_enabled=False), return_value=[])[1].called
        )
        logs, runner = self.step(on, return_value=["A aday: SENT"])
        self.assertTrue(runner.called)
        self.assertEqual(logs, ["LONG %2 deney: A aday: SENT"])

    def test_experiment_failure_is_logged_and_contained(self) -> None:
        logs, _ = self.step(Settings(long_2pct_experiment_enabled=True), side_effect=ValueError("x"))
        self.assertEqual(logs, ["LONG %2 deney hatasi: x"])

    def test_status_text_includes_the_experiment_only_when_enabled(self) -> None:
        now = datetime(2026, 11, 8, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            on = service._notification_status_text(
                Settings(scalp_state_dir=state, long_2pct_experiment_enabled=True), now
            )
            off = service._notification_status_text(
                Settings(scalp_state_dir=state, long_2pct_experiment_enabled=False), now
            )
        self.assertIn("Deneysel LONG %2", on)
        self.assertNotIn("Deneysel LONG %2", off)


if __name__ == "__main__":
    unittest.main()
