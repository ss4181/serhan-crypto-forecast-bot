from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crypto_forecaster.commands import (
    load_members,
    load_pending_members,
    poll_and_answer,
    safe_name,
    save_members,
)
from crypto_forecaster.config import Settings
from crypto_forecaster.telegram import TelegramError, telegram_menu_keyboard

OWNER = 500100
MEMBER = 700200
STRANGER = 900300


class FakeNotifier:
    def __init__(self, updates: list[dict]) -> None:
        self._updates = updates
        self.sent: list[tuple[int, str]] = []
        self.offsets: list[int] = []
        self.callbacks: list[str] = []

    def get_updates(self, *, offset: int, limit: int = 20) -> list[dict]:
        self.offsets.append(offset)
        return [item for item in self._updates if item["update_id"] >= offset][:limit]

    def send_reply(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: dict[str, object] | None = None,
    ) -> int:
        self.sent.append((chat_id, text))
        return len(self.sent)

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        self.callbacks.append(callback_query_id)


def update(update_id: int, sender: int, text: str, chat: int | None = None) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "from": {"id": sender, "first_name": f"Kisi {sender}"},
            "chat": {
                "id": chat if chat is not None else sender,
                "type": "private" if chat is None or chat == sender else "group",
            },
            "text": text,
        },
    }


def callback_update(
    update_id: int, sender: int, data: str, chat: int | None = None
) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"callback-{update_id}",
            "from": {"id": sender, "first_name": f"Kisi {sender}"},
            "message": {
                "chat": {
                    "id": chat if chat is not None else sender,
                    "type": "private" if chat is None or chat == sender else "group",
                }
            },
            "data": data,
        },
    }


def run(
    directory: Path,
    updates: list[dict],
    members: dict[int, str] | None = None,
    scalp_performance_text=None,
):
    settings = Settings(
        telegram_state_dir=directory, outcome_state_dir=directory / "outcomes"
    )
    if members:
        save_members(directory, members)
    notifier = FakeNotifier(updates)
    outcome = poll_and_answer(
        settings,
        status_text=lambda: "DURUM METNI",
        performance_text=lambda days: f"KARNE {days} GUN",
        scalp_performance_text=scalp_performance_text,
        notifier=notifier,
        reply_markup=telegram_menu_keyboard(),
    )
    return outcome, notifier, settings


@patch.dict(os.environ, {"CRYPTO_TELEGRAM_OWNER_ID": str(OWNER)}, clear=False)
class CommandTests(unittest.TestCase):
    def test_stranger_gets_no_reply_at_all(self) -> None:
        # Replying to unknown senders would let anyone make the bot emit
        # traffic on demand, so an unauthorised command is answered with silence.
        with tempfile.TemporaryDirectory() as directory:
            outcome, notifier, _ = run(Path(directory), [update(1, STRANGER, "/durum")])
        self.assertEqual(notifier.sent, [])
        self.assertEqual(outcome.refused, 1)
        self.assertEqual(outcome.answered, 0)

    def test_member_can_ask_for_status_and_performance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome, notifier, _ = run(
                Path(directory),
                [update(1, MEMBER, "/durum"), update(2, MEMBER, "/performans 7")],
                members={MEMBER: "Ortak"},
            )
        self.assertEqual(outcome.answered, 2)
        self.assertEqual(notifier.sent[0], (MEMBER, "DURUM METNI"))
        self.assertEqual(notifier.sent[1], (MEMBER, "KARNE 7 GUN"))

    def test_member_cannot_see_other_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, notifier, _ = run(
                Path(directory),
                [update(1, MEMBER, "/kisiler")],
                members={MEMBER: "Ortak", STRANGER: "Gizli Kisi"},
            )
        self.assertIn("yalnızca bot sahibine", notifier.sent[0][1])
        self.assertNotIn(str(STRANGER), notifier.sent[0][1])

    def test_group_commands_are_never_answered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome, notifier, _ = run(
                Path(directory), [update(1, OWNER, "/durum", chat=-100123)]
            )
        self.assertEqual(outcome.refused, 1)
        self.assertEqual(notifier.sent, [])

    def test_join_request_requires_owner_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome, notifier, _ = run(
                root,
                [
                    update(1, STRANGER, "/katil"),
                    callback_update(2, OWNER, f"approve_member:{STRANGER}"),
                ],
            )
            stored = load_members(root)
            pending = load_pending_members(root)
        self.assertEqual(outcome.refused, 0)
        self.assertEqual(stored.get(STRANGER), f"Kisi {STRANGER}")
        self.assertEqual(pending, {})
        self.assertEqual(notifier.sent[0][0], STRANGER)
        self.assertEqual(notifier.sent[1][0], OWNER)
        self.assertIn("YENİ ERİŞİM İSTEĞİ", notifier.sent[1][1])
        self.assertEqual(notifier.sent[-1][0], STRANGER)
        self.assertIn("onaylandı", notifier.sent[-1][1])

    def test_duplicate_join_request_notifies_owner_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, notifier, _ = run(
                Path(directory),
                [update(1, STRANGER, "/katil"), update(2, STRANGER, "/katil")],
            )
        owner_messages = [message for chat, message in notifier.sent if chat == OWNER]
        self.assertEqual(len(owner_messages), 1)

    def test_authorized_inline_button_returns_explanations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome, notifier, _ = run(
                Path(directory), [callback_update(1, OWNER, "explanations")]
            )
        self.assertEqual(outcome.answered, 1)
        self.assertEqual(notifier.callbacks, ["callback-1"])
        self.assertIn("ACIKLAMALAR", notifier.sent[0][1])
        self.assertIn("SCALP AILELERI", notifier.sent[0][1])

    def test_start_and_members_inline_buttons_match_existing_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome, notifier, _ = run(
                Path(directory),
                [
                    callback_update(1, OWNER, "start"),
                    callback_update(2, OWNER, "members"),
                ],
            )
        self.assertEqual(outcome.answered, 2)
        self.assertIn("/durum", notifier.sent[0][1])
        self.assertIn(f"Sahip: {OWNER}", notifier.sent[1][1])

    def test_unauthorized_inline_button_is_silent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome, notifier, _ = run(
                Path(directory), [callback_update(1, STRANGER, "explanations")]
            )
        self.assertEqual(outcome.refused, 1)
        self.assertEqual(outcome.answered, 0)
        self.assertEqual(notifier.callbacks, ["callback-1"])
        self.assertEqual(notifier.sent, [])

    def test_performance_defaults_to_thirty_days(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, notifier, _ = run(
                Path(directory), [update(1, OWNER, "/performans")], members={}
            )
        self.assertEqual(notifier.sent[0][1], "KARNE 30 GUN")

    def test_scalp_scorecard_button_uses_the_requested_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, notifier, _ = run(
                Path(directory),
                [callback_update(1, OWNER, "scalp_performance:7")],
                scalp_performance_text=lambda days: f"SCALP KARNE {days} GUN",
            )
        self.assertEqual(notifier.sent[0][1], "SCALP KARNE 7 GUN")

    def test_only_the_owner_may_add_people(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome, notifier, _ = run(
                root,
                [update(1, MEMBER, f"/ekle {STRANGER} Yeni")],
                members={MEMBER: "Ortak"},
            )
            stored = load_members(root)
        self.assertEqual(outcome.answered, 1)
        self.assertIn("yalnızca bot sahibi", notifier.sent[0][1])
        self.assertNotIn(STRANGER, stored)

    def test_owner_adds_a_person_who_can_then_ask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run(root, [update(1, OWNER, f"/ekle {STRANGER} Ayse Yilmaz")])
            self.assertEqual(load_members(root).get(STRANGER), "Ayse Yilmaz")
            _, notifier, _ = run(root, [update(2, STRANGER, "/durum")])
        self.assertEqual(notifier.sent[0], (STRANGER, "DURUM METNI"))

    def test_owner_can_remove_a_person(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run(root, [update(1, OWNER, f"/sil {MEMBER}")], members={MEMBER: "Ortak"})
            self.assertEqual(load_members(root), {})

    def test_add_requires_a_numeric_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, notifier, _ = run(
                Path(directory), [update(1, OWNER, "/ekle @kullanici")]
            )
        self.assertIn("sayisal Telegram kimligi", notifier.sent[0][1])

    def test_offset_advances_so_commands_are_answered_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            updates = [update(41, OWNER, "/durum")]
            run(root, updates)
            _, notifier, _ = run(root, updates)
        self.assertEqual(notifier.offsets, [42])
        self.assertEqual(notifier.sent, [])

    def test_plain_chatter_is_not_a_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome, notifier, _ = run(Path(directory), [update(1, OWNER, "gunaydin")])
        self.assertEqual(notifier.sent, [])
        self.assertEqual(outcome.answered, 0)

    def test_unknown_command_is_refused_politely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, notifier, _ = run(Path(directory), [update(1, OWNER, "/alsat BTC 1000")])
        self.assertIn("Bilinmeyen komut", notifier.sent[0][1])

    def test_help_lists_owner_commands_only_for_the_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, owner_view, _ = run(root, [update(1, OWNER, "/yardim")])
            _, member_view, _ = run(
                root, [update(2, MEMBER, "/yardim")], members={MEMBER: "O"}
            )
        self.assertIn("/ekle", owner_view.sent[0][1])
        self.assertNotIn("/ekle", member_view.sent[0][1])

    def test_names_cannot_smuggle_markup(self) -> None:
        self.assertEqual(safe_name("<b>Ahmet</b>"), "bAhmetb")
        self.assertEqual(safe_name("   "), "isimsiz")
        self.assertEqual(len(safe_name("A" * 200)), 40)
        self.assertEqual(safe_name("Şeyma Öz-Çelik"), "Şeyma Öz-Çelik")


class DisabledCommandTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_without_an_owner_nothing_is_polled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outcome, notifier, _ = run(Path(directory), [update(1, OWNER, "/durum")])
        self.assertEqual(outcome.received, 0)
        self.assertEqual(notifier.offsets, [])


class FailureVisibilityTests(unittest.TestCase):
    @patch.dict(os.environ, {"CRYPTO_TELEGRAM_OWNER_ID": str(OWNER)}, clear=False)
    def test_a_send_failure_is_counted_and_explained(self) -> None:
        # A dropped reply used to look exactly like a command that was never
        # sent: consumed, unanswered, unlogged.
        class BrokenNotifier(FakeNotifier):
            def send_reply(
                self,
                chat_id: int,
                text: str,
                *,
                reply_markup: dict[str, object] | None = None,
            ) -> int:
                raise TelegramError("kanal reddetti")

        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(telegram_state_dir=Path(directory))
            outcome = poll_and_answer(
                settings,
                status_text=lambda: "DURUM",
                performance_text=lambda days: "KARNE",
                notifier=BrokenNotifier([update(1, OWNER, "/durum")]),
            )
        self.assertEqual(outcome.answered, 0)
        self.assertEqual(outcome.failed, 1)
        self.assertIn("kanal reddetti", outcome.detail)

    @patch.dict(os.environ, {"CRYPTO_TELEGRAM_OWNER_ID": str(OWNER)}, clear=False)
    def test_a_broken_status_report_still_answers_the_asker(self) -> None:
        def exploding() -> str:
            raise ValueError("mesaj 4096 karakteri asti")

        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(telegram_state_dir=Path(directory))
            notifier = FakeNotifier([update(1, OWNER, "/durum")])
            outcome = poll_and_answer(
                settings,
                status_text=exploding,
                performance_text=lambda days: "KARNE",
                notifier=notifier,
            )
        # The person asking gets told, the cycle survives, the log records why.
        self.assertEqual(outcome.answered, 1)
        self.assertEqual(outcome.failed, 1)
        self.assertIn("4096", outcome.detail)
        self.assertIn("uretilemedi", notifier.sent[0][1])


if __name__ == "__main__":
    unittest.main()
