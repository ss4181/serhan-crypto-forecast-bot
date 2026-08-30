from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from importlib.resources import files
from typing import Any


UNIVERSE_SCHEMA = "trade1-universe-v1"
UNIVERSE_RESOURCE = "resources/trade1_universe_v1.json"
SCALP_SYMBOLS_ENV = "CRYPTO_SCALP_SYMBOLS"
_SYMBOL_PATTERN = re.compile(r"[A-Z0-9]{1,20}USDT\Z")
_EXPECTED_COUNTS = {"core30": 30, "extended59": 59}
_ALLOWED_STRATEGIES = {"S1", "S1+S4", "S2", "S3"}


@dataclass(frozen=True, slots=True)
class UniverseEntry:
    spot_symbol: str
    perpetual_symbol: str
    group: str
    trade1_strategies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UniverseManifest:
    version: str
    source: str
    entries: tuple[UniverseEntry, ...]
    scalp_horizons_minutes: tuple[int, ...]
    scalp_round_trip_cost_bps: float
    scalp_status: str

    def selected_entries(self, raw: str | None = None) -> tuple[UniverseEntry, ...]:
        configured = os.environ.get(SCALP_SYMBOLS_ENV, "") if raw is None else raw
        requested = [item.strip().upper() for item in configured.split(",") if item.strip()]
        if not requested:
            return self.entries
        by_symbol = {entry.spot_symbol: entry for entry in self.entries}
        unknown = [symbol for symbol in requested if symbol not in by_symbol]
        if unknown:
            raise ValueError(
                f"{SCALP_SYMBOLS_ENV} trade1 evreninde olmayan sembol iceriyor: "
                + ", ".join(unknown)
            )
        seen: set[str] = set()
        selected: list[UniverseEntry] = []
        for symbol in requested:
            if symbol not in seen:
                selected.append(by_symbol[symbol])
                seen.add(symbol)
        return tuple(selected)


def load_trade1_universe(path: Path | None = None) -> UniverseManifest:
    if path is None:
        resource = files("crypto_forecaster").joinpath(UNIVERSE_RESOURCE)
        try:
            payload = json.loads(resource.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("Trade1 evren manifesti okunamadi") from None
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("Trade1 evren manifesti okunamadi") from None
    return _parse_manifest(payload)


def _parse_manifest(payload: Any) -> UniverseManifest:
    if not isinstance(payload, dict) or payload.get("schema") != UNIVERSE_SCHEMA:
        raise ValueError("Trade1 evren manifesti semasi gecersiz")
    version = _text(payload.get("version"), "evren surumu")
    source = _text(payload.get("source"), "evren kaynagi")
    groups = payload.get("groups")
    overrides = payload.get("perpetual_overrides", {})
    policy = payload.get("scalp_policy")
    if not isinstance(groups, dict) or not isinstance(overrides, dict) or not isinstance(policy, dict):
        raise ValueError("Trade1 evren manifesti yapisi gecersiz")

    entries: list[UniverseEntry] = []
    seen_spot: set[str] = set()
    seen_perpetual: set[str] = set()
    for group, expected_count in _EXPECTED_COUNTS.items():
        item = groups.get(group)
        if not isinstance(item, dict):
            raise ValueError(f"Trade1 evreninde {group} grubu eksik")
        raw_symbols = item.get("spot_symbols")
        raw_strategies = item.get("trade1_strategies")
        if not isinstance(raw_symbols, list) or len(raw_symbols) != expected_count:
            raise ValueError(f"Trade1 {group} grubu {expected_count} sembol icermeli")
        if not isinstance(raw_strategies, list) or not raw_strategies:
            raise ValueError(f"Trade1 {group} strateji yetkileri gecersiz")
        strategies = tuple(_text(value, "strateji") for value in raw_strategies)
        if set(strategies) - _ALLOWED_STRATEGIES or len(set(strategies)) != len(strategies):
            raise ValueError(f"Trade1 {group} strateji yetkileri gecersiz")
        for raw_symbol in raw_symbols:
            spot = _symbol(raw_symbol)
            perpetual = _symbol(overrides.get(spot, spot))
            if spot in seen_spot or perpetual in seen_perpetual:
                raise ValueError("Trade1 evreninde yinelenen sembol var")
            entries.append(
                UniverseEntry(
                    spot_symbol=spot,
                    perpetual_symbol=perpetual,
                    group=group,
                    trade1_strategies=strategies,
                )
            )
            seen_spot.add(spot)
            seen_perpetual.add(perpetual)

    horizons = policy.get("evaluation_horizons_minutes")
    try:
        parsed_horizons = tuple(int(value) for value in horizons)
        cost = float(policy.get("round_trip_cost_bps"))
    except (TypeError, ValueError):
        raise ValueError("Trade1 scalp arastirma politikasi gecersiz") from None
    if parsed_horizons != (15, 30, 60) or cost <= 0:
        raise ValueError("Trade1 scalp arastirma politikasi gecersiz")
    status = _text(policy.get("status"), "scalp arastirma durumu")
    if status != "research_only":
        raise ValueError("Scalp evreni yalniz research_only olabilir")
    return UniverseManifest(
        version=version,
        source=source,
        entries=tuple(entries),
        scalp_horizons_minutes=parsed_horizons,
        scalp_round_trip_cost_bps=cost,
        scalp_status=status,
    )


def _symbol(value: object) -> str:
    symbol = _text(value, "sembol").upper()
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"Trade1 evreninde gecersiz sembol: {symbol}")
    return symbol


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Trade1 {label} gecersiz")
    return value.strip()


__all__ = [
    "SCALP_SYMBOLS_ENV",
    "UNIVERSE_SCHEMA",
    "UniverseEntry",
    "UniverseManifest",
    "load_trade1_universe",
]
