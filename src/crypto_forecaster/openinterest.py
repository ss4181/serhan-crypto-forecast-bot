"""Collect perpetual open interest forward, because history cannot be bought.

Binance keeps roughly 30 days of `openInterestHist` and answers HTTP 400 beyond
it, so open interest cannot be backtested against a year of candles today.  It
can be backtested in six months -- but only if collection starts now.  Nothing
here feeds a model; this is a recorder, and its output carries no claim.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .config import validate_symbol


FUTURES_ORIGIN = "https://fapi.binance.com"
PERIOD = "5m"
PAGE_LIMIT = 500
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
COLUMNS = ("timestamp_ms", "open_interest", "open_interest_value")
# The endpoint publishes on a 5 minute grid; asking more often just burns calls.
MINIMUM_REFRESH_MS = 5 * 60 * 1000


class OpenInterestError(RuntimeError):
    pass


def open_interest_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / f"{validate_symbol(symbol)}_open_interest.csv"


def fetch_open_interest(symbol: str, *, opener: Callable[..., object] | None = None) -> pd.DataFrame:
    query = urlencode({"symbol": validate_symbol(symbol), "period": PERIOD, "limit": PAGE_LIMIT})
    request = Request(
        f"{FUTURES_ORIGIN}/futures/data/openInterestHist?{query}",
        headers={"Accept": "application/json", "User-Agent": "btc-eth-probability-bot/0.1"},
        method="GET",
    )
    try:
        response = (opener or urlopen)(request, timeout=20.0)
        with response:  # type: ignore[attr-defined]
            status = response.getcode()  # type: ignore[attr-defined]
            raw = response.read(MAX_RESPONSE_BYTES + 1)  # type: ignore[attr-defined]
    except (HTTPError, URLError, TimeoutError, OSError):
        raise OpenInterestError("Acik pozisyon verisi alinamadi") from None
    except Exception:
        raise OpenInterestError("Acik pozisyon verisi alinamadi") from None
    if status != 200 or not raw or len(raw) > MAX_RESPONSE_BYTES:
        raise OpenInterestError("Acik pozisyon yaniti gecersiz")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise OpenInterestError("Acik pozisyon yaniti JSON degil") from None
    if not isinstance(decoded, list):
        raise OpenInterestError("Acik pozisyon yaniti liste degil")
    rows = []
    for item in decoded:
        if not isinstance(item, dict):
            raise OpenInterestError("Acik pozisyon satiri gecersiz")
        try:
            rows.append(
                {
                    "timestamp_ms": int(item["timestamp"]),
                    "open_interest": float(item["sumOpenInterest"]),
                    "open_interest_value": float(item["sumOpenInterestValue"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            raise OpenInterestError("Acik pozisyon alani gecersiz") from None
    return pd.DataFrame(rows, columns=list(COLUMNS))


def update_open_interest(
    data_dir: Path,
    symbol: str,
    *,
    opener: Callable[..., object] | None = None,
    now: datetime | None = None,
) -> int:
    """Append whatever is new.  Returns how many rows were added."""
    path = open_interest_path(data_dir, symbol)
    existing = load_open_interest(data_dir, symbol)
    current_ms = int((now or datetime.now(timezone.utc)).timestamp() * 1000)
    if not existing.empty:
        newest = int(existing["timestamp_ms"].iloc[-1])
        if current_ms - newest < MINIMUM_REFRESH_MS:
            return 0
    fetched = fetch_open_interest(symbol, opener=opener)
    if fetched.empty:
        return 0
    combined = (
        pd.concat([existing, fetched], ignore_index=True)
        .drop_duplicates("timestamp_ms", keep="last")
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )
    added = len(combined) - len(existing)
    if added <= 0:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return added


def load_open_interest(data_dir: Path, symbol: str) -> pd.DataFrame:
    path = open_interest_path(data_dir, symbol)
    if not path.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame(columns=list(COLUMNS))
    if set(COLUMNS) - set(frame.columns):
        return pd.DataFrame(columns=list(COLUMNS))
    frame = frame.loc[:, list(COLUMNS)]
    frame["timestamp_ms"] = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    frame = frame.dropna(subset=["timestamp_ms"])
    frame["timestamp_ms"] = frame["timestamp_ms"].astype("int64")
    return frame.sort_values("timestamp_ms").reset_index(drop=True)


def coverage(data_dir: Path, symbol: str) -> tuple[int, float]:
    """Rows held and days spanned, for reporting how usable the record is yet."""
    frame = load_open_interest(data_dir, symbol)
    if len(frame) < 2:
        return len(frame), 0.0
    span_ms = int(frame["timestamp_ms"].iloc[-1]) - int(frame["timestamp_ms"].iloc[0])
    return len(frame), span_ms / 86_400_000


__all__ = [
    "OpenInterestError",
    "coverage",
    "fetch_open_interest",
    "load_open_interest",
    "open_interest_path",
    "update_open_interest",
]
