from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from urllib.request import HTTPRedirectHandler, Request, build_opener


TOKEN_ENV = "CRYPTO_TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "CRYPTO_TELEGRAM_CHAT_ID"
API_ORIGIN = "https://api.telegram.org"
_TOKEN_PATTERN = re.compile(r"[0-9]{5,20}:[A-Za-z0-9_-]{20,128}\Z")
_CHAT_PATTERN = re.compile(r"(?:-?[1-9][0-9]{0,19}|@[A-Za-z][A-Za-z0-9_]{4,31})\Z")
_SIGNAL_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class TelegramError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TelegramDelivery:
    status: str
    message_id: int | None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class TelegramNotifier:
    def __init__(self, *, opener=None, timeout_seconds: float = 20.0) -> None:  # type: ignore[no-untyped-def]
        token = os.environ.get(TOKEN_ENV, "")
        chat_id = os.environ.get(CHAT_ID_ENV, "")
        if not _TOKEN_PATTERN.fullmatch(token):
            raise TelegramError(f"{TOKEN_ENV} tanimli veya gecerli degil")
        if not _CHAT_PATTERN.fullmatch(chat_id):
            raise TelegramError(f"{CHAT_ID_ENV} tanimli veya gecerli degil")
        self._token = token
        self._chat_id = chat_id
        self._opener = opener or build_opener(_NoRedirect()).open
        self._timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        return f"{type(self).__name__}(credentials=<redacted>)"

    def send_message(self, text: str) -> int:
        if not isinstance(text, str) or not 1 <= len(text) <= 4096 or "\x00" in text:
            raise ValueError("Gecersiz Telegram mesaji")
        payload = json.dumps(
            {
                "chat_id": self._chat_id,
                "text": text,
                "link_preview_options": {"is_disabled": True},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            f"{API_ORIGIN}/bot{self._token}/sendMessage",
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "btc-eth-probability-bot/0.1",
            },
            method="POST",
        )
        try:
            response = self._opener(request, timeout=self._timeout_seconds)
            with response:
                status = response.getcode()
                raw = response.read(1024 * 1024 + 1)
        except Exception:
            raise TelegramError("Telegram istegi basarisiz") from None
        if status != 200 or not raw or len(raw) > 1024 * 1024:
            raise TelegramError("Telegram yaniti gecersiz")
        try:
            decoded = json.loads(raw.decode("utf-8"))
            result = decoded["result"]
            message_id = result["message_id"]
            response_chat = result["chat"]
            response_chat_id = response_chat["id"]
        except (UnicodeError, json.JSONDecodeError, KeyError, TypeError):
            raise TelegramError("Telegram yaniti dogrulanamadi") from None
        if decoded.get("ok") is not True or not isinstance(message_id, int) or message_id <= 0:
            raise TelegramError("Telegram mesaji kabul etmedi")
        if self._chat_id.lstrip("-").isdigit():
            if isinstance(response_chat_id, bool) or str(response_chat_id) != self._chat_id:
                raise TelegramError("Telegram hedef kanali dogrulanamadi")
        else:
            response_username = response_chat.get("username")
            if not isinstance(response_username, str) or (
                f"@{response_username}".lower() != self._chat_id.lower()
            ):
                raise TelegramError("Telegram hedef kanali dogrulanamadi")
        return message_id

    def deliver_once(self, *, signal_id: str, text: str, state_dir: Path) -> TelegramDelivery:
        if not _SIGNAL_PATTERN.fullmatch(signal_id):
            raise ValueError("Gecersiz sinyal kimligi")
        state_dir.mkdir(parents=True, exist_ok=True)
        intent = state_dir / f"{signal_id}.intent.json"
        receipt = state_dir / f"{signal_id}.receipt.json"
        if receipt.exists():
            payload = _read_state(receipt)
            if (
                payload.get("schema") != "telegram-receipt-v1"
                or payload.get("signal_id") != signal_id
                or isinstance(payload.get("message_id"), bool)
                or not isinstance(payload.get("message_id"), int)
                or int(payload["message_id"]) <= 0
            ):
                raise TelegramError("Telegram teslimat makbuzu dogrulanamadi")
            return TelegramDelivery(status="DEDUPLICATED", message_id=int(payload["message_id"]))
        if intent.exists():
            return TelegramDelivery(status="UNCERTAIN", message_id=None)
        try:
            with intent.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump({"schema": "telegram-intent-v1", "signal_id": signal_id}, handle)
                handle.write("\n")
        except FileExistsError:
            return TelegramDelivery(status="UNCERTAIN", message_id=None)
        try:
            message_id = self.send_message(text)
        except TelegramError:
            return TelegramDelivery(status="UNCERTAIN", message_id=None)
        try:
            with receipt.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    {
                        "schema": "telegram-receipt-v1",
                        "signal_id": signal_id,
                        "message_id": message_id,
                    },
                    handle,
                )
                handle.write("\n")
        except OSError:
            return TelegramDelivery(status="UNCERTAIN", message_id=None)
        return TelegramDelivery(status="SENT", message_id=message_id)


def _read_state(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise TelegramError("Telegram teslimat durumu bozuk") from None
    if not isinstance(payload, dict):
        raise TelegramError("Telegram teslimat durumu bozuk")
    return payload


__all__ = ["CHAT_ID_ENV", "TOKEN_ENV", "TelegramDelivery", "TelegramError", "TelegramNotifier"]
