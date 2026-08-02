from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener


TOKEN_ENV = "CRYPTO_TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "CRYPTO_TELEGRAM_CHAT_ID"
OWNER_ID_ENV = "CRYPTO_TELEGRAM_OWNER_ID"
ROLE_ENV = "CRYPTO_BOT_ROLE"
API_ORIGIN = "https://api.telegram.org"
SEND_ATTEMPTS = 3
MAXIMUM_RETRY_SECONDS = 30.0
_TOKEN_PATTERN = re.compile(r"[0-9]{5,20}:[A-Za-z0-9_-]{20,128}\Z")
_CHAT_PATTERN = re.compile(r"(?:-?[1-9][0-9]{0,19}|@[A-Za-z][A-Za-z0-9_]{4,31})\Z")
_SIGNAL_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class TelegramError(RuntimeError):
    """Something went wrong and the message may or may not have been sent."""


class TelegramRejected(TelegramError):
    """Telegram answered and refused: nothing was delivered.

    Worth separating, because a refusal is safe to retry once its cause is
    fixed, while a timeout is not -- the message may already be in the channel.
    """


@dataclass(frozen=True, slots=True)
class TelegramDelivery:
    status: str
    message_id: int | None
    detail: str = ""


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

    def _api(self, method: str, payload: dict[str, object]) -> object:
        request = Request(
            f"{API_ORIGIN}/bot{self._token}/{method}",
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "btc-eth-probability-bot/0.1",
            },
            method="POST",
        )
        status, raw = self._post(request)
        if status != 200 or not raw or len(raw) > 1024 * 1024:
            raise TelegramError("Telegram yaniti gecersiz")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise TelegramError("Telegram yaniti dogrulanamadi") from None
        if not isinstance(decoded, dict) or decoded.get("ok") is not True:
            raise TelegramRejected("Telegram istegi kabul etmedi")
        return decoded.get("result")

    def get_updates(self, *, offset: int, limit: int = 20) -> list[dict[str, object]]:
        """Read pending commands.  Nothing here is ever treated as an instruction."""
        result = self._api(
            "getUpdates",
            {
                "offset": int(offset),
                "limit": max(1, min(int(limit), 100)),
                "timeout": 0,
                "allowed_updates": ["message"],
            },
        )
        if not isinstance(result, list):
            raise TelegramError("Telegram guncelleme listesi gecersiz")
        return [item for item in result if isinstance(item, dict)]

    def send_reply(self, chat_id: int, text: str) -> int:
        """Answer one authorised person in their own chat, not in the channel."""
        if isinstance(chat_id, bool) or not isinstance(chat_id, int):
            raise ValueError("Gecersiz sohbet kimligi")
        _validate_text(text)
        result = self._api(
            "sendMessage",
            {"chat_id": chat_id, "text": text, "link_preview_options": {"is_disabled": True}},
        )
        if not isinstance(result, dict) or not isinstance(result.get("message_id"), int):
            raise TelegramError("Telegram yaniti dogrulanamadi")
        return int(result["message_id"])

    def send_message(self, text: str) -> int:
        _validate_text(text)
        result = self._api(
            "sendMessage",
            {
                "chat_id": self._chat_id,
                "text": text,
                "link_preview_options": {"is_disabled": True},
            },
        )
        try:
            message_id = result["message_id"]  # type: ignore[index]
            response_chat = result["chat"]  # type: ignore[index]
            response_chat_id = response_chat["id"]
        except (KeyError, TypeError):
            raise TelegramError("Telegram yaniti dogrulanamadi") from None
        if not isinstance(message_id, int) or isinstance(message_id, bool) or message_id <= 0:
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

    def _post(self, request: Request) -> tuple[int, bytes]:
        """Send once, retrying only when Telegram itself rejected the call.

        A rejected request (429 or 5xx) definitely did not deliver a message, so
        retrying is safe.  A timeout or connection error may have delivered it,
        so those are never retried: at-most-once beats a duplicate alert.
        """
        for attempt in range(SEND_ATTEMPTS):
            try:
                response = self._opener(request, timeout=self._timeout_seconds)
                with response:
                    return int(response.getcode()), response.read(1024 * 1024 + 1)
            except HTTPError as error:
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt == SEND_ATTEMPTS - 1:
                    raise TelegramRejected(
                        f"Telegram istegi reddedildi (HTTP {error.code})"
                    ) from None
                time.sleep(_retry_after_seconds(error))
            except Exception:
                raise TelegramError("Telegram istegi basarisiz") from None
        raise TelegramRejected("Telegram istegi reddedildi")

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
            return TelegramDelivery(
                status="UNCERTAIN",
                message_id=None,
                detail="onceki denemenin sonucu bilinmiyor; tekrar denenmez",
            )
        try:
            with intent.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump({"schema": "telegram-intent-v1", "signal_id": signal_id}, handle)
                handle.write("\n")
        except FileExistsError:
            return TelegramDelivery(
                status="UNCERTAIN", message_id=None, detail="es zamanli gonderim denemesi"
            )
        try:
            message_id = self.send_message(text)
        except TelegramRejected as error:
            # Telegram refused, so nothing reached the channel.  Drop the intent
            # so this signal is retried once the cause -- usually a missing
            # posting permission -- is fixed.
            intent.unlink(missing_ok=True)
            return TelegramDelivery(status="REDDEDILDI", message_id=None, detail=str(error))
        except TelegramError as error:
            return TelegramDelivery(status="UNCERTAIN", message_id=None, detail=str(error))
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
            return TelegramDelivery(
                status="UNCERTAIN", message_id=None, detail="makbuz yazilamadi"
            )
        return TelegramDelivery(status="SENT", message_id=message_id)


def digest_signal_id(label: str, bucket: int) -> str:
    return sha256(f"{label}|{int(bucket)}".encode("ascii")).hexdigest()


def is_primary() -> bool:
    """Only one instance may talk to Telegram.

    Two senders would double-post every alert, because each keeps its own
    delivery receipts, and Telegram refuses concurrent getUpdates callers with
    a 409 so commands would be lost as well.  The always-on host is primary;
    the GitHub Actions job runs as standby and stays quiet.
    """
    return os.environ.get(ROLE_ENV, "primary").strip().lower() != "standby"


def _validate_text(text: str) -> None:
    if not isinstance(text, str) or not 1 <= len(text) <= 4096 or "\x00" in text:
        raise ValueError("Gecersiz Telegram mesaji")


def _retry_after_seconds(error: HTTPError) -> float:
    header = error.headers.get("Retry-After") if error.headers else None
    for candidate in (header, _body_retry_after(error)):
        try:
            seconds = float(candidate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if seconds >= 0:
            return min(seconds, MAXIMUM_RETRY_SECONDS)
    return 1.0


def _body_retry_after(error: HTTPError) -> object:
    try:
        payload = json.loads(error.read(64 * 1024).decode("utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("parameters"), dict):
        return payload["parameters"].get("retry_after")
    return None


def _read_state(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise TelegramError("Telegram teslimat durumu bozuk") from None
    if not isinstance(payload, dict):
        raise TelegramError("Telegram teslimat durumu bozuk")
    return payload


__all__ = [
    "CHAT_ID_ENV",
    "OWNER_ID_ENV",
    "ROLE_ENV",
    "TOKEN_ENV",
    "is_primary",
    "TelegramDelivery",
    "TelegramError",
    "TelegramRejected",
    "TelegramNotifier",
    "digest_signal_id",
]
