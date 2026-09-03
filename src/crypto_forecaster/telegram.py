from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

TOKEN_ENV = "CRYPTO_TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "CRYPTO_TELEGRAM_CHAT_ID"
OWNER_ID_ENV = "CRYPTO_TELEGRAM_OWNER_ID"
ROLE_ENV = "CRYPTO_BOT_ROLE"
DELIVERY_MODE_ENV = "CRYPTO_TELEGRAM_DELIVERY_MODE"
API_ORIGIN = "https://api.telegram.org"
SEND_ATTEMPTS = 3
MAXIMUM_RETRY_SECONDS = 30.0
DIRECT_MODE = "direct"
CHANNEL_MODE = "channel"
MEMBER_SCHEMA = "telegram-members-v1"
DASHBOARD_URL = "https://ss4181.github.io/serhan-crypto-forecast-bot/scalp.html"
PRIVATE_COMMANDS = (
    ("start", "Ana menuyu ac"),
    ("durum", "Guncel model durumunu goster"),
    ("performans", "Sinyal performansini goster"),
    ("scalpkarne", "Scalp ileri-test karnesini goster"),
    ("aciklamalar", "Terimleri ve stratejileri acikla"),
    ("katil", "Ozel erisim iste"),
)
OWNER_COMMANDS = PRIVATE_COMMANDS + (
    ("kisiler", "Onayli aboneleri yonet"),
    ("bekleyenler", "Erisim isteklerini yonet"),
)
_TOKEN_PATTERN = re.compile(r"[0-9]{5,20}:[A-Za-z0-9_-]{20,128}\Z")
_CHAT_PATTERN = re.compile(r"(?:-?[1-9][0-9]{0,19}|@[A-Za-z][A-Za-z0-9_]{4,31})\Z")
_SIGNAL_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def telegram_menu_keyboard(*, is_owner: bool = False) -> dict[str, object]:
    """Return the private-chat menu; membership controls are owner-only."""
    rows: list[list[dict[str, str]]] = [
        [
            {"text": "🏠 Başlangıç", "callback_data": "start"},
            {"text": "📖 Açıklamalar", "callback_data": "explanations"},
        ],
        [
            {"text": "📊 Güncel Durum", "callback_data": "status"},
            {"text": "📈 Performans (30g)", "callback_data": "performance:30"},
        ],
        [
            {"text": "🧪 Scalp Karne (30g)", "callback_data": "scalp_performance:30"},
        ],
    ]
    if is_owner:
        rows.append(
            [
                {"text": "👥 Aboneler", "callback_data": "members"},
                {"text": "📨 Bekleyenler", "callback_data": "pending_members"},
            ]
        )
    return {"inline_keyboard": rows}


def telegram_channel_keyboard() -> dict[str, object]:
    """A channel post must never expose private command callbacks."""
    return {
        "inline_keyboard": [[{"text": "📊 Canlı Sinyal Paneli", "url": DASHBOARD_URL}]]
    }


def telegram_delivery_mode() -> str:
    raw = os.environ.get(DELIVERY_MODE_ENV, DIRECT_MODE).strip().lower()
    return raw if raw in {DIRECT_MODE, CHANNEL_MODE} else DIRECT_MODE


def telegram_configured() -> bool:
    token = os.environ.get(TOKEN_ENV, "")
    if not _TOKEN_PATTERN.fullmatch(token):
        return False
    if telegram_delivery_mode() == DIRECT_MODE:
        return _owner_identifier() is not None
    return bool(_CHAT_PATTERN.fullmatch(os.environ.get(CHAT_ID_ENV, "")))


def _owner_identifier() -> int | None:
    raw = os.environ.get(OWNER_ID_ENV, "").strip()
    if not raw.isdigit():
        return None
    identifier = int(raw)
    return identifier if 0 < identifier <= 10**19 else None


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
    def __init__(
        self,
        *,
        state_dir: Path | None = None,
        opener=None,
        timeout_seconds: float = 20.0,
    ) -> None:  # type: ignore[no-untyped-def]
        token = os.environ.get(TOKEN_ENV, "")
        chat_id = os.environ.get(CHAT_ID_ENV, "")
        if not _TOKEN_PATTERN.fullmatch(token):
            raise TelegramError(f"{TOKEN_ENV} tanimli veya gecerli degil")
        mode = telegram_delivery_mode()
        owner = _owner_identifier()
        if mode == CHANNEL_MODE and not _CHAT_PATTERN.fullmatch(chat_id):
            raise TelegramError(f"{CHAT_ID_ENV} tanimli veya gecerli degil")
        if mode == DIRECT_MODE and owner is None:
            raise TelegramError(f"{OWNER_ID_ENV} tanimli veya gecerli degil")
        self._token = token
        self._chat_id = chat_id
        self._mode = mode
        self._owner_id = owner
        self._state_dir = state_dir or Path("state/telegram")
        self._opener = opener or build_opener(_NoRedirect()).open
        self._timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        return f"{type(self).__name__}(credentials=<redacted>)"

    def _api(self, method: str, payload: dict[str, object]) -> object:
        request = Request(
            f"{API_ORIGIN}/bot{self._token}/{method}",
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            ),
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
                "allowed_updates": ["message", "callback_query"],
            },
        )
        if not isinstance(result, list):
            raise TelegramError("Telegram guncelleme listesi gecersiz")
        return [item for item in result if isinstance(item, dict)]

    def send_reply(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: dict[str, object] | None = None,
    ) -> int:
        """Answer one authorised person in their own chat, not in the channel."""
        if isinstance(chat_id, bool) or not isinstance(chat_id, int) or chat_id <= 0:
            raise ValueError("Gecersiz sohbet kimligi")
        _validate_text(text)
        return self._send_to_chat(chat_id, text, reply_markup=reply_markup)

    def send_message(
        self, text: str, *, reply_markup: dict[str, object] | None = None
    ) -> int:
        _validate_text(text)
        if self._mode == DIRECT_MODE:
            message_ids: dict[int, int] = {}
            failures: list[str] = []
            for recipient in self._direct_recipients():
                try:
                    message_ids[recipient] = self._send_to_chat(
                        recipient,
                        text,
                        reply_markup=telegram_menu_keyboard(
                            is_owner=recipient == self._owner_id
                        ),
                    )
                except TelegramError as error:
                    failures.append(f"{recipient}: {error}")
            if self._owner_id not in message_ids:
                detail = failures[0] if failures else "sahip mesaji gonderilemedi"
                raise TelegramError(detail)
            return message_ids[self._owner_id]
        return self._send_to_chat(self._chat_id, text, reply_markup=reply_markup)

    def _send_to_chat(
        self,
        chat_id: int | str,
        text: str,
        *,
        reply_markup: dict[str, object] | None = None,
    ) -> int:
        payload: dict[str, object] = {
            "chat_id": chat_id,
            "text": text,
            "link_preview_options": {"is_disabled": True},
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        result = self._api("sendMessage", payload)
        try:
            message_id = result["message_id"]  # type: ignore[index]
            response_chat = result["chat"]  # type: ignore[index]
            response_chat_id = response_chat["id"]
        except (KeyError, TypeError):
            raise TelegramError("Telegram yaniti dogrulanamadi") from None
        if (
            not isinstance(message_id, int)
            or isinstance(message_id, bool)
            or message_id <= 0
        ):
            raise TelegramError("Telegram mesaji kabul etmedi")
        expected_chat_id = str(chat_id)
        if expected_chat_id.lstrip("-").isdigit():
            if (
                isinstance(response_chat_id, bool)
                or str(response_chat_id) != expected_chat_id
            ):
                raise TelegramError("Telegram hedefi dogrulanamadi")
        else:
            response_username = response_chat.get("username")
            if not isinstance(response_username, str) or (
                f"@{response_username}".lower() != expected_chat_id.lower()
            ):
                raise TelegramError("Telegram hedefi dogrulanamadi")
        return message_id

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        """Stop Telegram's loading spinner after an inline button is pressed."""
        if (
            not isinstance(callback_query_id, str)
            or not 1 <= len(callback_query_id) <= 256
            or "\x00" in callback_query_id
        ):
            raise ValueError("Gecersiz callback kimligi")
        payload: dict[str, object] = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text[:200]
        self._api("answerCallbackQuery", payload)

    def configure_private_command_menu(self) -> None:
        """Publish safe Telegram command menus, with owner controls scoped privately."""
        self._api(
            "setMyCommands",
            {
                "commands": _command_payload(PRIVATE_COMMANDS),
                "scope": {"type": "all_private_chats"},
            },
        )
        if self._owner_id is not None:
            self._api(
                "setMyCommands",
                {
                    "commands": _command_payload(OWNER_COMMANDS),
                    "scope": {"type": "chat", "chat_id": self._owner_id},
                },
            )
        self._api(
            "deleteMyCommands",
            {"scope": {"type": "all_group_chats"}},
        )

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

    def deliver_once(
        self,
        *,
        signal_id: str,
        text: str,
        state_dir: Path,
        reply_markup: dict[str, object] | None = None,
    ) -> TelegramDelivery:
        if not _SIGNAL_PATTERN.fullmatch(signal_id):
            raise ValueError("Gecersiz sinyal kimligi")
        if self._mode == DIRECT_MODE:
            return self._deliver_direct_once(
                signal_id=signal_id,
                text=text,
                state_dir=state_dir,
            )
        return self._deliver_single_once(
            signal_id=signal_id,
            text=text,
            state_dir=state_dir,
            sender=lambda: self._send_to_chat(
                self._chat_id, text, reply_markup=reply_markup
            ),
        )

    def _deliver_direct_once(
        self,
        *,
        signal_id: str,
        text: str,
        state_dir: Path,
    ) -> TelegramDelivery:
        results: list[tuple[int, TelegramDelivery]] = []
        for recipient in self._direct_recipients():
            result = self._deliver_single_once(
                signal_id=signal_id,
                text=text,
                state_dir=state_dir / "direct" / str(recipient),
                sender=lambda recipient=recipient: self._send_to_chat(
                    recipient,
                    text,
                    reply_markup=telegram_menu_keyboard(
                        is_owner=recipient == self._owner_id
                    ),
                ),
            )
            results.append((recipient, result))
        owner_result = next(
            (result for recipient, result in results if recipient == self._owner_id),
            None,
        )
        if owner_result is None:
            return TelegramDelivery(
                status="REDDEDILDI", message_id=None, detail="sahip hedefi bulunamadi"
            )
        failed = [
            f"{recipient}:{result.status}"
            for recipient, result in results
            if result.status not in {"SENT", "DEDUPLICATED"}
        ]
        successful = [
            result for _, result in results if result.status in {"SENT", "DEDUPLICATED"}
        ]
        if successful:
            status = (
                "DEDUPLICATED"
                if all(result.status == "DEDUPLICATED" for result in successful)
                and not failed
                else "SENT"
            )
            detail = f"{len(successful)}/{len(results)} ozel teslimat"
            if failed:
                detail += "; basarisiz: " + ", ".join(failed)
            return TelegramDelivery(
                status=status,
                message_id=owner_result.message_id,
                detail=detail,
            )
        return owner_result

    def _deliver_single_once(
        self,
        *,
        signal_id: str,
        text: str,
        state_dir: Path,
        sender,
    ) -> TelegramDelivery:  # type: ignore[no-untyped-def]
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
            return TelegramDelivery(
                status="DEDUPLICATED", message_id=int(payload["message_id"])
            )
        if intent.exists():
            return TelegramDelivery(
                status="UNCERTAIN",
                message_id=None,
                detail="onceki denemenin sonucu bilinmiyor; tekrar denenmez",
            )
        try:
            with intent.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    {"schema": "telegram-intent-v1", "signal_id": signal_id}, handle
                )
                handle.write("\n")
        except FileExistsError:
            return TelegramDelivery(
                status="UNCERTAIN",
                message_id=None,
                detail="es zamanli gonderim denemesi",
            )
        try:
            message_id = sender()
        except TelegramRejected as error:
            # Telegram refused, so nothing reached the recipient. Drop the intent
            # so this signal is retried once the cause -- usually a missing
            # posting permission -- is fixed.
            intent.unlink(missing_ok=True)
            return TelegramDelivery(
                status="REDDEDILDI", message_id=None, detail=str(error)
            )
        except TelegramError as error:
            return TelegramDelivery(
                status="UNCERTAIN", message_id=None, detail=str(error)
            )
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

    def _direct_recipients(self) -> tuple[int, ...]:
        owner = self._owner_id
        if owner is None:
            return ()
        recipients = {owner}
        path = self._state_dir / "members.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return (owner,)
        if not isinstance(payload, dict) or payload.get("schema") != MEMBER_SCHEMA:
            return (owner,)
        for item in payload.get("members", []):
            if not isinstance(item, dict):
                continue
            identifier = item.get("id")
            if (
                isinstance(identifier, int)
                and not isinstance(identifier, bool)
                and identifier > 0
            ):
                recipients.add(identifier)
        return (owner, *sorted(recipients - {owner}))


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


def _command_payload(commands: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    return [
        {"command": command, "description": description}
        for command, description in commands
    ]


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
    "DELIVERY_MODE_ENV",
    "OWNER_ID_ENV",
    "ROLE_ENV",
    "TOKEN_ENV",
    "TelegramDelivery",
    "TelegramError",
    "TelegramNotifier",
    "TelegramRejected",
    "digest_signal_id",
    "is_primary",
    "telegram_channel_keyboard",
    "telegram_configured",
    "telegram_delivery_mode",
    "telegram_menu_keyboard",
]
