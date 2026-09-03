"""Answer questions asked in Telegram, instead of only pushing alerts.

Message text arriving from Telegram is untrusted input.  It is matched against
a fixed command table and never interpreted as an instruction, never evaluated,
and never used to reach anything outside this module's own read-only helpers.
No command can place, size, or authorise a trade, because the project has no
trading surface at all.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .telegram import (
    OWNER_ID_ENV,
    TelegramError,
    TelegramNotifier,
    telegram_menu_keyboard,
)

MEMBER_SCHEMA = "telegram-members-v1"
PENDING_MEMBER_SCHEMA = "telegram-pending-members-v1"
OFFSET_SCHEMA = "telegram-update-offset-v1"
MAXIMUM_UPDATES_PER_RUN = 20
MAXIMUM_NAME_LENGTH = 40
MAXIMUM_PENDING_MEMBERS = 200
_NAME_ALLOWED = re.compile(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü ._-]")
_COMMAND = re.compile(
    r"\A/([a-z_]{1,20})(?:@[A-Za-z0-9_]{1,32})?(?:\s+(.{0,200}))?\Z",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    received: int
    answered: int
    refused: int
    failed: int = 0
    detail: str = ""


def owner_id() -> int | None:
    raw = os.environ.get(OWNER_ID_ENV, "").strip()
    if not raw.lstrip("-").isdigit():
        return None
    value = int(raw)
    return value if value > 0 else None


def load_members(state_dir: Path) -> dict[int, str]:
    path = state_dir / "members.json"
    _harden_private_path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("schema") != MEMBER_SCHEMA:
        return {}
    members: dict[int, str] = {}
    for item in payload.get("members", []):
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier <= 0
        ):
            continue
        members[identifier] = safe_name(str(item.get("name", "")))
    return members


def save_members(state_dir: Path, members: dict[int, str]) -> None:
    payload = {
        "schema": MEMBER_SCHEMA,
        "members": [
            {"id": identifier, "name": members[identifier]}
            for identifier in sorted(members)
        ],
    }
    _write_private_json(state_dir / "members.json", payload)


def safe_name(value: str) -> str:
    cleaned = _NAME_ALLOWED.sub("", value).strip()
    return cleaned[:MAXIMUM_NAME_LENGTH] or "isimsiz"


def load_pending_members(state_dir: Path) -> dict[int, dict[str, object]]:
    path = state_dir / "pending_members.json"
    _harden_private_path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("schema") != PENDING_MEMBER_SCHEMA:
        return {}
    pending: dict[int, dict[str, object]] = {}
    for item in payload.get("requests", []):
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        if not _is_identifier(identifier):
            continue
        pending[int(identifier)] = {
            "name": safe_name(str(item.get("name", ""))),
            "requested_at": str(item.get("requested_at", ""))[:40],
        }
    return pending


def save_pending_members(
    state_dir: Path, pending: dict[int, dict[str, object]]
) -> None:
    payload = {
        "schema": PENDING_MEMBER_SCHEMA,
        "requests": [
            {
                "id": identifier,
                "name": safe_name(str(pending[identifier].get("name", ""))),
                "requested_at": str(pending[identifier].get("requested_at", ""))[:40],
            }
            for identifier in sorted(pending)
        ],
    }
    _write_private_json(state_dir / "pending_members.json", payload)


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _harden_private_path(path: Path) -> None:
    """Best-effort hardening for state created by older, less strict releases."""
    try:
        path.parent.chmod(0o700)
        if path.exists():
            path.chmod(0o600)
    except OSError:
        pass


def poll_and_answer(
    settings: Settings,
    *,
    status_text: Callable[[], str],
    performance_text: Callable[[int], str],
    scalp_performance_text: Callable[[int], str] | None = None,
    explanation_text: Callable[[], str] | None = None,
    reply_markup: dict[str, object] | None = None,
    notifier: TelegramNotifier | None = None,
    now: datetime | None = None,
) -> CommandOutcome:
    owner = owner_id()
    if owner is None:
        return CommandOutcome(received=0, answered=0, refused=0)
    state_dir = settings.telegram_state_dir
    client = notifier or TelegramNotifier(state_dir=state_dir)
    offset = _load_offset(state_dir)
    try:
        updates = client.get_updates(offset=offset, limit=MAXIMUM_UPDATES_PER_RUN)
    except TelegramError as error:
        return CommandOutcome(
            received=0, answered=0, refused=0, failed=1, detail=f"getUpdates: {error}"
        )
    members = load_members(state_dir)
    pending = load_pending_members(state_dir)
    current = now or datetime.now(UTC)
    answered = refused = failed = 0
    detail = ""
    highest = offset - 1
    for update in updates:
        update_id = update.get("update_id")
        if isinstance(update_id, int) and not isinstance(update_id, bool):
            highest = max(highest, update_id)
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            callback_id = callback.get("id")
            sender = callback.get("from")
            callback_message = callback.get("message")
            callback_chat = (
                callback_message.get("chat")
                if isinstance(callback_message, dict)
                else None
            )
            data = callback.get("data")
            sender_id = sender.get("id") if isinstance(sender, dict) else None
            chat_id = (
                callback_chat.get("id") if isinstance(callback_chat, dict) else None
            )
            chat_type = (
                callback_chat.get("type") if isinstance(callback_chat, dict) else None
            )
            if (
                not isinstance(callback_id, str)
                or not isinstance(data, str)
                or not _is_identifier(sender_id)
                or not _is_identifier(chat_id)
            ):
                continue
            if chat_type != "private" or int(chat_id) != int(sender_id):
                refused += 1
                try:
                    client.answer_callback_query(
                        callback_id, "Dugmeler yalniz botun ozel sohbetinde calisir."
                    )
                except (TelegramError, ValueError) as error:
                    failed += 1
                    detail = str(error)
                continue
            if (
                data == "request_access"
                and sender_id != owner
                and sender_id not in members
            ):
                is_new, request_name = _register_access_request(
                    state_dir, sender, pending=pending, now=current
                )
                try:
                    client.answer_callback_query(callback_id, "Istegin alindi.")
                    _send_reply(
                        client,
                        int(chat_id),
                        _access_request_received_text(is_new),
                        reply_markup=_access_request_keyboard(),
                    )
                    answered += 1
                    if is_new:
                        _send_owner_access_request(
                            client, owner, int(sender_id), request_name
                        )
                except (TelegramError, ValueError) as error:
                    failed += 1
                    detail = str(error)
                continue
            if data.startswith(("approve_member:", "decline_member:")):
                if sender_id != owner:
                    refused += 1
                    try:
                        client.answer_callback_query(
                            callback_id, "Bu islem yalnizca bot sahibine acik."
                        )
                    except (TelegramError, ValueError) as error:
                        failed += 1
                        detail = str(error)
                    continue
                action, _, raw_target = data.partition(":")
                target = int(raw_target) if raw_target.isdigit() else 0
                decision, target_message = _decide_access_request(
                    state_dir,
                    target,
                    approve=action == "approve_member",
                    members=members,
                    pending=pending,
                )
                try:
                    client.answer_callback_query(callback_id, decision[:200])
                    _send_reply(
                        client,
                        owner,
                        decision,
                        reply_markup=telegram_menu_keyboard(is_owner=True),
                    )
                    answered += 1
                    if target_message is not None and target > 0:
                        _send_reply(
                            client,
                            target,
                            target_message,
                            reply_markup=telegram_menu_keyboard(is_owner=False),
                        )
                except (TelegramError, ValueError) as error:
                    failed += 1
                    detail = str(error)
                continue
            if sender_id != owner and sender_id not in members:
                refused += 1
                try:
                    client.answer_callback_query(callback_id, "Yetkiniz yok.")
                except (TelegramError, ValueError) as error:
                    failed += 1
                    detail = str(error)
                continue
            try:
                client.answer_callback_query(callback_id)
            except (TelegramError, ValueError) as error:
                failed += 1
                detail = str(error)
            try:
                reply = _answer_callback(
                    data,
                    sender_id=int(sender_id),
                    owner=owner,
                    members=members,
                    pending=pending,
                    status_text=status_text,
                    performance_text=performance_text,
                    scalp_performance_text=scalp_performance_text,
                    explanation_text=explanation_text or format_explanations,
                )
            except Exception as error:  # a bad answer must not kill the cycle
                failed += 1
                detail = f"dugme yaniti uretilemedi: {error}"
                reply = "⚠️ Bilgi su an uretilemedi. Sunucu gunlugunde ayrinti var."
            if reply is None:
                continue
            try:
                _send_reply(
                    client,
                    int(chat_id),
                    reply,
                    reply_markup=telegram_menu_keyboard(is_owner=sender_id == owner),
                )
                answered += 1
            except (TelegramError, ValueError) as error:
                failed += 1
                detail = str(error)
            continue
        message = update.get("message")
        if not isinstance(message, dict):
            continue
        sender = message.get("from")
        chat = message.get("chat")
        text = message.get("text")
        if (
            not isinstance(sender, dict)
            or not isinstance(chat, dict)
            or not isinstance(text, str)
        ):
            continue
        sender_id = sender.get("id")
        chat_id = chat.get("id")
        chat_type = chat.get("type")
        if not _is_identifier(sender_id) or not _is_identifier(chat_id):
            continue
        if chat_type != "private" or int(chat_id) != int(sender_id):
            refused += 1
            continue
        if sender_id != owner and sender_id not in members:
            command = _command_name(text)
            if command in {"start", "help", "yardim", "myid", "kimlik"}:
                public_reply = _public_access_text(int(sender_id))
                try:
                    _send_reply(
                        client,
                        int(chat_id),
                        public_reply,
                        reply_markup=_access_request_keyboard(),
                    )
                    answered += 1
                except (TelegramError, ValueError) as error:
                    failed += 1
                    detail = str(error)
                continue
            if command in {"katil", "join"}:
                is_new, request_name = _register_access_request(
                    state_dir, sender, pending=pending, now=current
                )
                try:
                    _send_reply(
                        client,
                        int(chat_id),
                        _access_request_received_text(is_new),
                        reply_markup=_access_request_keyboard(),
                    )
                    answered += 1
                    if is_new:
                        _send_owner_access_request(
                            client, owner, int(sender_id), request_name
                        )
                except (TelegramError, ValueError) as error:
                    failed += 1
                    detail = str(error)
                continue
            refused += 1
            continue
        try:
            reply = _answer(
                text,
                sender_id=int(sender_id),
                owner=owner,
                members=members,
                state_dir=state_dir,
                status_text=status_text,
                performance_text=performance_text,
                scalp_performance_text=scalp_performance_text,
                explanation_text=explanation_text or format_explanations,
                now=current,
            )
        except Exception as error:  # a bad answer must not kill the cycle
            failed += 1
            detail = f"yanit uretilemedi: {error}"
            reply = "⚠️ Durum raporu su an uretilemedi. Sunucu gunlugunde ayrinti var."
        if reply is None:
            continue
        try:
            _send_reply(
                client,
                int(chat_id),
                reply,
                reply_markup=telegram_menu_keyboard(is_owner=sender_id == owner),
            )
            answered += 1
        except (TelegramError, ValueError) as error:
            # Silence here used to consume the command and leave no trace, so a
            # broken /durum looked identical to one that was never sent.
            failed += 1
            detail = str(error)
    if highest >= offset:
        _save_offset(state_dir, highest + 1)
    return CommandOutcome(
        received=len(updates),
        answered=answered,
        refused=refused,
        failed=failed,
        detail=detail,
    )


def _command_name(text: str) -> str:
    match = _COMMAND.match(text.strip())
    return match.group(1) if match is not None else ""


def _public_access_text(identifier: int) -> str:
    return "\n".join(
        [
            "🔐 TRADE3 ÖZEL ERİŞİM",
            "",
            "Bu bot sinyalleri onaylı kişilere ayrı ayrı özel mesajla gönderir.",
            "Kullanıcılar birbirini göremez ve üyelik kararını yalnız bot sahibi verir.",
            "",
            f"Telegram kimliğiniz: {identifier}",
            "Erişim istemek için aşağıdaki düğmeye dokunun veya /katil yazın.",
        ]
    )


def _access_request_keyboard() -> dict[str, object]:
    return {
        "inline_keyboard": [
            [{"text": "📨 Erişim İste", "callback_data": "request_access"}]
        ]
    }


def _approval_keyboard(identifier: int) -> dict[str, object]:
    return {
        "inline_keyboard": [
            [
                {
                    "text": "✅ Onayla",
                    "callback_data": f"approve_member:{identifier}",
                },
                {
                    "text": "❌ Reddet",
                    "callback_data": f"decline_member:{identifier}",
                },
            ]
        ]
    }


def _display_name(sender: dict[str, Any]) -> str:
    first = str(sender.get("first_name", "")).strip()
    last = str(sender.get("last_name", "")).strip()
    username = str(sender.get("username", "")).strip()
    label = " ".join(part for part in (first, last) if part)
    if username:
        label = f"{label} @{username}".strip()
    return safe_name(label)


def _register_access_request(
    state_dir: Path,
    sender: dict[str, Any],
    *,
    pending: dict[int, dict[str, object]],
    now: datetime,
) -> tuple[bool, str]:
    identifier = sender.get("id")
    if not _is_identifier(identifier):
        raise ValueError("Gecersiz Telegram kimligi")
    numeric_id = int(identifier)
    name = _display_name(sender)
    is_new = numeric_id not in pending
    if is_new and len(pending) >= MAXIMUM_PENDING_MEMBERS:
        raise ValueError("Bekleyen uyelik siniri dolu")
    pending[numeric_id] = {
        "name": name,
        "requested_at": now.astimezone(UTC).isoformat(),
    }
    save_pending_members(state_dir, pending)
    return is_new, name


def _access_request_received_text(is_new: bool) -> str:
    if is_new:
        return (
            "📨 Erişim isteğiniz bot sahibine iletildi. Yalnızca sahibi onaylayabilir."
        )
    return "ℹ️ Erişim isteğiniz zaten bekliyor. Tekrar bildirim gönderilmedi."


def _send_owner_access_request(
    client: TelegramNotifier, owner: int, identifier: int, name: str
) -> None:
    _send_reply(
        client,
        owner,
        "\n".join(
            [
                "📨 YENİ ERİŞİM İSTEĞİ",
                "",
                f"Kişi: {name}",
                f"Telegram kimliği: {identifier}",
                "",
                "Karar yalnızca sana aittir.",
            ]
        ),
        reply_markup=_approval_keyboard(identifier),
    )


def _decide_access_request(
    state_dir: Path,
    identifier: int,
    *,
    approve: bool,
    members: dict[int, str],
    pending: dict[int, dict[str, object]],
) -> tuple[str, str | None]:
    request = pending.get(identifier)
    if request is None:
        return "ℹ️ Bu kimlik için bekleyen erişim isteği yok.", None
    name = safe_name(str(request.get("name", "")))
    pending.pop(identifier, None)
    save_pending_members(state_dir, pending)
    if not approve:
        return (
            f"❌ {name} ({identifier}) reddedildi. Erişim verilmedi.",
            "❌ Erişim isteğiniz bot sahibi tarafından reddedildi.",
        )
    members[identifier] = name
    save_members(state_dir, members)
    return (
        f"✅ {name} ({identifier}) onaylandı. Sinyaller artık özel mesajla gidecek.",
        "\n".join(
            [
                "✅ Erişiminiz onaylandı.",
                "Yeni sinyaller size ayrı ayrı özel mesaj olarak gelecek.",
                "Komutlar ve menü için /start yazabilirsiniz.",
            ]
        ),
    )


def _send_reply(
    client: TelegramNotifier,
    chat_id: int,
    text: str,
    *,
    reply_markup: dict[str, object] | None,
) -> int:
    if reply_markup is None:
        return client.send_reply(chat_id, text)
    return client.send_reply(chat_id, text, reply_markup=reply_markup)


def _answer(
    text: str,
    *,
    sender_id: int,
    owner: int,
    members: dict[int, str],
    state_dir: Path,
    status_text: Callable[[], str],
    performance_text: Callable[[int], str],
    scalp_performance_text: Callable[[int], str] | None,
    explanation_text: Callable[[], str],
    now: datetime,
) -> str | None:
    match = _COMMAND.match(text.strip())
    if match is None:
        return None
    command = match.group(1)
    argument = (match.group(2) or "").strip()
    if command in ("yardim", "start", "help"):
        return _help_text(is_owner=sender_id == owner)
    if command in ("aciklamalar", "explanations", "strateji"):
        return explanation_text()
    if command in ("durum", "status"):
        return status_text()
    if command in ("performans", "karne"):
        return performance_text(_days_argument(argument))
    if command in ("scalpkarne", "scalp_performance"):
        if scalp_performance_text is None:
            return "ℹ️ Scalp karnesi henuz etkin degil."
        return scalp_performance_text(_days_argument(argument))
    if command in ("kisiler", "members"):
        if sender_id != owner:
            return "🔒 Abone listesi yalnızca bot sahibine açıktır."
        return _member_list(owner, members)
    if command in ("bekleyenler", "pending"):
        if sender_id != owner:
            return "🔒 Bekleyen istekler yalnızca bot sahibine açıktır."
        return _pending_member_list(load_pending_members(state_dir))
    if command in ("onayla", "reddet"):
        if sender_id != owner:
            return "🔒 Bu komutu yalnızca bot sahibi kullanabilir."
        token = argument.split()[0] if argument.split() else ""
        if not token.isdigit():
            return f"❗ Kullanım: /{command} <Telegram kimliği>"
        decision, _ = _decide_access_request(
            state_dir,
            int(token),
            approve=command == "onayla",
            members=members,
            pending=load_pending_members(state_dir),
        )
        return decision
    if command in ("ekle", "sil"):
        if sender_id != owner:
            return "🔒 Bu komutu yalnızca bot sahibi kullanabilir."
        return _change_members(command, argument, members=members, state_dir=state_dir)
    return "❓ Bilinmeyen komut. /yardim yazin."


def _answer_callback(
    data: str,
    *,
    sender_id: int,
    owner: int,
    members: dict[int, str],
    pending: dict[int, dict[str, object]],
    status_text: Callable[[], str],
    performance_text: Callable[[int], str],
    scalp_performance_text: Callable[[int], str] | None,
    explanation_text: Callable[[], str],
) -> str | None:
    if data in ("start", "help"):
        return _help_text(is_owner=sender_id == owner)
    if data == "explanations":
        return explanation_text()
    if data == "status":
        return status_text()
    if data == "members":
        if sender_id != owner:
            return "🔒 Abone listesi yalnızca bot sahibine açıktır."
        return _member_list(owner, members)
    if data == "pending_members":
        if sender_id != owner:
            return "🔒 Bekleyen istekler yalnızca bot sahibine açıktır."
        return _pending_member_list(pending)
    if data.startswith("performance:"):
        return performance_text(_days_argument(data.partition(":")[2]))
    if data.startswith("scalp_performance:"):
        if scalp_performance_text is None:
            return "ℹ️ Scalp karnesi henuz etkin degil."
        return scalp_performance_text(_days_argument(data.partition(":")[2]))
    return "❓ Bilinmeyen dugme. /yardim yazin."


def _change_members(
    command: str, argument: str, *, members: dict[int, str], state_dir: Path
) -> str:
    parts = argument.split(maxsplit=1)
    if not parts or not parts[0].isdigit():
        return (
            "❗ Kullanim: /ekle 123456789 Ad Soyad\n"
            "Kisinin sayisal Telegram kimligi gerekir; kullanici adi yeterli degildir."
        )
    identifier = int(parts[0])
    if identifier <= 0 or identifier > 10**19:
        return "❗ Gecersiz Telegram kimligi."
    if command == "sil":
        if members.pop(identifier, None) is None:
            return f"ℹ️ {identifier} zaten yetkili listesinde degil."
        save_members(state_dir, members)
        return f"✅ {identifier} abonelikten çıkarıldı. Kalan abone: {len(members)}."
    name = safe_name(parts[1]) if len(parts) > 1 else "isimsiz"
    members[identifier] = name
    save_members(state_dir, members)
    return (
        f"✅ {name} ({identifier}) eklendi. Sinyaller artık özel mesajla gidecek.\n"
        f"📋 Toplam abone: {len(members)}.\n\n"
        "⚠️ Bot yalnız bildirim ve sorgulama sağlar; emir vermez."
    )


def _member_list(owner: int, members: dict[int, str]) -> str:
    lines = ["👥 ÖZEL MESAJ ABONELERİ", "", f"👑 Sahip: {owner}"]
    if not members:
        lines.append("ℹ️ Başka onaylı abone yok.")
        lines.append("Eklemek icin: /ekle <kimlik> <ad>")
    else:
        lines.append("")
        lines.append("📋 Onaylı aboneler")
        for identifier in sorted(members):
            lines.append(f"• {members[identifier]} — {identifier}")
    return "\n".join(lines)


def _pending_member_list(pending: dict[int, dict[str, object]]) -> str:
    lines = ["📨 BEKLEYEN ERİŞİM İSTEKLERİ", ""]
    if not pending:
        lines.append("ℹ️ Bekleyen istek yok.")
        return "\n".join(lines)
    for identifier in sorted(pending):
        name = safe_name(str(pending[identifier].get("name", "")))
        lines.append(f"• {name} — {identifier}")
    lines.extend(
        [
            "",
            "Karar vermek için istek mesajındaki düğmeleri kullanın veya:",
            "/onayla <kimlik> · /reddet <kimlik>",
        ]
    )
    return "\n".join(lines)


def _help_text(*, is_owner: bool) -> str:
    lines = [
        "🤖 TRADE3 ARAŞTIRMA BOTU",
        "",
        "📌 KOMUTLAR",
        "📊 /durum — alti modelin su anki durumu",
        "📈 /performans [gun] — gonderilen sinyallerin gercek sonucu (varsayilan 30 gun)",
        "🧪 /scalpkarne [gun] — scalp ileri-test sonuclari (varsayilan 30 gun)",
        "📖 /aciklamalar — bildirim alanlari ve strateji mantigi",
        "🏠 /yardim — bu liste",
    ]
    if is_owner:
        lines.extend(
            [
                "",
                "🔐 SAHİP KOMUTLARI",
                "👥 /kisiler — yalnız sana görünen abone listesi",
                "📨 /bekleyenler — erişim talepleri",
                "✅ /onayla <kimlik> — talebi onayla",
                "❌ /reddet <kimlik> — talebi reddet",
                "➕ /ekle <kimlik> <ad> — doğrudan abone ekle",
                "➖ /sil <kimlik> — aboneliği kaldır",
            ]
        )
    lines.extend(
        [
            "",
            "⚠️ Bot emir vermez, borsa hesabina baglanmaz ve yatirim tavsiyesi vermez.",
        ]
    )
    return "\n".join(lines)


def format_explanations() -> str:
    """Explain every alert field and the research-only scalp hypotheses."""
    return "\n".join(
        [
            "📖 ACIKLAMALAR — bildirimleri nasil okumali?",
            "",
            "🧩 SINYAL ALANLARI",
            "• Sinyal fiyati: kosulun olustugu kapanmis 5m mum fiyati; guncel fiyat degildir.",
            "• Guncel mark: bildirim anindaki vadeli referans fiyati; sinyal fiyatindan farkli olabilir.",
            "• Aile: sinyali ureten teknik gozlem turu.",
            "• Skor: kosulun gucu; olasilik veya kar tahmini degildir.",
            "• Tetikleyici: gozlemin neden olustugunu aciklar.",
            "• Beklenen ufuk: scalp ileri-testinin sabit 15/30/60 dakika cikislari.",
            "• Piyasa: bildirimlerin olculdugu Binance USD-M vadeli kontrati.",
            "• 24s kapali mum getirisi/sirasi: son kapanmis 24 saatin evrendeki yeri.",
            "• 1s hacim / onceki 24s medyani: son saatin hacmi, onceki saatlerin tipik hacmine gore.",
            "• Funding: vadeli kontratta long-short arasindaki periyodik odeme; pozitifse long taraf oder.",
            "• Spread: en iyi alis-satis farki; dusuk olmasi daha sagliklidir.",
            "• Maliyet: komisyon + spread + tahmini kayma dahil gidis-donus maliyeti.",
            "  1 bps = %0,01; maliyet sonrasi hareket pozitif degilse avantaj yoktur.",
            "• %2/%3 hedef bildirimi: sinyal fiyatindan, YUKARI icin +; ASAGI icin - yonunde hesaplanir.",
            "  Her kademe sinyal basina bir kez bildirilir; hedefe ulasmak garanti veya emir degildir.",
            "",
            "🧪 SCALP AILELERI (ARASTIRMA)",
            "• F1 hacim momentumu: yukari mum + log-hacim z en az 3.",
            "• F2 kaskad dusus: 30dk getiri en az 3 sigma asagi + hacim z en az 2.",
            "• F3 kirilim devami: onceki 12s zirve ustu kapanis + hacim z en az 2.",
            "• B1 boga kirilimi: boga rejiminde onceki 24s zirve ustu + hacim z en az 1.",
            "• B2 geri cekilme donusu: trend icinde kontrollu dusus sonrasi toparlanma.",
            "• B3 goreli guc gecisi: 24s getiride ilk %10'a yeni giris + kisa donem yukari.",
            "",
            "🧭 REJIM VE ETIKET",
            "• BOGA: BTC/ETH trendi ve piyasalarin genis bolumu yukari.",
            "• GECIS: kosullar karisik; KAPALI: boga kosulu yok.",
            "• Genislik: 89 piyasanin 48 saatlik trend filtresinin ustunde olan orani.",
            "• RADAR: tek aile goruldu; KURULUM: ayni coinde coklu aile + boga rejimi + uygun spread.",
            "  Ikisi de islem emri veya yatirim tavsiyesi degildir.",
            "",
            "📊 BT 15/30/60dk",
            "• Yalnizca kapanmis ileri-test sonuclaridir; her ufuk ayri hesaplanir.",
            "• Yukari/asagi olasiligi: gecmiste fiyat hangi yonde hareket etti.",
            "• Medyan hareket: tipik brut hareket (bps ve %).",
            "• Medyan net hareket: tahmini maliyet cikarildiktan sonraki tipik hareket.",
            "• Yon ozeti: o coindeki ailelerin yerlesmis BT sonuclarinin n agirlikli sentezi.",
            "  15/30/60dk ayri okunur; KARIŞIK, gecmis verinin net bir yon vermedigini anlatir.",
            "• n: hesaba giren sonuc sayisi; n dusukse belirsizlik yuksektir.",
            "",
            "🤖 BTC/ETH MODEL MESAJLARI",
            "• Yukari/asagi yuzdeleri: kalibre edilmis model olasiligi.",
            "• Medyan kapanis: benzer gecmis durumlarin tahmini kapanis seviyesi.",
            "• Net beklenti ve isabet: tamamen dis-ornek walk-forward backtest metrikleri.",
            "",
            "⚠️ Bot emir vermez; tum scalp bildirimleri gozlem ve arastirma amaclidir.",
        ]
    )


def _days_argument(argument: str) -> int:
    token = argument.split()[0] if argument.split() else ""
    if token.isdigit():
        return max(1, min(int(token), 365))
    return 30


def _is_identifier(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value != 0


def _load_offset(state_dir: Path) -> int:
    path = state_dir / "update_offset.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict) or payload.get("schema") != OFFSET_SCHEMA:
        return 0
    offset = payload.get("offset")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return 0
    return offset


def _save_offset(state_dir: Path, offset: int) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "update_offset.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"schema": OFFSET_SCHEMA, "offset": int(offset)}) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "CommandOutcome",
    "format_explanations",
    "load_members",
    "load_pending_members",
    "owner_id",
    "poll_and_answer",
    "safe_name",
    "save_members",
    "save_pending_members",
]
