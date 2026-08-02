"""Answer questions asked in Telegram, instead of only pushing alerts.

Message text arriving from Telegram is untrusted input.  It is matched against
a fixed command table and never interpreted as an instruction, never evaluated,
and never used to reach anything outside this module's own read-only helpers.
No command can place, size, or authorise a trade, because the project has no
trading surface at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

from .config import Settings
from .telegram import OWNER_ID_ENV, TelegramError, TelegramNotifier


MEMBER_SCHEMA = "telegram-members-v1"
OFFSET_SCHEMA = "telegram-update-offset-v1"
MAXIMUM_UPDATES_PER_RUN = 20
MAXIMUM_NAME_LENGTH = 40
_NAME_ALLOWED = re.compile(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü ._-]")
_COMMAND = re.compile(r"\A/([a-z_]{1,20})(?:@[A-Za-z0-9_]{1,32})?(?:\s+(.{0,200}))?\Z", re.S)


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
        if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier <= 0:
            continue
        members[identifier] = safe_name(str(item.get("name", "")))
    return members


def save_members(state_dir: Path, members: dict[int, str]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MEMBER_SCHEMA,
        "members": [
            {"id": identifier, "name": members[identifier]} for identifier in sorted(members)
        ],
    }
    path = state_dir / "members.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_name(value: str) -> str:
    cleaned = _NAME_ALLOWED.sub("", value).strip()
    return cleaned[:MAXIMUM_NAME_LENGTH] or "isimsiz"


def poll_and_answer(
    settings: Settings,
    *,
    status_text: Callable[[], str],
    performance_text: Callable[[int], str],
    notifier: TelegramNotifier | None = None,
    now: datetime | None = None,
) -> CommandOutcome:
    owner = owner_id()
    if owner is None:
        return CommandOutcome(received=0, answered=0, refused=0)
    client = notifier or TelegramNotifier()
    state_dir = settings.telegram_state_dir
    offset = _load_offset(state_dir)
    try:
        updates = client.get_updates(offset=offset, limit=MAXIMUM_UPDATES_PER_RUN)
    except TelegramError as error:
        return CommandOutcome(
            received=0, answered=0, refused=0, failed=1, detail=f"getUpdates: {error}"
        )
    members = load_members(state_dir)
    answered = refused = failed = 0
    detail = ""
    highest = offset - 1
    for update in updates:
        update_id = update.get("update_id")
        if isinstance(update_id, int) and not isinstance(update_id, bool):
            highest = max(highest, update_id)
        message = update.get("message")
        if not isinstance(message, dict):
            continue
        sender = message.get("from")
        chat = message.get("chat")
        text = message.get("text")
        if not isinstance(sender, dict) or not isinstance(chat, dict) or not isinstance(text, str):
            continue
        sender_id = sender.get("id")
        chat_id = chat.get("id")
        if not _is_identifier(sender_id) or not _is_identifier(chat_id):
            continue
        if sender_id != owner and sender_id not in members:
            # Silence, not a refusal message: an unknown sender must not be
            # able to make the bot emit traffic on demand.
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
                now=now or datetime.now(timezone.utc),
            )
        except Exception as error:  # a bad answer must not kill the cycle
            failed += 1
            detail = f"yanit uretilemedi: {error}"
            reply = "Durum raporu su an uretilemedi. Sunucu gunlugunde ayrinti var."
        if reply is None:
            continue
        try:
            client.send_reply(int(chat_id), reply)
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


def _answer(
    text: str,
    *,
    sender_id: int,
    owner: int,
    members: dict[int, str],
    state_dir: Path,
    status_text: Callable[[], str],
    performance_text: Callable[[int], str],
    now: datetime,
) -> str | None:
    match = _COMMAND.match(text.strip())
    if match is None:
        return None
    command = match.group(1)
    argument = (match.group(2) or "").strip()
    if command in ("yardim", "start", "help"):
        return _help_text(is_owner=sender_id == owner)
    if command in ("durum", "status"):
        return status_text()
    if command in ("performans", "karne"):
        return performance_text(_days_argument(argument))
    if command in ("kisiler", "members"):
        return _member_list(owner, members)
    if command in ("ekle", "sil"):
        if sender_id != owner:
            return "Bu komutu yalnizca kanal sahibi kullanabilir."
        return _change_members(command, argument, members=members, state_dir=state_dir)
    return "Bilinmeyen komut. /yardim yazin."


def _change_members(
    command: str, argument: str, *, members: dict[int, str], state_dir: Path
) -> str:
    parts = argument.split(maxsplit=1)
    if not parts or not parts[0].isdigit():
        return (
            "Kullanim: /ekle 123456789 Ad Soyad\n"
            "Kisinin sayisal Telegram kimligi gerekir; kullanici adi yeterli degildir."
        )
    identifier = int(parts[0])
    if identifier <= 0 or identifier > 10**19:
        return "Gecersiz Telegram kimligi."
    if command == "sil":
        if members.pop(identifier, None) is None:
            return f"{identifier} zaten yetkili listesinde degil."
        save_members(state_dir, members)
        return f"{identifier} listeden cikarildi. Kalan yetkili: {len(members)}."
    name = safe_name(parts[1]) if len(parts) > 1 else "isimsiz"
    members[identifier] = name
    save_members(state_dir, members)
    return (
        f"{name} ({identifier}) eklendi. Artik bota /durum ve /performans sorabilir.\n"
        f"Toplam yetkili: {len(members)}.\n\n"
        "Not: bu yetki yalnizca sorgulama icindir; bot emir vermez."
    )


def _member_list(owner: int, members: dict[int, str]) -> str:
    lines = ["👥 YETKILI KISILER", "", f"Sahip: {owner}"]
    if not members:
        lines.append("Baska yetkili kisi yok. Eklemek icin: /ekle <kimlik> <ad>")
    else:
        for identifier in sorted(members):
            lines.append(f"• {members[identifier]} — {identifier}")
    return "\n".join(lines)


def _help_text(*, is_owner: bool) -> str:
    lines = [
        "🤖 BTC/ETH olasilik botu — komutlar",
        "",
        "/durum — alti modelin su anki durumu",
        "/performans [gun] — gonderilen sinyallerin gercek sonucu (varsayilan 30 gun)",
        "/kisiler — yetkili kisiler",
        "/yardim — bu liste",
    ]
    if is_owner:
        lines.extend(
            [
                "",
                "Sahip komutlari:",
                "/ekle <kimlik> <ad> — sorgulama yetkisi ver",
                "/sil <kimlik> — yetkiyi kaldir",
            ]
        )
    lines.extend(
        [
            "",
            "Bot emir vermez, borsa hesabina baglanmaz ve yatirim tavsiyesi vermez.",
        ]
    )
    return "\n".join(lines)


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
        json.dumps({"schema": OFFSET_SCHEMA, "offset": int(offset)}) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


__all__ = [
    "CommandOutcome",
    "load_members",
    "owner_id",
    "poll_and_answer",
    "safe_name",
    "save_members",
]
