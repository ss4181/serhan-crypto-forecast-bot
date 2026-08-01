from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from crypto_forecaster.commands import load_members, poll_and_answer, safe_name, save_members
from crypto_forecaster.config import Settings


OWNER = 500100
MEMBER = 700200
STRANGER = 900300


class FakeNotifier:
    def __init__(self, updates: list[dict]) -> None:
        self._updates = updates
        self.sent: list[tuple[int, str]] = []
        self.offsets: list[int] = []

    def get_updates(self, *, offset: int, limit: int = 20) -> list[dict]:
        self.offsets.append(offset)
        return [item for item in self._updates if item["update_id"] >= offset][:limit]

    def send_reply(self, chat_id: int, text: str) -> int:
        self.sent.append((chat_id, text))
        return len(self.sent)


def update(update_id: int, sender: int, text: str, chat: int | None = None) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "from": {"id": sender},
            "chat": {"id": chat if chat is not None else sender},
            "text": text,
        },
    }


def run(directory: Path, updates: list[dict], members: dict[int, str] | None = None):
    settings = Settings(telegram_state_dir=directory, outcome_state_dir=directory / "outcomes")
    if members:
        save_members(directory, members)
    notifier = FakeNotifier(updates)
    outcome = poll_and_answer(
        settings,
        status_text=lambda: "DURUM METNI",
        performance_text=lambda days: f"KARNE {days} GUN",
        notifier=notifier,
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

    def test_performance_defaults_to_thirty_days(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, notifier, _ = run(
                Path(directory), [update(1, OWNER, "/performans")], members={}
            )
        self.assertEqual(notifier.sent[0][1], "KARNE 30 GUN")

    def test_only_the_owner_may_add_people(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome, notifier, _ = run(
                root, [update(1, MEMBER, f"/ekle {STRANGER} Yeni")], members={MEMBER: "Ortak"}
            )
            stored = load_members(root)
        self.assertEqual(outcome.answered, 1)
        self.assertIn("yalnizca kanal sahibi", notifier.sent[0][1])
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
            _, notifier, _ = run(Path(directory), [update(1, OWNER, "/ekle @kullanici")])
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
            _, member_view, _ = run(root, [update(2, MEMBER, "/yardim")], members={MEMBER: "O"})
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


if __name__ == "__main__":
    unittest.main()
