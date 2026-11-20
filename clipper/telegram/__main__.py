"""Telegram clip inbox CLI.

python -m clipper.telegram link            once: after sending /start to the bot
python -m clipper.telegram bot             long polling while the Mac is on
python -m clipper.telegram poll            handle pending button presses / replies once
python -m clipper.telegram deliver [-n N]  send rendered clips now
python -m clipper.telegram refresh         re-edit the captions of clips still waiting
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from clipper import db
from clipper.config import REPO_ROOT, load_config
from clipper.proc import AlreadyRunning, run_lock
from clipper.telegram.api import BotAPI, TelegramError, get_token
from clipper.telegram.inbox import (
    BOT_LOCK,
    HELP,
    OFFSET_KEY,
    Inbox,
    deliver,
    find_start,
    open_inbox,
    poll,
    refresh_captions,
    write_chat_id,
)

log = logging.getLogger("clipper.telegram")

LONG_POLL_SEC = 50


def link(config_path: Path) -> int:
    cfg = load_config(config_path)
    api = BotAPI(get_token(cfg.telegram.keychain_service))
    me = api.call("getMe")
    updates = api.call("getUpdates", {"timeout": 0, "allowed_updates": ["message"]})
    chat = find_start(updates)
    if chat is None:
        print(f"No /start found. In Telegram, open @{me['username']}, send /start, "
              "then run this again.")  # fmt: skip
        return 1
    write_chat_id(config_path, int(chat["id"]))
    conn = db.connect(load_config(config_path).data_dir)
    try:
        db.set_setting(conn, OFFSET_KEY, str(max(u["update_id"] for u in updates) + 1))
    finally:
        conn.close()
    api.call(
        "sendMessage",
        {"chat_id": chat["id"], "parse_mode": "HTML",
         "text": "✅ Linked. New clips will arrive here.\n\n" + HELP},
    )  # fmt: skip
    who = chat.get("username") or chat.get("first_name") or "?"
    print(f"Linked @{me['username']} to chat {chat['id']} ({who}); saved in {config_path.name}.")
    return 0


def run_bot(ix: Inbox) -> int:
    log.info("bot polling (Ctrl-C to stop)")
    backoff = 5.0
    while True:
        try:
            poll(ix, LONG_POLL_SEC)
            backoff = 5.0
        except TelegramError as exc:
            if exc.code == 409:  # another getUpdates (e.g. a pipeline run) is polling
                log.info("another poller is active; waiting")
                time.sleep(10)
            else:
                log.warning("%s; retrying in %.0f s", exc, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)


def _on_sigterm(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clipper.telegram", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("link", help="link the chat that sent /start")
    sub.add_parser("bot", help="long polling: buttons, replies, commands")
    sub.add_parser("poll", help="handle pending updates once")
    send = sub.add_parser("deliver", help="send rendered clips now")
    send.add_argument("-n", "--limit", type=int, help="max clips (default telegram.max_per_run)")
    sub.add_parser("refresh", help="re-edit captions of clips still waiting (new format)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    # the bot runs around the clock: rotate its log; echo to the terminal only if there is one
    handlers: list[logging.Handler] = [
        RotatingFileHandler(cfg.dir("logs") / "telegram.log", maxBytes=5_000_000, backupCount=3)
    ]
    if sys.stderr.isatty():
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        if args.command == "link":
            return link(args.config)
        conn = db.connect(cfg.data_dir)
        try:
            ix = open_inbox(cfg, conn)
            if ix is None:
                log.error("telegram is not enabled/linked (see config.yaml telegram:)")
                return 1
            if args.command == "poll":
                log.info("handled %d update(s)", poll(ix, 0))
                return 0
            if args.command == "refresh":
                edited, failed = refresh_captions(ix)
                log.info("refreshed %d message(s), %d failed", edited, failed)
                return 1 if failed else 0
            if args.command == "deliver":
                with run_lock(cfg.data_dir):  # a re-encode is a heavy step: not during a run
                    sent, failed = deliver(ix, args.limit or cfg.telegram.max_per_run)
                log.info("delivered %d clip(s), %d failed", sent, failed)
                return 1 if failed else 0
            with run_lock(cfg.data_dir, BOT_LOCK):
                return run_bot(ix)
        finally:
            conn.close()
    except AlreadyRunning as exc:
        log.error("not starting: %s", exc)
        return 4
    except TelegramError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
