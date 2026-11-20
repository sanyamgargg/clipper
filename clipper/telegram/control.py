"""On-demand runs from Telegram (PLAN.md §17): a pinned control panel with ▶️ Start /
⏹ Stop / 📊 Status, and /run, /stop, /status.

The bot starts the pipeline as a detached child (its own process group) through
scripts/run_pipeline.sh, which loads the Claude token and runs under caffeinate -i. The
run writes data/run.status.json; the bot turns it into the panel text and edits the
pinned message when it changes. Only one run at a time (the run lock).

- Stop: data/run.stop; the run finishes the clip it is rendering, delivers it, and exits.
- Stop now: SIGTERM to the run's process group (the run then kills its whisper/ffmpeg
  child), SIGKILL if it is still there after a grace period.
"""

from __future__ import annotations

import html
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from clipper import db, runstate
from clipper.config import REPO_ROOT, Config
from clipper.proc import LOCK_NAME, lock_held
from clipper.telegram.api import TelegramError
from clipper.telegram.inbox import Inbox

log = logging.getLogger(__name__)

SCRIPT = REPO_ROOT / "scripts" / "run_pipeline.sh"
PANEL_KEY = "panel_message_id"
PANEL_TEXT_KEY = "panel_text"
PGID_KEY = "run_pgid"
STOP_NOW_GRACE_SEC = 15
LID_NOTE = "💡 Keep the lid open: caffeinate stops idle sleep, but closing the lid stops the run."

_proc: subprocess.Popen | None = None  # the run this bot process started


def run_active(cfg: Config) -> bool:
    """A run holds the lock, or the one we just started is still coming up."""
    return lock_held(cfg.data_dir) or (_proc is not None and _proc.poll() is None)


# --- panel ------------------------------------------------------------------------------


def _local(ts: str | None) -> str:
    if not ts:
        return "?"
    return datetime.fromisoformat(ts).astimezone().strftime("%d %b %H:%M")


def _minutes_since(ts: str | None) -> int:
    if not ts:
        return 0
    started = datetime.fromisoformat(ts)
    return int((datetime.now(started.tzinfo) - started).total_seconds() // 60)


def panel_text(cfg: Config, status: dict[str, Any], running: bool, stopping: bool) -> str:
    esc = html.escape
    clips, cost = status.get("clips", 0), status.get("cost_usd") or 0.0
    if running:
        head = "⏹ Stopping after the current clip…" if stopping else "▶️ Running"
        lines = [f"🎬 <b>Clipper — {head}</b>"]
        cur = status.get("current") or {}
        done, cap = status.get("videos_done", 0), status.get("max_videos") or "?"
        if cur.get("title"):
            lines.append(f"Video {done + 1}/{cap}: {esc(cur['title'])}")
            lines.append(f"Now: {esc(cur.get('stage', ''))}")
        elif cur.get("stage"):
            lines.append(f"Now: {esc(cur['stage'])}")
        elif status.get("state") == "starting":
            lines.append("Starting…")
        lines.append(f"{clips} clip(s) so far · ${cost:.2f} Claude · "
                     f"{_minutes_since(status.get('started_at'))} min")  # fmt: skip
        lines += ["", LID_NOTE]
        return "\n".join(lines)
    lines = ["🎬 <b>Clipper — Idle</b>"]
    if status.get("finished_at"):
        lines.append(f"Last run {_local(status.get('finished_at'))}: "
                     f"{esc(str(status.get('reason') or 'finished'))} · "
                     f"{status.get('videos_done', 0)} video(s), {clips} clip(s), "
                     f"${cost:.2f}")  # fmt: skip
    lines.append(f"▶️ Start clips new videos one by one (up to "
                 f"{cfg.pipeline.max_videos_per_run}) until you stop it.")  # fmt: skip
    return "\n".join(lines)


def panel_buttons(running: bool, stopping: bool) -> dict[str, Any]:
    status = {"text": "📊 Status", "callback_data": "ctl:status"}
    if not running:
        row = [{"text": "▶️ Start", "callback_data": "ctl:start"}, status]
    elif stopping:
        row = [{"text": "⛔ Stop now", "callback_data": "ctl:stopnow"}, status]
    else:
        row = [{"text": "⏹ Stop", "callback_data": "ctl:stop"}, status]
    return {"inline_keyboard": [row]}


def current_panel(ix: Inbox) -> tuple[str, dict[str, Any]]:
    running = run_active(ix.cfg)
    stopping = running and runstate.stop_requested(ix.cfg.data_dir)
    status = runstate.read_status(ix.cfg.data_dir)
    return panel_text(ix.cfg, status, running, stopping), panel_buttons(running, stopping)


def refresh_panel(ix: Inbox, repost: bool = False) -> None:
    """Edit the pinned panel if its text changed; post (and pin) it if there is none or
    `repost` (moves it to the bottom of the chat)."""
    text, markup = current_panel(ix)
    key = text + json.dumps(markup)
    message_id = db.get_setting(ix.conn, PANEL_KEY)
    if message_id and not repost:
        if db.get_setting(ix.conn, PANEL_TEXT_KEY) == key:
            return
        try:
            ix.api.call("editMessageText",
                        {"chat_id": ix.chat_id, "message_id": int(message_id), "text": text,
                         "parse_mode": "HTML", "reply_markup": markup}, retries=1)  # fmt: skip
            db.set_setting(ix.conn, PANEL_TEXT_KEY, key)
            return
        except TelegramError as exc:
            if "not modified" in str(exc):
                db.set_setting(ix.conn, PANEL_TEXT_KEY, key)
                return
            log.info("panel message gone (%s); posting a new one", exc)
    if message_id and repost:
        try:
            ix.api.call("deleteMessage", {"chat_id": ix.chat_id, "message_id": int(message_id)},
                        retries=1)  # fmt: skip
        except TelegramError:
            pass
    sent = ix.say(text, reply_markup=markup)
    new_id = int(sent["message_id"])
    try:
        ix.api.call("pinChatMessage", {"chat_id": ix.chat_id, "message_id": new_id,
                                       "disable_notification": True}, retries=1)  # fmt: skip
    except TelegramError as exc:
        log.warning("could not pin the control panel: %s", exc)
    db.set_setting(ix.conn, PANEL_KEY, str(new_id))
    db.set_setting(ix.conn, PANEL_TEXT_KEY, key)


# --- start / stop -----------------------------------------------------------------------


def start(ix: Inbox, args: list[str] | None = None,
          spawn: Callable[..., subprocess.Popen] = subprocess.Popen) -> str:  # fmt: skip
    """Launch a run (args go to clipper.run, e.g. --url X or --max-videos N)."""
    global _proc
    if run_active(ix.cfg):
        refresh_panel(ix)
        return "A run is already going"
    data = ix.cfg.data_dir
    runstate.clear_stop(data)
    runstate.write_status(data, {"state": "starting", "started_at": db.now(), "videos_done": 0,
                                 "clips": 0, "cost_usd": 0.0, "current": None, "reason": None,
                                 "finished_at": None,
                                 "max_videos": ix.cfg.pipeline.max_videos_per_run})  # fmt: skip
    _proc = spawn(["/bin/bash", str(SCRIPT), *(args or [])], cwd=REPO_ROOT,
                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                  stderr=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
    db.set_setting(ix.conn, PGID_KEY, str(_proc.pid))
    log.info("started a run (pid %d): %s", _proc.pid, " ".join(args or []) or "continuous")
    refresh_panel(ix)
    return "Started"


def _lock_pid(cfg: Config) -> int | None:
    try:
        return int((cfg.data_dir / LOCK_NAME).read_text().strip())
    except (OSError, ValueError):
        return None


def _signal(kill: Callable[[int, int], None], pgid: int, sig: int) -> None:
    try:
        kill(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def stop(ix: Inbox, now: bool = False, kill: Callable[[int, int], None] = os.killpg,
         sleep: Callable[[float], None] = time.sleep) -> str:  # fmt: skip
    if not run_active(ix.cfg):
        runstate.clear_stop(ix.cfg.data_dir)
        refresh_panel(ix)
        return "Nothing is running"
    runstate.request_stop(ix.cfg.data_dir)
    if not now:
        runstate.update_status(ix.cfg.data_dir, state="stopping")
        refresh_panel(ix)
        return "Stopping after the current clip"
    pgid = int(db.get_setting(ix.conn, PGID_KEY) or 0) or _lock_pid(ix.cfg)
    if not pgid:
        return "Cannot find the run's process"
    log.warning("stop now: SIGTERM to process group %d", pgid)
    _signal(kill, pgid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_NOW_GRACE_SEC
    while run_active(ix.cfg) and time.monotonic() < deadline:
        sleep(0.5)
    if run_active(ix.cfg):
        log.warning("stop now: still running, SIGKILL")
        _signal(kill, pgid, signal.SIGKILL)
        child = ix.cfg.data_dir / f"{LOCK_NAME}.child"  # whisper/ffmpeg: own process group
        try:
            _signal(kill, int(child.read_text().strip()), signal.SIGKILL)
        except (OSError, ValueError):
            pass
    reap(ix)
    refresh_panel(ix)
    return "Stopped"


def reap(ix: Inbox) -> int | None:
    """Collect the run we started once it has exited; report it if the run itself could
    not (it never started properly, or was killed). Returns its exit code, if it ended."""
    global _proc
    if _proc is None or _proc.poll() is None:
        return None
    code = _proc.returncode
    _proc = None
    db.set_setting(ix.conn, PGID_KEY, "")
    runstate.clear_stop(ix.cfg.data_dir)
    status = runstate.read_status(ix.cfg.data_dir)
    if status.get("state") != "finished":
        reason = ("another run (e.g. a re-render) holds the lock" if code == 4
                  else f"ended with exit code {code}")  # fmt: skip
        runstate.update_status(ix.cfg.data_dir, state="finished", reason=reason, current=None,
                               finished_at=db.now())  # fmt: skip
        ix.say(f"⚠️ Run {html.escape(reason)}. Log: data/logs/pipeline-*.log")
    return code


# --- Telegram entry points --------------------------------------------------------------


def on_button(ix: Inbox, action: str) -> str:
    if action == "start":
        return start(ix)
    if action == "stop":
        return stop(ix)
    if action == "stopnow":
        return stop(ix, now=True)
    if action == "status":
        refresh_panel(ix)
        return "Running" if run_active(ix.cfg) else "Idle"
    return "Unknown button"


def on_command(ix: Inbox, command: str, args: list[str]) -> bool:
    """/run, /run <url>, /run <n>, /stop, /stop now, /panel. False if not one of them."""
    if command == "/run":
        extra: list[str] = []
        if args and args[0].startswith("http"):
            extra = ["--url", args[0]]
        elif args and args[0].isdigit() and int(args[0]) > 0:
            extra = ["--max-videos", args[0]]
        elif args:
            ix.say("Usage: /run · /run &lt;youtube url&gt; · /run &lt;number of videos&gt;")
            return True
        ix.say(start(ix, extra))
        return True
    if command == "/stop":
        now = bool(args) and args[0].lower() == "now"
        ix.say(stop(ix, now=now))
        return True
    if command == "/panel":
        refresh_panel(ix, repost=True)
        return True
    return False
