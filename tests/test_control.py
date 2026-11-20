"""Telegram control panel (PLAN.md §17): start/stop state machine, single run, stop now,
panel text and buttons, commands. Subprocess and Telegram are faked."""

import signal

import pytest

from clipper import db, runstate
from clipper.config import Config, Machine, Pipeline, Telegram
from clipper.telegram import control
from clipper.telegram import inbox as ib

CHAT = 4242


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.next_id = 500

    def call(self, method, params=None, files=None, timeout=60, retries=3):
        self.calls.append((method, dict(params or {})))
        if method == "sendMessage":
            self.next_id += 1
            return {"message_id": self.next_id}
        return True

    def last(self, method):
        return next(p for m, p in reversed(self.calls) if m == method)

    def count(self, method):
        return sum(m == method for m, _ in self.calls)


class FakeProc:
    def __init__(self, pid=777):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode


@pytest.fixture
def ix(tmp_path, monkeypatch):
    cfg = Config(machine=Machine(data_dir=tmp_path, min_free_gb=0),
                 pipeline=Pipeline(max_videos_per_run=6),
                 telegram=Telegram(enabled=True, chat_id=CHAT))  # fmt: skip
    conn = db.connect(tmp_path)
    monkeypatch.setattr(control, "_proc", None)
    monkeypatch.setattr(control, "lock_held", lambda data_dir: False)
    yield ib.Inbox(cfg, conn, FakeAPI(), CHAT)
    conn.close()


def spawner(procs):
    def spawn(cmd, **kw):
        assert kw["start_new_session"] is True  # own process group: Stop now can kill it all
        procs.append(cmd)
        return FakeProc()

    return spawn


def press(ix, data, chat=CHAT):
    ib.handle_update(ix, {"update_id": 1, "callback_query": {
        "id": "cb", "data": data, "message": {"message_id": 1, "chat": {"id": chat}}}})  # fmt: skip


def text_of(ix):
    return control.current_panel(ix)[0]


def buttons_of(ix):
    return [b["callback_data"] for b in control.current_panel(ix)[1]["inline_keyboard"][0]]


def test_idle_panel_offers_start(ix):
    assert text_of(ix).startswith("🎬 <b>Clipper — Idle</b>")
    assert "up to 6" in text_of(ix)
    assert buttons_of(ix) == ["ctl:start", "ctl:status"]


def test_start_launches_one_run_through_the_pipeline_script(ix):
    procs = []
    assert control.start(ix, spawn=spawner(procs)) == "Started"
    assert procs == [["/bin/bash", str(control.SCRIPT)]]
    assert runstate.read_status(ix.cfg.data_dir)["state"] == "starting"
    assert buttons_of(ix) == ["ctl:stop", "ctl:status"]
    assert "Running" in text_of(ix) and "Keep the lid open" in text_of(ix)
    # a second Start while running only shows the status
    assert control.start(ix, spawn=spawner(procs)) == "A run is already going"
    assert len(procs) == 1


def test_start_is_refused_while_another_run_holds_the_lock(ix, monkeypatch):
    monkeypatch.setattr(control, "lock_held", lambda data_dir: True)
    procs = []
    assert control.start(ix, spawn=spawner(procs)) == "A run is already going"
    assert procs == []


def test_running_panel_shows_video_stage_and_counts(ix):
    control.start(ix, spawn=spawner([]))
    runstate.update_status(ix.cfg.data_dir, state="running", videos_done=1, max_videos=6, clips=3,
                           cost_usd=0.18, current={"id": "v", "title": "TJR <live>",
                                                   "stage": "transcribing"})  # fmt: skip
    text = text_of(ix)
    assert "Video 2/6: TJR &lt;live&gt;" in text and "Now: transcribing" in text
    assert "3 clip(s) so far · $0.18 Claude" in text


def test_stop_sets_the_flag_and_offers_stop_now(ix):
    control.start(ix, spawn=spawner([]))
    assert control.stop(ix) == "Stopping after the current clip"
    assert runstate.stop_requested(ix.cfg.data_dir)
    assert "Stopping after the current clip" in text_of(ix)
    assert buttons_of(ix) == ["ctl:stopnow", "ctl:status"]


def test_stop_now_terminates_then_kills_the_process_group(ix, monkeypatch):
    control.start(ix, spawn=spawner([]))
    kills = []
    monkeypatch.setattr(control, "STOP_NOW_GRACE_SEC", 0)
    (ix.cfg.data_dir / "clipper.lock.child").write_text("888")  # whisper: its own group
    result = control.stop(ix, now=True, kill=lambda pid, sig: kills.append((pid, sig)),
                          sleep=lambda s: None)  # fmt: skip
    assert result == "Stopped"
    assert kills == [(777, signal.SIGTERM), (777, signal.SIGKILL), (888, signal.SIGKILL)]


def test_stop_now_stops_at_sigterm_when_the_run_exits(ix):
    control.start(ix, spawn=spawner([]))
    proc = control._proc
    kills = []

    def kill(pid, sig):
        kills.append(sig)
        proc.returncode = -15  # the run exits on SIGTERM

    runstate.update_status(ix.cfg.data_dir, state="finished", reason="stopped now")
    control.stop(ix, now=True, kill=kill, sleep=lambda s: None)
    assert kills == [signal.SIGTERM]
    assert control._proc is None and not runstate.stop_requested(ix.cfg.data_dir)


def test_reap_reports_a_run_that_died_without_finishing(ix):
    control.start(ix, spawn=spawner([]))
    control._proc.returncode = 1
    assert control.reap(ix) == 1
    status = runstate.read_status(ix.cfg.data_dir)
    assert (status["state"], status["reason"]) == ("finished", "ended with exit code 1")
    assert "⚠️ Run ended with exit code 1" in ix.api.last("sendMessage")["text"]
    assert text_of(ix).startswith("🎬 <b>Clipper — Idle</b>")
    assert "Last run" in text_of(ix) and "ended with exit code 1" in text_of(ix)


def test_reap_is_quiet_when_the_run_reported_itself(ix):
    control.start(ix, spawn=spawner([]))
    runstate.update_status(ix.cfg.data_dir, state="finished", reason="no new videos left")
    control._proc.returncode = 0
    sent = ix.api.count("sendMessage")
    control.reap(ix)
    assert ix.api.count("sendMessage") == sent


def test_panel_is_posted_pinned_and_then_only_edited_on_change(ix):
    control.refresh_panel(ix)
    assert ix.api.last("pinChatMessage")["message_id"] == 501
    assert db.get_setting(ix.conn, control.PANEL_KEY) == "501"
    control.refresh_panel(ix)
    assert ix.api.count("editMessageText") == 0  # nothing changed
    control.start(ix, spawn=spawner([]))  # refreshes the panel
    assert ix.api.last("editMessageText")["message_id"] == 501
    assert ix.api.count("sendMessage") == 1


def test_buttons_and_commands(ix, monkeypatch):
    started = []
    monkeypatch.setattr(control, "start", lambda ix, args=None: started.append(args) or "Started")
    press(ix, "ctl:start")
    assert started == [None] and ix.api.last("answerCallbackQuery")["text"] == "Started"
    for text, args in [("/run", []), ("/run 2", ["--max-videos", "2"]),
                       ("/run https://youtu.be/abc", ["--url", "https://youtu.be/abc"])]:  # fmt: skip
        ib.handle_update(ix, {"update_id": 2, "message": {
            "message_id": 9, "chat": {"id": CHAT}, "text": text}})  # fmt: skip
        assert started[-1] == args
    ib.on_command(ix, "/run", ["lots"])
    assert ix.api.last("sendMessage")["text"].startswith("Usage: /run")


def test_stop_command_variants(ix, monkeypatch):
    calls = []
    monkeypatch.setattr(control, "stop", lambda ix, now=False: calls.append(now) or "ok")
    ib.on_command(ix, "/stop", [])
    ib.on_command(ix, "/stop", ["now"])
    assert calls == [False, True]


def test_control_from_another_chat_is_ignored(ix, monkeypatch):
    started = []
    monkeypatch.setattr(control, "start", lambda ix, args=None: started.append(args) or "x")
    press(ix, "ctl:start", chat=666)
    ib.handle_update(ix, {"update_id": 3, "message": {
        "message_id": 9, "chat": {"id": 666}, "text": "/run"}})  # fmt: skip
    assert started == [] and ix.api.calls == []


def test_stop_when_idle(ix):
    assert control.stop(ix) == "Nothing is running"
    assert not runstate.stop_requested(ix.cfg.data_dir)
