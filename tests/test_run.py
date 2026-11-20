"""Orchestrator: resume after a kill, retries, Claude failure. Heavy stages are faked."""

import json
from collections import Counter

import pytest

from clipper import db, run, runstate
from clipper import status as status_mod
from clipper.config import Campaign, Config, Machine
from clipper.llm.claude_cli import ClaudeCLIError, LLMResult
from clipper.stages import pick as pick_mod
from clipper.stages.pick import Clip, save_clips
from clipper.stages.transcribe import whisper_cmd, whisper_prompt


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        machine=Machine(data_dir=tmp_path, min_free_gb=0),
        campaigns=[Campaign(id="tjr", name="TJR")],
    )


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path)
    yield c
    c.close()


class Fakes:
    """Stand-ins for the heavy stages that write the same files as the real ones."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.kill_transcribe_of: set[str] = set()
        self.claude_down = False
        self.audio_broken = False
        self.broken_clips: set[str] = set()

    def fetch_audio(self, cfg, meta):
        self.calls["audio"] += 1
        if self.audio_broken:
            raise RuntimeError("network down")
        wav = cfg.dir("sources") / f"{meta.id}.wav"
        wav.write_text("wav")
        return wav

    def transcribe(self, cfg, video_id, wav, vocabulary=None):
        self.calls["transcribe"] += 1
        self.vocabulary = vocabulary
        assert wav.exists()
        if video_id in self.kill_transcribe_of:
            raise KeyboardInterrupt  # the process is killed mid-transcription
        out = cfg.dir("transcripts")
        (out / f"{video_id}.words.json").write_text("[]")
        (out / f"{video_id}.compact.txt").write_text("[00:00] hi\n")
        return out / f"{video_id}.words.json", out / f"{video_id}.compact.txt"

    def pick(self, cfg, conn, video_id, campaign, words_path, compact_path):
        self.calls["pick"] += 1
        if self.claude_down:
            raise ClaudeCLIError("credit exhausted", LLMResult("", 1, "sonnet"))
        clips = [
            Clip(
                f"{video_id}_{n:02d}", video_id, n * 40.0, n * 40.0 + 30, "hook", "t", "t", "r", 0.9
            )
            for n in (1, 2)
        ]
        save_clips(clips, cfg.dir("transcripts") / f"{video_id}.clips.json")
        for c in clips:
            db.add_clip(conn, c.id, c.video_id, c.start_sec, c.end_sec, c.hook_text, c.title,
                        c.reason, c.score)  # fmt: skip
        return clips

    def fetch_segment(self, cfg, meta, index, start, end):
        self.calls["segment"] += 1
        seg = cfg.dir("sources") / f"{meta.id}_{index:02d}.mp4"
        seg.write_text("segment")
        return seg

    def render_clip(self, cfg, clip, segment, words, campaign):
        self.calls["render"] += 1
        assert campaign.id == "tjr"
        if clip.id in self.broken_clips:
            raise RuntimeError("ffmpeg died")
        out = cfg.dir("clips") / f"{clip.id}.mp4"
        out.write_text("clip")
        return out


@pytest.fixture
def fakes(monkeypatch) -> Fakes:
    f = Fakes()
    for name in ("fetch_audio", "transcribe", "pick", "fetch_segment", "render_clip"):
        monkeypatch.setattr(run, name, getattr(f, name))
    return f


def add_two(conn) -> None:
    for vid in ("v1", "v2"):
        db.add_video(conn, vid, "tjr", f"https://x/{vid}", vid, 600, "20261101")


def statuses(conn) -> dict[str, str]:
    return {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM videos")}


def go(cfg, conn) -> int:
    return run.run_pipeline(cfg, conn)


def test_full_run_takes_videos_to_done_and_deletes_sources(cfg, conn, fakes):
    add_two(conn)
    assert go(cfg, conn) == run.EXIT_OK
    assert statuses(conn) == {"v1": "done", "v2": "done"}
    assert sorted(p.name for p in cfg.dir("clips").glob("*.mp4")) == [
        "v1_01.mp4", "v1_02.mp4", "v2_01.mp4", "v2_02.mp4",
    ]  # fmt: skip
    assert list(cfg.dir("sources").iterdir()) == []  # wav and segments are gone
    assert {r["status"] for r in conn.execute("SELECT status FROM clips")} == {"rendered"}


def test_kill_during_transcription_then_rerun_finishes_without_repeating_work(cfg, conn, fakes):
    add_two(conn)
    fakes.kill_transcribe_of = {"v2"}
    with pytest.raises(KeyboardInterrupt):
        go(cfg, conn)
    assert statuses(conn) == {"v1": "transcribed", "v2": "audio_downloaded"}
    assert db.get_video(conn, "v2")["attempts"] == 0  # a kill is not the video's failure

    fakes.kill_transcribe_of = set()
    assert go(cfg, conn) == run.EXIT_OK
    assert statuses(conn) == {"v1": "done", "v2": "done"}
    # v1's transcript and both downloads were not redone; one Claude call per video
    assert fakes.calls == {"audio": 2, "transcribe": 3, "pick": 2, "segment": 4, "render": 4}
    assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 4

    assert go(cfg, conn) == run.EXIT_OK  # a third run has nothing to do
    assert fakes.calls["render"] == 4 and fakes.calls["pick"] == 2


def test_kill_during_render_does_not_duplicate_clips(cfg, conn, fakes, monkeypatch):
    add_two(conn)
    real_render = fakes.render_clip

    def dying_render(cfg, clip, segment, words, campaign):
        if clip.id == "v1_02":
            raise KeyboardInterrupt
        return real_render(cfg, clip, segment, words, campaign)

    monkeypatch.setattr(run, "render_clip", dying_render)
    with pytest.raises(KeyboardInterrupt):
        go(cfg, conn)
    assert statuses(conn) == {"v1": "picked", "v2": "picked"}

    monkeypatch.setattr(run, "render_clip", real_render)
    assert go(cfg, conn) == run.EXIT_OK
    assert statuses(conn) == {"v1": "done", "v2": "done"}
    assert fakes.calls["render"] == 4  # v1_01 was not rendered twice
    assert fakes.calls["pick"] == 2
    assert conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 4


def test_claude_failure_leaves_video_transcribed_and_still_renders_picked_ones(cfg, conn, fakes):
    add_two(conn)
    assert go(cfg, conn) == run.EXIT_OK  # v1, v2 fully done
    db.add_video(conn, "v3", "tjr", "https://x/v3", "v3", 600)
    db.add_video(conn, "v4", "tjr", "https://x/v4", "v4", 600)
    # v3 was picked on an earlier night but not rendered yet
    for _ in range(2):
        db.advance_video(conn, "v3")
    fakes.transcribe(cfg, "v3", fakes.fetch_audio(cfg, run._meta(db.get_video(conn, "v3"))))
    fakes.pick(cfg, conn, "v3", None, None, None)
    db.advance_video(conn, "v3")
    assert statuses(conn)["v3"] == "picked"

    fakes.claude_down = True
    assert go(cfg, conn) == run.EXIT_CLAUDE
    assert statuses(conn)["v4"] == "transcribed"
    row = db.get_video(conn, "v4")
    assert (row["attempts"], row["error"]) == (0, None)  # not counted as a failure
    assert statuses(conn)["v3"] == "done"  # rendering went ahead

    fakes.claude_down = False
    assert go(cfg, conn) == run.EXIT_OK
    assert statuses(conn)["v4"] == "done"


def test_transcribe_gets_the_campaign_vocabulary(cfg, conn, fakes):
    cfg.campaigns[0].vocabulary = ["TJR", "PnL"]
    add_two(conn)
    go(cfg, conn)
    assert fakes.vocabulary == ["TJR", "PnL"]
    assert whisper_prompt(["TJR", " PnL ", "", "prop firm"]) == "TJR, PnL, prop firm."
    assert whisper_prompt([]) == ""


def test_whisper_prompt_is_always_carried_into_every_window(tmp_path):
    plain = whisper_cmd("whisper-cli", tmp_path / "m.bin", tmp_path / "a.wav", tmp_path / "o")
    assert "--prompt" not in plain and "--carry-initial-prompt" not in plain
    primed = whisper_cmd("whisper-cli", tmp_path / "m.bin", tmp_path / "a.wav", tmp_path / "o",
                         "TJR, PnL.")  # fmt: skip
    assert primed[-3:] == ["--prompt", "TJR, PnL.", "--carry-initial-prompt"]
    assert primed[:-3] == plain


def test_failed_stage_is_retried_on_three_later_runs_then_given_up(cfg, conn, fakes):
    db.add_video(conn, "v1", "tjr", "https://x/v1", "v1", 600)
    fakes.audio_broken = True
    for expected_attempts in (1, 2, 3, 4):
        assert go(cfg, conn) == run.EXIT_FAILURES
        row = db.get_video(conn, "v1")
        assert (row["status"], row["attempts"]) == ("failed", expected_attempts)
        assert row["error"] == "RuntimeError: network down"
    assert go(cfg, conn) == run.EXIT_OK  # given up: nothing left to try
    assert fakes.calls["audio"] == 4  # first try + 3 retries


def test_failed_video_recovers_on_retry(cfg, conn, fakes):
    db.add_video(conn, "v1", "tjr", "https://x/v1", "v1", 600)
    fakes.audio_broken = True
    go(cfg, conn)
    fakes.audio_broken = False
    assert go(cfg, conn) == run.EXIT_OK
    row = db.get_video(conn, "v1")
    assert (row["status"], row["attempts"], row["error"]) == ("done", 0, None)


def test_one_bad_video_does_not_block_the_other(cfg, conn, fakes, monkeypatch):
    add_two(conn)
    real = fakes.transcribe

    def flaky(cfg, video_id, wav, vocabulary=None):
        if video_id == "v1":
            raise RuntimeError("no speech found")
        return real(cfg, video_id, wav, vocabulary)

    monkeypatch.setattr(run, "transcribe", flaky)
    assert go(cfg, conn) == run.EXIT_FAILURES
    assert statuses(conn) == {"v1": "failed", "v2": "done"}
    assert db.get_video(conn, "v1")["failed_status"] == "audio_downloaded"


def test_failed_clip_is_retried_and_video_finishes_with_the_rest(cfg, conn, fakes):
    db.add_video(conn, "v1", "tjr", "https://x/v1", "v1", 600)
    fakes.broken_clips = {"v1_02"}
    assert go(cfg, conn) == run.EXIT_FAILURES
    assert statuses(conn) == {"v1": "picked"}  # waits for the clip's retries
    for _ in range(3):
        go(cfg, conn)
    assert statuses(conn) == {"v1": "done"}  # gave up on v1_02, kept v1_01
    clips = db.clip_rows(conn, "v1")
    assert (clips["v1_01"]["status"], clips["v1_02"]["status"]) == ("rendered", "failed")
    assert clips["v1_02"]["attempts"] == 4
    assert fakes.calls["pick"] == 1


def test_real_pick_calls_claude_once_and_adds_no_duplicate_clips(cfg, conn, monkeypatch):
    out = cfg.dir("transcripts")
    words = [{"w": f"w{i}.", "s": i * 5.0, "e": i * 5.0 + 4} for i in range(20)]
    (out / "v.words.json").write_text(json.dumps(words))
    (out / "v.compact.txt").write_text("[00:00] w0.\n")
    reply = json.dumps(
        {
            "clips": [
                {"start": "00:00", "end": "00:30", "hook_text": "h", "title": "t", "score": 0.9}
            ]
        }
    )
    asked: list[str] = []

    def fake_claude(cfg, prompt, cwd):
        asked.append(prompt)
        return LLMResult(reply, 0, "sonnet", 100, 20, 0.01)

    monkeypatch.setattr(pick_mod, "run_claude", fake_claude)
    args = (cfg, conn, "v", cfg.campaign("tjr"), out / "v.words.json", out / "v.compact.txt")
    first = pick_mod.pick(*args)
    conn.execute("DELETE FROM clips")  # as if the run died before the rows were written
    second = pick_mod.pick(*args)
    assert len(asked) == 1 and first == second
    assert list(db.clip_rows(conn, "v")) == ["v_01"]
    assert conn.execute("SELECT COUNT(*), SUM(cost_usd) FROM llm_calls").fetchone()[:] == (1, 0.01)


def test_status_report_shows_counts_and_month_cost(cfg, conn, fakes):
    add_two(conn)
    go(cfg, conn)
    db.add_video(conn, "v3", "tjr", "https://x/v3", "A third video", 5400)
    db.fail_video(conn, "v3", "RuntimeError: boom")
    db.add_video(conn, "long", "tjr", "u", "too long", 9000, status=db.SKIPPED_TOO_LONG)
    db.log_llm_call(conn, "v1", "sonnet", 8000, 2000, 0.05, 0)
    db.log_llm_call(conn, "v2", "sonnet", 9000, 1000, 0.07, 0)
    conn.execute(
        "INSERT INTO llm_calls(video_id, model, cost_usd, exit_code, created_at)"
        " VALUES ('old', 'sonnet', 9.99, 0, '2020-01-05T00:00:00+00:00')"
    )
    text = status_mod.report(conn)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    assert "done 2" in lines and "failed 1" in lines and "skipped_too_long 1" in lines
    assert "rendered 4" in lines
    assert any(line.startswith("v3 tjr failed (at new) 1 0/0 90m A third video") for line in lines)
    assert "v3: RuntimeError: boom" in lines
    assert not any(line.startswith("long ") for line in lines)
    assert lines[-1].endswith("2 calls (0 failed), 17,000 in / 3,000 out tokens, $0.12")


# --- continuous on-demand runs (PLAN.md §17) ------------------------------------------


class FakeInbox:
    def __init__(self, conn) -> None:
        self.conn = conn
        self.said: list[str] = []
        self.sent: list[str] = []

    def say(self, text, **extra):
        self.said.append(text)
        return {"message_id": 1}


@pytest.fixture
def inbox(monkeypatch, conn):
    box = FakeInbox(conn)
    monkeypatch.setattr(run, "open_inbox", lambda cfg, conn: box)
    monkeypatch.setattr(run, "lock_held", lambda *a: True)  # the bot is polling, not the run

    def send(ix, row):
        box.sent.append(row["id"])
        db.update_clip(conn, row["id"], status="delivered")

    monkeypatch.setattr(run, "send_clip", send)
    monkeypatch.setattr(run, "deliver", lambda ix, limit: (0, 0))
    return box


class Source:
    """discover_next stand-in: hands out videos one by one."""

    def __init__(self, conn, ids) -> None:
        self.conn, self.ids, self.calls = conn, list(ids), 0

    def __call__(self, cfg, conn):
        self.calls += 1
        if not self.ids:
            return None
        vid = self.ids.pop(0)
        db.add_video(conn, vid, "tjr", f"https://x/{vid}", vid, 600, "20261101")
        return vid


def test_continuous_run_goes_video_by_video_and_delivers_each_clip_at_once(
    cfg, conn, fakes, inbox, monkeypatch
):
    order = []
    real_send = run.send_clip

    def send(ix, row):
        order.append(("send", row["id"], fakes.calls["render"]))
        real_send(ix, row)

    monkeypatch.setattr(run, "send_clip", send)
    source = Source(conn, ["v1", "v2"])
    assert run.run_continuous(cfg, conn, next_video=source) == run.EXIT_OK
    # each clip is sent right after it is rendered, before the next one renders
    assert order == [("send", "v1_01", 1), ("send", "v1_02", 2), ("send", "v2_01", 3),
                     ("send", "v2_02", 4)]  # fmt: skip
    assert statuses(conn) == {"v1": "done", "v2": "done"}
    status = runstate.read_status(cfg.data_dir)
    assert (status["state"], status["reason"], status["videos_done"], status["clips"]) == (
        "finished", "no new videos left", 2, 4)  # fmt: skip
    assert "✅ 2 clip(s) from <b>v1</b>" in inbox.said
    assert inbox.said[-1].startswith("⏹ Run finished: no new videos left · 2 video(s), 4 clip(s)")


def test_continuous_run_stops_at_the_video_cap(cfg, conn, fakes, inbox):
    source = Source(conn, ["v1", "v2", "v3"])
    run.run_continuous(cfg, conn, max_videos=2, next_video=source)
    assert statuses(conn) == {"v1": "done", "v2": "done"}
    assert runstate.read_status(cfg.data_dir)["reason"] == "reached the limit of 2 video(s)"


def test_graceful_stop_finishes_the_clip_delivers_it_and_exits(cfg, conn, fakes, inbox,
                                                               monkeypatch):  # fmt: skip
    real_render = fakes.render_clip

    def render_then_stop(cfg_, clip, *a):
        out = real_render(cfg_, clip, *a)
        if clip.id == "v1_01" and fakes.calls["render"] == 1:
            runstate.request_stop(cfg.data_dir)  # Stop pressed while the first clip renders
        return out

    monkeypatch.setattr(run, "render_clip", render_then_stop)
    source = Source(conn, ["v1", "v2"])
    run.run_continuous(cfg, conn, next_video=source)
    assert inbox.sent == ["v1_01"]  # the clip being rendered is still delivered
    assert db.get_clip(conn, "v1_02")["status"] == "picked"  # the rest waits
    assert statuses(conn) == {"v1": "picked"} and source.calls == 1  # no next video
    assert runstate.read_status(cfg.data_dir)["reason"] == "stopped by you"
    assert not runstate.stop_requested(cfg.data_dir)  # cleared for the next Start
    # the next Start resumes the same video before looking for new ones
    run.run_continuous(cfg, conn, next_video=source)
    assert inbox.sent == ["v1_01", "v1_02", "v2_01", "v2_02"]


def test_stop_between_stages(cfg, conn, fakes, inbox, monkeypatch):
    real = fakes.transcribe

    def transcribe_then_stop(*a, **k):
        out = real(*a, **k)
        runstate.request_stop(cfg.data_dir)
        return out

    monkeypatch.setattr(run, "transcribe", transcribe_then_stop)
    run.run_continuous(cfg, conn, next_video=Source(conn, ["v1"]))
    assert statuses(conn) == {"v1": "transcribed"} and fakes.calls["pick"] == 0


def test_claude_failure_ends_the_run_with_exit_2(cfg, conn, fakes, inbox):
    fakes.claude_down = True
    code = run.run_continuous(cfg, conn, next_video=Source(conn, ["v1", "v2"]))
    assert code == run.EXIT_CLAUDE and statuses(conn) == {"v1": "transcribed"}
    assert "claude -p failed" in runstate.read_status(cfg.data_dir)["reason"]


def test_a_failed_video_does_not_stop_the_run(cfg, conn, fakes, inbox, monkeypatch):
    real = fakes.fetch_audio

    def flaky(cfg_, meta):
        if meta.id == "v1":
            raise RuntimeError("HTTP Error 403")
        return real(cfg_, meta)

    monkeypatch.setattr(run, "fetch_audio", flaky)
    run.run_continuous(cfg, conn, next_video=Source(conn, ["v1", "v2"]))
    assert statuses(conn) == {"v1": "failed", "v2": "done"}
    assert any(s.startswith("⚠️ No clips from <b>v1</b>: RuntimeError: HTTP Error 403")
               for s in inbox.said)  # fmt: skip


def test_url_run_does_only_that_video(cfg, conn, fakes, inbox):
    add_two(conn)
    source = Source(conn, ["v9"])
    run.run_continuous(cfg, conn, only_video="v2", next_video=source)
    assert statuses(conn) == {"v1": "new", "v2": "done"} and source.calls == 0
    assert runstate.read_status(cfg.data_dir)["reason"] == "that video is done"
