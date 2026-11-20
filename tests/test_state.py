"""Status machine, retry, discovery selection and the run lock."""

import os
import subprocess
import sys
from datetime import date

import pytest

from clipper import db, proc
from clipper.config import Campaign, Config, Machine, Pipeline
from clipper.stages import discover as discover_mod
from clipper.stages.discover import Candidate, frontier, too_old
from clipper.stages.fetch import VideoMeta


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path)
    yield c
    c.close()


def add(conn, vid: str, status: str = "new") -> None:
    db.add_video(conn, vid, "tjr", f"https://x/{vid}", f"title {vid}", 600, "20260101", status)


def status(conn, vid: str) -> str:
    return db.get_video(conn, vid)["status"]


def cand(vid: str, minutes: float = 10, date: str = "", campaign: str = "tjr") -> Candidate:
    return Candidate(vid, f"https://x/{vid}", vid, minutes * 60, campaign, date)


# --- status transitions --------------------------------------------------------


def test_video_walks_the_pipeline_in_order(conn):
    add(conn, "v")
    seen = ["new"] + [db.advance_video(conn, "v") for _ in range(5)]
    assert tuple(seen) == db.PIPELINE
    assert seen == ["new", "audio_downloaded", "transcribed", "picked", "rendered", "done"]
    with pytest.raises(ValueError):
        db.advance_video(conn, "v")  # done is final


def test_next_status_rejects_states_outside_the_pipeline():
    for bad in ("done", "failed", "skipped_too_long", "nonsense"):
        with pytest.raises(ValueError):
            db.next_status(bad)


def test_failure_records_error_stage_and_attempts(conn):
    add(conn, "v", "audio_downloaded")
    db.fail_video(conn, "v", "RuntimeError: whisper died")
    row = db.get_video(conn, "v")
    assert (row["status"], row["failed_status"], row["attempts"]) == (
        "failed",
        "audio_downloaded",
        1,
    )
    assert row["error"] == "RuntimeError: whisper died"


def test_requeue_returns_video_to_the_stage_it_failed_in(conn):
    add(conn, "v", "transcribed")
    db.fail_video(conn, "v", "boom")
    assert db.requeue_failed(conn, max_retries=3) == 1
    row = db.get_video(conn, "v")
    assert (row["status"], row["attempts"], row["failed_status"]) == ("transcribed", 1, None)


def test_failed_video_is_retried_three_times_then_stays_failed(conn):
    add(conn, "v")
    retries = 0
    for _ in range(10):  # each loop = one nightly run that fails again
        db.fail_video(conn, "v", "boom")
        if not db.requeue_failed(conn, max_retries=3):
            break
        retries += 1
    row = db.get_video(conn, "v")
    assert retries == 3
    assert (row["status"], row["attempts"], row["failed_status"]) == ("failed", 4, "new")
    assert db.videos_in(conn, "new") == []


def test_success_clears_error_and_attempts(conn):
    add(conn, "v")
    db.fail_video(conn, "v", "boom")
    db.requeue_failed(conn, 3)
    assert db.advance_video(conn, "v") == "audio_downloaded"
    row = db.get_video(conn, "v")
    assert (row["attempts"], row["error"], row["failed_status"]) == (0, None, None)


def test_add_video_never_overwrites_a_known_id(conn):
    assert db.add_video(conn, "v", "tjr", "u", "t", 600) is True
    db.advance_video(conn, "v")
    assert db.add_video(conn, "v", "tjr", "u", "t", 600) is False
    assert status(conn, "v") == "audio_downloaded"


def test_videos_in_returns_oldest_first(conn):
    for vid in ("b", "a", "c"):
        add(conn, vid)
    assert [r["id"] for r in db.videos_in(conn, "new")] == ["b", "a", "c"]


def test_in_progress_counts_unfinished_and_retriable_videos(conn):
    add(conn, "new")
    add(conn, "picked", "picked")
    add(conn, "done", "done")
    add(conn, "long", db.SKIPPED_TOO_LONG)
    add(conn, "retry")
    db.fail_video(conn, "retry", "boom")  # 1 attempt: will be retried
    add(conn, "dead")
    for _ in range(4):
        db.fail_video(conn, "dead", "boom")
        db.requeue_failed(conn, 3)
    assert status(conn, "dead") == "failed"
    assert db.in_progress_count(conn, max_retries=3) == 3  # new, picked, retry


def test_clip_fail_retry_and_render(conn):
    db.add_clip(conn, "v_01", "v", 0, 30, "hook", "title", "reason", 0.9)
    db.fail_clip(conn, "v_01")
    assert db.clip_rows(conn, "v")["v_01"]["status"] == "failed"
    assert db.requeue_failed(conn, 3) == 1
    db.mark_clip_rendered(conn, "v_01", "/clips/v_01.mp4", "fit")
    row = db.clip_rows(conn, "v")["v_01"]
    assert (row["status"], row["layout"], row["attempts"]) == ("rendered", "fit", 1)
    db.add_clip(conn, "v_01", "v", 0, 30, "hook", "title", "reason", 0.9)  # picked again
    assert db.clip_rows(conn, "v")["v_01"]["status"] == "rendered"  # not reset, no duplicate
    assert len(db.clip_rows(conn, "v")) == 1


def test_clip_gives_up_after_retries(conn):
    db.add_clip(conn, "v_01", "v", 0, 30, "h", "t", "r", 0.9)
    for _ in range(4):
        db.fail_clip(conn, "v_01")
        db.requeue_failed(conn, 3)
    assert db.clip_rows(conn, "v")["v_01"]["status"] == "failed"


def test_migration_adds_columns_to_an_m1_database(tmp_path):
    import sqlite3

    old = sqlite3.connect(tmp_path / "clipper.db")
    old.executescript(
        "CREATE TABLE videos(id TEXT PRIMARY KEY, campaign_id TEXT, url TEXT, title TEXT,"
        " duration_sec REAL, status TEXT, error TEXT, created_at TEXT, updated_at TEXT);"
        "INSERT INTO videos(id, status) VALUES ('old', 'done');"
    )
    old.close()
    conn = db.connect(tmp_path)
    row = db.get_video(conn, "old")
    assert (row["status"], row["attempts"], row["failed_status"]) == ("done", 0, None)
    assert "old" in db.known_video_ids(conn)
    conn.close()


# --- discovery: newest unseen first, then the backlog -----------------------------------


def test_frontier_skips_known_and_sets_aside_long_ones():
    entries = [cand("a"), cand("long", 121), cand("edge", 120), cand("later")]
    front, too_long = frontier(entries, known={"a"}, max_minutes=120)
    assert front.id == "edge"  # exactly 120 min is allowed
    assert [c.id for c in too_long] == ["long"]  # never looked past the frontier
    assert frontier([cand("a")], {"a"}, 120) == (None, [])


def test_too_old():
    today = date(2026, 11, 20)
    assert not too_old("20261021", 30, today)  # exactly 30 days
    assert too_old("20261020", 30, today)
    assert not too_old("", 30, today)  # unknown date: allowed


def make_cfg(tmp_path, **pipeline) -> Config:
    return Config(
        machine=Machine(data_dir=tmp_path, min_free_gb=0),
        pipeline=Pipeline(**pipeline),
        campaigns=[
            Campaign(id="tjr", name="TJR", sources=["src-a", "src-b"]),
            Campaign(id="off", name="Off", sources=["src-off"], active=False),
        ],
    )


TODAY = date(2026, 11, 20)
LISTINGS = {
    "src-a": [cand("a1"), cand("a-long", 300), cand("a2"), cand("a3"), cand("a-old")],
    "src-b": [cand("b1"), cand("b2")],
}
DATES = {"a-long": "20261115", "a1": "20261101", "a2": "20261025", "a3": "20261022",
         "a-old": "20260901", "b1": "20261110", "b2": "20261030"}  # fmt: skip


@pytest.fixture
def fake_youtube(monkeypatch):
    probed: list[str] = []

    def fake_probe(url: str, campaign_id: str) -> VideoMeta:
        vid = url.rsplit("/", 1)[-1]
        probed.append(vid)
        return VideoMeta(vid, url, vid, 600, DATES[vid], campaign_id)

    monkeypatch.setattr(discover_mod, "list_source", lambda url, cid: LISTINGS[url])
    monkeypatch.setattr(discover_mod, "probe", fake_probe)
    return probed


def test_discover_next_goes_newest_first_then_into_the_backlog(conn, tmp_path, fake_youtube):
    cfg = make_cfg(tmp_path)
    order = []
    while (vid := discover_mod.discover_next(cfg, conn, TODAY)) is not None:
        order.append(vid)
        db.advance_video(conn, vid)  # pretend it was clipped
    # newest across sources each time; a-old (80 days) is past max_age_days: never taken
    assert order == ["b1", "a1", "b2", "a2", "a3"]
    assert status(conn, "a-long") == db.SKIPPED_TOO_LONG
    assert db.get_video(conn, "a-old") is None
    assert db.get_video(conn, "b1")["upload_date"] == "20261110"
    assert "src-off" not in str(fake_youtube)  # inactive campaign never listed


def test_discover_next_probes_only_each_sources_frontier(conn, tmp_path, fake_youtube):
    assert discover_mod.discover_next(make_cfg(tmp_path), conn, TODAY) == "b1"
    assert sorted(fake_youtube) == ["a1", "b1"]


def test_discover_next_respects_max_source_minutes(conn, tmp_path, fake_youtube):
    cfg = make_cfg(tmp_path, max_source_minutes=400)
    add(conn, "b1", "done")
    add(conn, "b2", "done")
    assert discover_mod.discover_next(cfg, conn, TODAY) == "a1"
    add(conn, "a1", "done")
    assert discover_mod.discover_next(cfg, conn, TODAY) == "a-long"  # short enough now
    assert not db.videos_in(conn, db.SKIPPED_TOO_LONG)


def test_discover_next_none_when_everything_is_too_old(conn, tmp_path, fake_youtube):
    assert discover_mod.discover_next(make_cfg(tmp_path, max_age_days=1), conn, TODAY) is None
    assert db.videos_in(conn, "new") == []


# --- run lock --------------------------------------------------------------------


def test_second_run_is_refused_while_the_first_holds_the_lock(tmp_path):
    with proc.run_lock(tmp_path):
        assert (tmp_path / proc.LOCK_NAME).read_text() == str(os.getpid())
        with pytest.raises(proc.AlreadyRunning), proc.run_lock(tmp_path):
            pass
        assert (tmp_path / proc.LOCK_NAME).exists()  # the refused run must not remove it
    assert not (tmp_path / proc.LOCK_NAME).exists()


def test_lock_is_released_when_the_run_crashes(tmp_path):
    with pytest.raises(KeyboardInterrupt), proc.run_lock(tmp_path):
        raise KeyboardInterrupt
    with proc.run_lock(tmp_path):
        pass


def test_stale_lock_from_a_dead_run_is_taken_over(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (tmp_path / proc.LOCK_NAME).write_text(str(dead.pid))
    with proc.run_lock(tmp_path):
        assert (tmp_path / proc.LOCK_NAME).read_text() == str(os.getpid())


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX only")
def test_stale_lock_takeover_kills_the_orphaned_child(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    orphan = subprocess.Popen(["sleep", "60"], start_new_session=True)
    (tmp_path / proc.LOCK_NAME).write_text(str(dead.pid))
    (tmp_path / (proc.LOCK_NAME + ".child")).write_text(str(orphan.pid))
    with proc.run_lock(tmp_path):
        assert orphan.wait(timeout=5) != 0  # killed


def test_run_kills_its_child_when_interrupted(tmp_path):
    with proc.run_lock(tmp_path), pytest.raises(subprocess.TimeoutExpired):
        proc.run(["sleep", "60"], timeout=0.3)
    assert not (tmp_path / (proc.LOCK_NAME + ".child")).exists()
    assert proc.run([sys.executable, "-c", "print('hi')"]).stdout == "hi\n"
