"""Orchestrator: runs every stage for pending work, one heavy step at a time.

    python -m clipper.run                         new videos from all active campaigns
    python -m clipper.run --url <video>           add one video, then run the pipeline
    python -m clipper.run --rerender <clip_id> [--layout crop|fit]

Each stage takes the videos waiting in one status and moves them to the next:
    new -> audio_downloaded -> transcribed -> picked -> rendered -> done
All state is in SQLite, so a run can be killed at any point and simply started again.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path

from clipper import db
from clipper.config import Config, load_config
from clipper.disk import InsufficientDiskSpace, ensure_free_space
from clipper.llm.claude_cli import ClaudeCLIError
from clipper.proc import AlreadyRunning, run_lock
from clipper.stages.discover import discover
from clipper.stages.fetch import VideoMeta, fetch_audio, fetch_segment, probe
from clipper.stages.pick import Clip, load_clips, pick, save_clips
from clipper.stages.render import render_clip
from clipper.stages.transcribe import transcribe
from clipper.telegram.api import TelegramError
from clipper.telegram.inbox import Inbox, deliver, open_inbox, poll
from clipper.timing import Word, load_words

log = logging.getLogger("clipper.run")

EXIT_OK, EXIT_FAILURES, EXIT_CLAUDE, EXIT_DISK, EXIT_LOCKED = 0, 1, 2, 3, 4


def _meta(row: sqlite3.Row) -> VideoMeta:
    return VideoMeta(
        id=row["id"],
        url=row["url"],
        title=row["title"] or "",
        duration_sec=row["duration_sec"] or 0.0,
        upload_date=row["upload_date"] or "",
        campaign_id=row["campaign_id"],
    )


def _step(conn: sqlite3.Connection, stage: str, video_id: str, work: Callable[[], object]) -> bool:
    """Run one stage for one video. Success advances its status; an error parks it in
    `failed` and the run carries on with the next item. Disk-full, Claude failures and
    interrupts are not the video's fault: they propagate and leave its status alone."""
    try:
        with db.stage_run(conn, stage, video_id):
            work()
    except (InsufficientDiskSpace, ClaudeCLIError):
        raise
    except Exception as exc:
        log.exception("%s failed for %s", stage, video_id)
        db.fail_video(conn, video_id, f"{type(exc).__name__}: {exc}")
        return False
    db.advance_video(conn, video_id)
    return True


# --- stages: each returns the number of failures ------------------------------


def stage_audio(cfg: Config, conn: sqlite3.Connection) -> int:
    failures = 0
    for row in db.videos_in(conn, "new"):
        meta = _meta(row)
        failures += not _step(conn, "fetch", meta.id, lambda m=meta: fetch_audio(cfg, m))
    return failures


def stage_transcribe(cfg: Config, conn: sqlite3.Connection) -> int:
    failures = 0
    for row in db.videos_in(conn, "audio_downloaded"):
        vid = row["id"]

        def work(v: str = vid, campaign_id: str = row["campaign_id"]) -> None:
            vocabulary = cfg.campaign(campaign_id).vocabulary
            transcribe(cfg, v, cfg.dir("sources") / f"{v}.wav", vocabulary)

        failures += not _step(conn, "transcribe", vid, work)
    return failures


def stage_pick(cfg: Config, conn: sqlite3.Connection) -> tuple[int, bool]:
    """Returns (failures, claude_ok). If `claude -p` fails (e.g. credit exhausted) the video
    stays at `transcribed` and picking stops for this run; later stages still run."""
    failures = 0
    transcripts = cfg.dir("transcripts")
    for row in db.videos_in(conn, "transcribed"):
        vid = row["id"]

        def work(v: str = vid, campaign_id: str = row["campaign_id"]) -> None:
            pick(cfg, conn, v, cfg.campaign(campaign_id), transcripts / f"{v}.words.json",
                 transcripts / f"{v}.compact.txt")  # fmt: skip

        try:
            failures += not _step(conn, "pick", vid, work)
        except ClaudeCLIError as exc:
            log.error("claude -p failed; %s stays at 'transcribed', no more picking this run: %s",
                      vid, exc)  # fmt: skip
            return failures, False
    return failures, True


def _render_one(
    cfg: Config,
    conn: sqlite3.Connection,
    meta: VideoMeta,
    index: int,
    clip: Clip,
    words: list[Word],
) -> bool:
    """Download one clip's segment, render it, delete the segment. Returns success."""
    out = cfg.dir("clips") / f"{clip.id}.mp4"
    try:
        with db.stage_run(conn, "render", clip.id):
            campaign = cfg.campaign(meta.campaign_id)
            segment = fetch_segment(cfg, meta, index, clip.start_sec, clip.end_sec)
            render_clip(cfg, clip, segment, words, campaign)
            segment.unlink()  # source video is deleted as soon as its clip exists
    except InsufficientDiskSpace:
        raise
    except Exception:
        log.exception("clip %s failed", clip.id)
        db.fail_clip(conn, clip.id)
        return False
    db.mark_clip_rendered(conn, clip.id, str(out), clip.layout)
    return True


def _render_video(cfg: Config, conn: sqlite3.Connection, row: sqlite3.Row) -> int:
    """Render the clips of one picked video that are still waiting. The video moves on
    once no clip is waiting or due a retry. Returns the number of clip failures."""
    vid = row["id"]
    transcripts = cfg.dir("transcripts")
    clips = load_clips(transcripts / f"{vid}.clips.json")
    words = load_words(transcripts / f"{vid}.words.json")
    meta = _meta(row)
    failures = 0
    for index, clip in enumerate(clips, start=1):
        state = db.clip_rows(conn, vid).get(clip.id)
        if state is None or state["status"] != "picked":
            continue  # rendered already, or failed and waiting for a later run
        failures += not _render_one(cfg, conn, meta, index, clip, words)

    states = db.clip_rows(conn, vid).values()
    retries = cfg.pipeline.max_retries
    if any(s["status"] == db.FAILED and s["attempts"] <= retries for s in states):
        return failures  # stays `picked`; the failed clips are retried on the next run
    rendered = sum(s["status"] == "rendered" for s in states)
    if rendered == 0:
        db.fail_video(conn, vid, "no clip could be rendered")
    else:
        if rendered < len(states):
            log.warning("%s: %d of %d clips rendered, the rest gave up", vid, rendered, len(states))
        db.advance_video(conn, vid)
    return failures


def stage_render(cfg: Config, conn: sqlite3.Connection) -> int:
    failures = 0
    for row in db.videos_in(conn, "picked"):
        try:
            failures += _render_video(cfg, conn, row)
        except InsufficientDiskSpace:
            raise
        except Exception as exc:
            log.exception("render failed for %s", row["id"])
            db.fail_video(conn, row["id"], f"{type(exc).__name__}: {exc}")
            failures += 1
    return failures


def stage_cleanup(cfg: Config, conn: sqlite3.Connection) -> int:
    """rendered -> done: delete the source audio and any leftover segments."""
    sources = cfg.dir("sources")
    for row in db.videos_in(conn, "rendered"):
        vid = row["id"]
        leftovers = [sources / f"{vid}.wav", *sources.glob(f"{vid}_[0-9][0-9].mp4"),
                     *sources.glob(f"{vid}*.tmp.*"), *sources.glob(f"{vid}*.part")]  # fmt: skip
        for path in leftovers:
            path.unlink(missing_ok=True)
        db.advance_video(conn, vid)
        log.info("%s done", vid)
    return 0


# --- entry points -------------------------------------------------------------


def run_pipeline(cfg: Config, conn: sqlite3.Connection, discover_new: bool = True) -> int:
    """One full pass over all pending work. Returns a process exit code."""
    ensure_free_space(cfg.data_dir, cfg.machine.min_free_gb)
    db.sync_campaigns(conn, cfg.campaigns)
    requeued = db.requeue_failed(conn, cfg.pipeline.max_retries)
    if requeued:
        log.info("retrying %d failed item(s)", requeued)
    if discover_new:
        with db.stage_run(conn, "discover", "-"):
            discover(cfg, conn)

    inbox = open_inbox(cfg, conn)
    if inbox:
        _poll_once(inbox)  # apply buttons pressed while the bot was not running

    failures = stage_audio(cfg, conn)
    failures += stage_transcribe(cfg, conn)
    pick_failures, claude_ok = stage_pick(cfg, conn)
    failures += pick_failures + stage_render(cfg, conn)
    failures += stage_rerender(cfg, conn)
    stage_cleanup(cfg, conn)
    if inbox:
        with db.stage_run(conn, "deliver", "-"):
            failures += deliver(inbox, cfg.telegram.max_per_run)[1]

    if not claude_ok:
        return EXIT_CLAUDE
    return EXIT_FAILURES if failures else EXIT_OK


def add_url(cfg: Config, conn: sqlite3.Connection, campaign_id: str | None, url: str) -> None:
    campaign = cfg.campaign(campaign_id)
    meta = probe(url, campaign.id)
    added = db.add_video(conn, meta.id, campaign.id, meta.url, meta.title, meta.duration_sec,
                         meta.upload_date)  # fmt: skip
    if not added:
        row = db.get_video(conn, meta.id)
        log.info("%s is already known (status %s)", meta.id, row["status"] if row else "?")


def _poll_once(inbox: Inbox) -> None:
    try:
        n = poll(inbox, 0)
        if n:
            log.info("telegram: handled %d update(s)", n)
    except TelegramError as exc:
        if exc.code == 409:
            log.info("telegram: the bot process is polling; skipped")
        else:
            log.warning("telegram poll failed: %s", exc)


def stage_rerender(cfg: Config, conn: sqlite3.Connection) -> int:
    """Clips asked for again (Telegram layout button), and retries of such clips whose
    video is already done. They go back to `rendered`, so deliver sends them again."""
    rows = conn.execute(
        "SELECT c.id FROM clips c JOIN videos v ON v.id = c.video_id WHERE c.status = 'rerender'"
        " OR (c.status = 'picked' AND v.status = 'done') ORDER BY c.created_at, c.id"
    ).fetchall()
    failures = 0
    for row in rows:
        failures += rerender(cfg, conn, row["id"], None) != EXIT_OK
    return failures


def rerender_all(cfg: Config, conn: sqlite3.Connection) -> int:
    """Render every rendered or delivered clip again with the current styling."""
    ids = [r["id"] for status in ("rendered", "delivered") for r in db.clips_in(conn, status)]
    log.info("re-rendering %d clip(s)", len(ids))
    failed = [cid for cid in ids if rerender(cfg, conn, cid, None) != EXIT_OK]
    if failed:
        log.error("failed: %s", ", ".join(failed))
    return EXIT_FAILURES if failed else EXIT_OK


def rerender(cfg: Config, conn: sqlite3.Connection, clip_id: str, layout: str | None) -> int:
    """Render one already-picked clip again, optionally switching its layout. The segment
    is downloaded again (sources are deleted after the first render)."""
    ensure_free_space(cfg.data_dir, cfg.machine.min_free_gb)
    video_id = clip_id.rsplit("_", 1)[0]
    clips_path = cfg.dir("transcripts") / f"{video_id}.clips.json"
    info_path = cfg.dir("sources") / f"{video_id}.info.json"
    if not clips_path.exists() or not info_path.exists():
        log.error("unknown clip %s: no picks or video info for %s", clip_id, video_id)
        return EXIT_FAILURES
    clips = load_clips(clips_path)
    index = next((i for i, c in enumerate(clips, start=1) if c.id == clip_id), None)
    if index is None:
        log.error("unknown clip %s (have: %s)", clip_id, ", ".join(c.id for c in clips))
        return EXIT_FAILURES
    clip = clips[index - 1]
    if layout and layout != clip.layout:
        log.info("%s: layout %s -> %s", clip.id, clip.layout, layout)
        clip.layout = layout
        save_clips(clips, clips_path)

    words = load_words(cfg.dir("transcripts") / f"{video_id}.words.json")
    ok = _render_one(cfg, conn, VideoMeta.load(info_path), index, clip, words)
    return EXIT_OK if ok else EXIT_FAILURES


def _on_sigterm(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)  # unwind normally: children are killed, the lock released


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clipper.run", description=__doc__.splitlines()[0])
    what = parser.add_mutually_exclusive_group()
    what.add_argument("--url", help="add this video (from a joined campaign), then run")
    what.add_argument("--rerender", metavar="CLIP_ID", help="render one picked clip again")
    what.add_argument(
        "--rerender-all", action="store_true", help="render all rendered/delivered clips again"
    )
    parser.add_argument("--layout", choices=["crop", "fit"], help="with --rerender: new layout")
    parser.add_argument("--campaign", help="with --url: campaign id (default: first active)")
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args(argv)
    if args.layout and not args.rerender:
        parser.error("--layout needs --rerender (new clips use render.default_layout)")

    cfg = load_config(args.config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(cfg.dir("logs") / "run.log")],
    )
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        with run_lock(cfg.data_dir):
            conn = db.connect(cfg.data_dir)
            try:
                if args.rerender:
                    return rerender(cfg, conn, args.rerender, args.layout)
                if args.rerender_all:
                    return rerender_all(cfg, conn)
                if args.url:
                    add_url(cfg, conn, args.campaign, args.url)
                return run_pipeline(cfg, conn, discover_new=not args.url)
            finally:
                conn.close()
    except AlreadyRunning as exc:
        log.error("not starting: %s", exc)
        return EXIT_LOCKED
    except InsufficientDiskSpace as exc:
        log.error("stopped: %s", exc)
        return EXIT_DISK
    except Exception:
        log.exception("run failed")
        return EXIT_FAILURES


if __name__ == "__main__":
    sys.exit(main())
