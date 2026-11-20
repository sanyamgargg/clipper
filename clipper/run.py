"""Orchestrator: one heavy step at a time, video after video.

    python -m clipper.run [--max-videos N]        continuous: video after video until stopped
    python -m clipper.run --url <video>           just this video (added if new)
    python -m clipper.run --rerender <clip_id> [--layout framed|crop|fit] [--new-hook]

Normally started from Telegram (▶️ Start / /run, PLAN.md §17) through
scripts/run_pipeline.sh; /stop (or data/run.stop) ends it after the current clip.
Each video goes new -> audio_downloaded -> transcribed -> picked -> rendered -> done.
All state is in SQLite, so a run can be killed at any point and simply started again.
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import signal
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path

from clipper import db, runstate
from clipper.config import Config, load_config
from clipper.disk import InsufficientDiskSpace, ensure_free_space
from clipper.llm.claude_cli import ClaudeCLIError
from clipper.proc import AlreadyRunning, lock_held, run_lock
from clipper.stages.discover import discover_next
from clipper.stages.fetch import VideoMeta, fetch_audio, fetch_segment, probe
from clipper.stages.pick import Clip, load_clips, pick, regenerate_headline, save_clips
from clipper.stages.render import render_clip
from clipper.stages.transcribe import transcribe
from clipper.telegram.api import TelegramError
from clipper.telegram.inbox import BOT_LOCK, Inbox, deliver, open_inbox, poll, send_clip
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


def _render_video(
    cfg: Config,
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    on_rendered: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """Render the clips of one picked video that are still waiting. The video moves on
    once no clip is waiting or due a retry. Returns the number of clip failures.
    `on_rendered(clip_id)` runs after each clip (continuous runs deliver it right away);
    `should_stop()` is checked before each clip: the video then stays `picked`."""
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
        if should_stop and should_stop():
            return failures  # graceful stop: the rest render on the next run
        if _render_one(cfg, conn, meta, index, clip, words):
            if on_rendered:
                on_rendered(clip.id)
        else:
            failures += 1

    states = db.clip_rows(conn, vid).values()
    retries = cfg.pipeline.max_retries
    if any(s["status"] == db.FAILED and s["attempts"] <= retries for s in states):
        return failures  # stays `picked`; the failed clips are retried on the next run
    # delivered (or already acted on in Telegram) clips were rendered too
    rendered = sum(s["status"] not in ("picked", db.FAILED) for s in states)
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


def _cleanup_one(cfg: Config, conn: sqlite3.Connection, vid: str) -> None:
    """rendered -> done: delete the source audio and any leftover segments."""
    sources = cfg.dir("sources")
    leftovers = [sources / f"{vid}.wav", *sources.glob(f"{vid}_[0-9][0-9].mp4"),
                 *sources.glob(f"{vid}*.tmp.*"), *sources.glob(f"{vid}*.part")]  # fmt: skip
    for path in leftovers:
        path.unlink(missing_ok=True)
    db.advance_video(conn, vid)
    log.info("%s done", vid)


def stage_cleanup(cfg: Config, conn: sqlite3.Connection) -> int:
    for row in db.videos_in(conn, "rendered"):
        _cleanup_one(cfg, conn, row["id"])
    return 0


# --- entry points -------------------------------------------------------------


def run_pipeline(cfg: Config, conn: sqlite3.Connection) -> int:
    """One pass over all pending work (no discovery), stage by stage. Returns an exit code.
    On-demand runs use run_continuous instead."""
    ensure_free_space(cfg.data_dir, cfg.machine.min_free_gb)
    db.sync_campaigns(conn, cfg.campaigns)
    requeued = db.requeue_failed(conn, cfg.pipeline.max_retries)
    if requeued:
        log.info("retrying %d failed item(s)", requeued)

    inbox = open_inbox(cfg, conn)
    if inbox and not lock_held(cfg.data_dir, BOT_LOCK):
        _poll_once(inbox)  # the bot is not running: apply buttons pressed meanwhile

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


# --- continuous on-demand runs (PLAN.md §17) ------------------------------------------

STAGE_LABEL = {
    "new": "downloading audio",
    "audio_downloaded": "transcribing",
    "transcribed": "picking clips (Claude)",
    "picked": "rendering",
    "rendered": "cleaning up",
}


class Progress:
    """The run's side of data/run.status.json, plus short Telegram notes."""

    def __init__(self, cfg: Config, conn: sqlite3.Connection, inbox: Inbox | None,
                 max_videos: int, only_video: str | None) -> None:  # fmt: skip
        self.cfg, self.conn, self.inbox = cfg, conn, inbox
        self.started_at = db.now()
        self.videos = 0
        self.clips = 0
        runstate.write_status(cfg.data_dir, {
            "state": "running", "pid": os.getpid(), "started_at": self.started_at,
            "max_videos": 1 if only_video else max_videos, "only_video": only_video,
            "videos_done": 0, "clips": 0, "cost_usd": 0.0, "current": None, "reason": None,
            "finished_at": None,
        })  # fmt: skip

    def cost(self) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE created_at >= ?",
            (self.started_at,),
        ).fetchone()
        return round(float(row[0]), 4)

    def set(self, **fields: object) -> None:
        state = "stopping" if runstate.stop_requested(self.cfg.data_dir) else "running"
        runstate.update_status(self.cfg.data_dir, state=state, videos_done=self.videos,
                               clips=self.clips, cost_usd=self.cost(), **fields)  # fmt: skip

    def finish(self, reason: str) -> None:
        runstate.update_status(self.cfg.data_dir, state="finished", reason=reason, current=None,
                               videos_done=self.videos, clips=self.clips, cost_usd=self.cost(),
                               finished_at=db.now())  # fmt: skip

    def say(self, text: str) -> None:
        if self.inbox is None:
            return
        try:
            self.inbox.say(text)
        except TelegramError as exc:
            log.warning("telegram: %s", exc)


def _next_in_progress(conn: sqlite3.Connection, skip: set[str]) -> sqlite3.Row | None:
    """The oldest video still on its way to done (resumed first), not handled this run."""
    marks = ",".join("?" * len(db.PIPELINE[:-1]))
    for row in conn.execute(
        f"SELECT * FROM videos WHERE status IN ({marks}) ORDER BY created_at, rowid",
        db.PIPELINE[:-1],
    ):
        if row["id"] not in skip:
            return row
    return None


def process_video(
    cfg: Config,
    conn: sqlite3.Connection,
    vid: str,
    progress: Progress,
    should_stop: Callable[[], bool],
) -> str:
    """Take one video as far as it goes: "ok" (done, failed or waiting for clip retries),
    "stopped" (stop flag seen between stages or clips) or "claude" (claude -p failed)."""
    transcripts = cfg.dir("transcripts")
    title = (db.get_video(conn, vid)["title"] or vid)[:80]
    delivered = 0

    def deliver_now(clip_id: str) -> None:
        nonlocal delivered
        progress.clips += 1
        delivered += 1
        if progress.inbox is not None:
            try:
                send_clip(progress.inbox, db.get_clip(conn, clip_id))
            except Exception as exc:  # noqa: BLE001 - stays `rendered`; delivered later
                log.warning("could not deliver %s yet: %s", clip_id, exc)
        progress.set()

    while True:
        row = db.get_video(conn, vid)
        status = row["status"]
        if status not in db.PIPELINE[:-1]:
            break  # done, failed or skipped
        if should_stop():
            return "stopped"
        progress.set(current={"id": vid, "title": title, "stage": STAGE_LABEL[status]})
        meta = _meta(row)
        campaign = cfg.campaign(row["campaign_id"])
        if status == "new":
            _step(conn, "fetch", vid, lambda m=meta: fetch_audio(cfg, m))
        elif status == "audio_downloaded":
            wav = cfg.dir("sources") / f"{vid}.wav"
            _step(conn, "transcribe", vid,
                  lambda w=wav, c=campaign: transcribe(cfg, vid, w, c.vocabulary))  # fmt: skip
        elif status == "transcribed":
            try:
                _step(conn, "pick", vid, lambda c=campaign: pick(
                    cfg, conn, vid, c, transcripts / f"{vid}.words.json",
                    transcripts / f"{vid}.compact.txt"))  # fmt: skip
            except ClaudeCLIError as exc:
                log.error("claude -p failed; %s stays at 'transcribed': %s", vid, exc)
                return "claude"
        elif status == "picked":
            _render_video(cfg, conn, row, deliver_now, should_stop)
            if db.get_video(conn, vid)["status"] == "picked":
                # stopped between clips, or failed clips wait for a retry on a later run
                return "stopped" if should_stop() else "ok"
        else:  # rendered
            _cleanup_one(cfg, conn, vid)
    if delivered:
        progress.say(f"✅ {delivered} clip(s) from <b>{html.escape(title)}</b>")
    elif db.get_video(conn, vid)["status"] == db.FAILED:
        error = (db.get_video(conn, vid)["error"] or "")[:200]
        progress.say(f"⚠️ No clips from <b>{html.escape(title)}</b>: {html.escape(error)}")
    return "ok"


def _deliver_waiting(inbox: Inbox | None, limit: int) -> int:
    if inbox is None:
        return 0
    with db.stage_run(inbox.conn, "deliver", "-"):
        return deliver(inbox, limit)[1]


def run_continuous(
    cfg: Config,
    conn: sqlite3.Connection,
    max_videos: int | None = None,
    only_video: str | None = None,
    next_video: Callable[[Config, sqlite3.Connection], str | None] | None = None,
) -> int:
    """Video after video until stopped, out of new videos, at the cap, Claude fails or the
    disk is full (PLAN.md §17). Clips go to Telegram as soon as each is rendered. The
    stop reason goes to data/run.status.json and the chat. Returns an exit code."""
    max_videos = max_videos or cfg.pipeline.max_videos_per_run
    next_video = next_video or discover_next
    inbox = open_inbox(cfg, conn)
    progress = Progress(cfg, conn, inbox, max_videos, only_video)

    def should_stop() -> bool:
        return runstate.stop_requested(cfg.data_dir)

    db.sync_campaigns(conn, cfg.campaigns)
    requeued = db.requeue_failed(conn, cfg.pipeline.max_retries)
    if requeued:
        log.info("retrying %d failed item(s)", requeued)
    if inbox and not lock_held(cfg.data_dir, BOT_LOCK):
        _poll_once(inbox)  # the bot is not running: apply buttons pressed meanwhile

    handled: set[str] = set()
    code, reason = EXIT_OK, ""
    try:
        while True:
            stage_rerender(cfg, conn)  # layout buttons pressed meanwhile
            _deliver_waiting(inbox, cfg.telegram.max_per_run)
            if should_stop():
                reason = "stopped by you"
                break
            ensure_free_space(cfg.data_dir, cfg.machine.min_free_gb)
            if only_video:
                row = db.get_video(conn, only_video) if only_video not in handled else None
                if row is None or row["status"] not in db.PIPELINE[:-1]:
                    reason = (
                        "that video is done" if handled else f"{only_video} has nothing left to do"
                    )
                    break
            else:
                if progress.videos >= max_videos:
                    reason = f"reached the limit of {max_videos} video(s)"
                    break
                row = _next_in_progress(conn, handled)
                if row is None:
                    progress.set(current={"id": "", "title": "", "stage": "looking for videos"})
                    with db.stage_run(conn, "discover", "-"):
                        vid = next_video(cfg, conn)
                    row = db.get_video(conn, vid) if vid else None
                if row is None:
                    reason = "no new videos left"
                    break
            handled.add(row["id"])
            outcome = process_video(cfg, conn, row["id"], progress, should_stop)
            progress.videos += 1
            progress.set()
            if outcome == "claude":
                code, reason = EXIT_CLAUDE, "claude -p failed (credit used up?)"
                break
            if outcome == "stopped":
                reason = "stopped by you"
                break
    except InsufficientDiskSpace as exc:
        code, reason = EXIT_DISK, f"low disk space: {exc}"
    except (SystemExit, KeyboardInterrupt):  # Stop now (SIGTERM) or Ctrl-C
        runstate.clear_stop(cfg.data_dir)
        progress.finish("stopped now")
        progress.say(f"⏹ Run stopped now · {progress.videos} video(s), {progress.clips} clip(s); "
                     "the current video resumes on the next Start")  # fmt: skip
        raise
    except Exception as exc:
        reason = f"crashed: {type(exc).__name__}: {exc}"[:200]
        progress.finish(reason)
        progress.say(f"💥 Run {html.escape(reason)}")
        raise
    _deliver_waiting(inbox, cfg.telegram.max_per_run)
    runstate.clear_stop(cfg.data_dir)
    progress.finish(reason)
    log.info("run finished: %s (%d videos, %d clips)", reason, progress.videos, progress.clips)
    progress.say(f"⏹ Run finished: {html.escape(reason)} · {progress.videos} video(s), "
                 f"{progress.clips} clip(s), ${progress.cost():.2f} Claude")  # fmt: skip
    return code


def add_url(cfg: Config, conn: sqlite3.Connection, campaign_id: str | None, url: str) -> str:
    """Add one video (from a joined campaign); returns its id."""
    campaign = cfg.campaign(campaign_id)
    meta = probe(url, campaign.id)
    added = db.add_video(conn, meta.id, campaign.id, meta.url, meta.title, meta.duration_sec,
                         meta.upload_date)  # fmt: skip
    if not added:
        row = db.get_video(conn, meta.id)
        log.info("%s is already known (status %s)", meta.id, row["status"] if row else "?")
    return meta.id


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


def rerender_all(cfg: Config, conn: sqlite3.Connection, new_hook: bool = False) -> int:
    """Render every rendered or delivered clip again with the current styling."""
    ids = [r["id"] for status in ("rendered", "delivered") for r in db.clips_in(conn, status)]
    log.info("re-rendering %d clip(s)", len(ids))
    failed = [cid for cid in ids if rerender(cfg, conn, cid, None, new_hook) != EXIT_OK]
    if failed:
        log.error("failed: %s", ", ".join(failed))
    return EXIT_FAILURES if failed else EXIT_OK


def rerender(
    cfg: Config,
    conn: sqlite3.Connection,
    clip_id: str,
    layout: str | None,
    new_hook: bool = False,
) -> int:
    """Render one already-picked clip again, optionally switching its layout and asking
    Claude for a new hook. The segment is downloaded again (sources are deleted after
    the first render)."""
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
    if new_hook:
        video = db.get_video(conn, video_id)
        try:
            head = regenerate_headline(cfg, conn, clip, cfg.campaign(video["campaign_id"]),
                                       words, video["title"] or "")  # fmt: skip
        except (ClaudeCLIError, ValueError) as exc:
            log.error("%s: no new hook: %s", clip.id, exc)
            return EXIT_CLAUDE
        log.info("%s: hook %r -> %r (accent %r, context %r)", clip.id, clip.hook_text,
                 head.hook_text, head.accent_word, head.context_line)  # fmt: skip
        clip.hook_text = head.hook_text
        clip.accent_word = head.accent_word
        clip.context_line = head.context_line
        save_clips(clips, clips_path)
        db.update_clip(conn, clip.id, hook_text=head.hook_text)
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
    parser.add_argument(
        "--layout", choices=["framed", "crop", "fit"], help="with --rerender: new layout"
    )
    parser.add_argument(
        "--new-hook", action="store_true", help="with --rerender(-all): ask Claude for a new hook"
    )
    parser.add_argument("--campaign", help="with --url: campaign id (default: first active)")
    parser.add_argument("--max-videos", type=int,
                        help="stop after this many source videos (default "
                             "pipeline.max_videos_per_run)")  # fmt: skip
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args(argv)
    if args.layout and not args.rerender:
        parser.error("--layout needs --rerender (new clips use render.default_layout)")
    if args.new_hook and not (args.rerender or args.rerender_all):
        parser.error("--new-hook needs --rerender or --rerender-all")

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
                    return rerender(cfg, conn, args.rerender, args.layout, args.new_hook)
                if args.rerender_all:
                    return rerender_all(cfg, conn, args.new_hook)
                only = add_url(cfg, conn, args.campaign, args.url) if args.url else None
                return run_continuous(cfg, conn, args.max_videos, only)
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
