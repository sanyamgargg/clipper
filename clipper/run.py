"""Orchestrator. M1: one video end to end.

    python -m clipper.run --url <video> [--campaign tjr]
    python -m clipper.run --rerender <clip_id> [--layout crop|fit]

fetch (audio only) -> transcribe -> pick -> per clip: fetch segment -> render.
Strictly sequential: one heavy step at a time. Safe to rerun: finished steps are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

from clipper import db
from clipper.config import Campaign, Config, load_config
from clipper.disk import InsufficientDiskSpace, ensure_free_space
from clipper.llm.claude_cli import ClaudeCLIError
from clipper.stages.fetch import VideoMeta, fetch_audio, fetch_segment, probe
from clipper.stages.pick import Clip, load_clips, pick, save_clips
from clipper.stages.render import render_clip
from clipper.stages.transcribe import transcribe
from clipper.timing import Word, load_words

log = logging.getLogger("clipper.run")


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
            segment = fetch_segment(cfg, meta, index, clip.start_sec, clip.end_sec)
            render_clip(cfg, clip, segment, words)
            segment.unlink()  # source video is deleted as soon as its clip exists
        ok = True
    except InsufficientDiskSpace:
        raise
    except Exception:
        log.exception("clip %s failed", clip.id)
        ok = False
    db.upsert_clip(conn, clip.id, clip.video_id, clip.start_sec, clip.end_sec, clip.hook_text,
                   clip.title, clip.reason, clip.score, status="rendered" if ok else "failed",
                   file_path=str(out) if ok else None, layout=clip.layout)  # fmt: skip
    return ok


def run_url(cfg: Config, conn: sqlite3.Connection, campaign: Campaign, url: str) -> int:
    """Process one video. Returns a process exit code (0 = all clips rendered)."""
    ensure_free_space(cfg.data_dir, cfg.machine.min_free_gb)
    transcripts = cfg.dir("transcripts")
    wav: Path | None = None

    with db.stage_run(conn, "fetch", url):
        meta = probe(url, campaign.id)
        db.upsert_video(conn, meta.id, campaign.id, meta.url, meta.title, meta.duration_sec,
                        status="downloaded")  # fmt: skip
        if not (transcripts / f"{meta.id}.words.json").exists():
            wav = fetch_audio(cfg, meta)

    try:
        with db.stage_run(conn, "transcribe", meta.id):
            words_path, compact_path = transcribe(cfg, meta.id, wav or Path())
        db.set_video_status(conn, meta.id, "transcribed")

        try:
            with db.stage_run(conn, "pick", meta.id):
                clips = pick(cfg, conn, meta.id, campaign, words_path, compact_path)
        except ClaudeCLIError as exc:
            # e.g. credit exhausted: keep the video at `transcribed`, try again another night
            log.error("pick stopped, video stays at 'transcribed': %s", exc)
            return 2
        db.set_video_status(conn, meta.id, "picked")

        words = load_words(words_path)
        failed = 0
        for index, clip in enumerate(clips, start=1):
            if (cfg.dir("clips") / f"{clip.id}.mp4").exists():
                log.info("clip already rendered: %s", clip.id)
            elif not _render_one(cfg, conn, meta, index, clip, words):
                failed += 1

        if failed:
            db.set_video_status(conn, meta.id, "failed", f"{failed} of {len(clips)} clips failed")
            return 1
        (cfg.dir("sources") / f"{meta.id}.wav").unlink(missing_ok=True)
        db.set_video_status(conn, meta.id, "done")
        log.info("done: %d clips in %s", len(clips), cfg.dir("clips"))
        return 0
    except Exception as exc:
        db.set_video_status(conn, meta.id, "failed", f"{type(exc).__name__}: {exc}")
        raise


def rerender(cfg: Config, conn: sqlite3.Connection, clip_id: str, layout: str | None) -> int:
    """Render one already-picked clip again, optionally switching its layout. The segment
    is downloaded again (sources are deleted after the first render)."""
    ensure_free_space(cfg.data_dir, cfg.machine.min_free_gb)
    video_id = clip_id.rsplit("_", 1)[0]
    clips_path = cfg.dir("transcripts") / f"{video_id}.clips.json"
    info_path = cfg.dir("sources") / f"{video_id}.info.json"
    if not clips_path.exists() or not info_path.exists():
        log.error("unknown clip %s: no picks or video info for %s", clip_id, video_id)
        return 1
    clips = load_clips(clips_path)
    index = next((i for i, c in enumerate(clips, start=1) if c.id == clip_id), None)
    if index is None:
        log.error("unknown clip %s (have: %s)", clip_id, ", ".join(c.id for c in clips))
        return 1
    clip = clips[index - 1]
    if layout and layout != clip.layout:
        log.info("%s: layout %s -> %s", clip.id, clip.layout, layout)
        clip.layout = layout
        save_clips(clips, clips_path)

    words = load_words(cfg.dir("transcripts") / f"{video_id}.words.json")
    ok = _render_one(cfg, conn, VideoMeta.load(info_path), index, clip, words)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clipper.run", description=__doc__.splitlines()[0])
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--url", help="video URL from a joined campaign")
    what.add_argument("--rerender", metavar="CLIP_ID", help="render one picked clip again")
    parser.add_argument("--layout", choices=["crop", "fit"], help="with --rerender: new layout")
    parser.add_argument("--campaign", help="campaign id in config.yaml (default: first active)")
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
    conn = db.connect(cfg.data_dir)
    try:
        if args.rerender:
            return rerender(cfg, conn, args.rerender, args.layout)
        return run_url(cfg, conn, cfg.campaign(args.campaign), args.url)
    except InsufficientDiskSpace as exc:
        log.error("stopped: %s", exc)
        return 3
    except Exception:
        log.exception("run failed")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
