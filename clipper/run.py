"""Orchestrator. M1: one video end to end.

    python -m clipper.run --url <video> [--campaign tjr]

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
from clipper.stages.fetch import fetch_audio, fetch_segment, probe
from clipper.stages.pick import pick
from clipper.stages.render import render_clip
from clipper.stages.transcribe import transcribe
from clipper.timing import load_words

log = logging.getLogger("clipper.run")


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
            out = cfg.dir("clips") / f"{clip.id}.mp4"
            if out.exists():
                log.info("clip already rendered: %s", out.name)
                continue
            try:
                with db.stage_run(conn, "render", clip.id):
                    segment = fetch_segment(cfg, meta, index, clip.start_sec, clip.end_sec)
                    render_clip(cfg, clip, segment, words)
                    segment.unlink()  # source video is deleted as soon as its clip exists
                status = "rendered"
            except InsufficientDiskSpace:
                raise
            except Exception:
                log.exception("clip %s failed", clip.id)
                status, failed = "failed", failed + 1
            db.upsert_clip(conn, clip.id, clip.video_id, clip.start_sec, clip.end_sec,
                           clip.hook_text, clip.title, clip.reason, clip.score, status=status,
                           file_path=str(out) if status == "rendered" else None)  # fmt: skip

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clipper.run", description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="video URL from a joined campaign")
    parser.add_argument("--campaign", help="campaign id in config.yaml (default: first active)")
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(cfg.dir("logs") / "run.log")],
    )
    campaign = cfg.campaign(args.campaign)
    conn = db.connect(cfg.data_dir)
    try:
        return run_url(cfg, conn, campaign, args.url)
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
