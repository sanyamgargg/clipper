"""discover: find new videos in the active campaigns' sources -> rows in `videos` (new).

The flat channel listing is cheap but has no upload dates, so: shortlist the first unseen
videos of each source, probe only those for their date, then keep the newest across all
campaigns. Videos over pipeline.max_source_minutes are recorded as skipped_too_long.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, replace

from clipper import db
from clipper.config import Config
from clipper.stages.fetch import probe, ytdlp

log = logging.getLogger(__name__)

LIST_DEPTH = 15  # newest entries looked at per source
NOT_READY = ("is_live", "is_upcoming")


@dataclass(frozen=True)
class Candidate:
    id: str
    url: str
    title: str
    duration_sec: float
    campaign_id: str
    upload_date: str = ""  # YYYYMMDD, known only after a probe


def list_source(url: str, campaign_id: str, depth: int = LIST_DEPTH) -> list[Candidate]:
    """Newest-first entries of a channel tab or playlist. Live/upcoming ones are left out."""
    out = ytdlp("--flat-playlist", "--playlist-end", str(depth), "--dump-json", url)
    candidates: list[Candidate] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("live_status") in NOT_READY or not entry.get("duration"):
            continue
        candidates.append(
            Candidate(
                id=entry["id"],
                url=entry.get("url") or entry.get("webpage_url") or "",
                title=entry.get("title") or "",
                duration_sec=float(entry["duration"]),
                campaign_id=campaign_id,
            )
        )
    return candidates


def shortlist(
    entries: list[Candidate], known: set[str], limit: int, max_minutes: float
) -> tuple[list[Candidate], list[Candidate]]:
    """Walk a newest-first listing: skip known ids, set aside videos that are too long,
    stop after `limit` usable ones. Returns (usable, too_long)."""
    usable: list[Candidate] = []
    too_long: list[Candidate] = []
    seen = set(known)
    for c in entries:
        if len(usable) >= limit:
            break
        if c.id in seen:
            continue
        seen.add(c.id)
        if c.duration_sec > max_minutes * 60:
            too_long.append(c)
        else:
            usable.append(c)
    return usable, too_long


def newest(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """The `limit` most recently uploaded. Undated ones go last; ties keep listing order."""
    return sorted(candidates, key=lambda c: c.upload_date, reverse=True)[:limit]


def discover(cfg: Config, conn: sqlite3.Connection) -> int:
    """Add up to videos_per_night new videos, minus those still in progress (so rerunning
    after a crash resumes the same videos instead of piling on more). Returns how many."""
    p = cfg.pipeline
    capacity = p.videos_per_night - db.in_progress_count(conn, p.max_retries)
    if capacity <= 0:
        log.info("no new videos fetched: %d already in progress", p.videos_per_night - capacity)
        return 0

    known = db.known_video_ids(conn)
    usable: list[Candidate] = []
    for campaign in cfg.campaigns:
        if not campaign.active:
            continue
        for source in campaign.sources:
            try:
                entries = list_source(source, campaign.id)
            except RuntimeError as exc:
                log.warning("cannot list %s: %s", source, exc)
                continue
            ok, too_long = shortlist(entries, known, capacity, p.max_source_minutes)
            for c in too_long:
                log.info("skipping %s (%.0f min > %g): %s", c.id, c.duration_sec / 60,
                         p.max_source_minutes, c.title)  # fmt: skip
                db.add_video(conn, c.id, c.campaign_id, c.url, c.title, c.duration_sec,
                             status=db.SKIPPED_TOO_LONG)  # fmt: skip
            known |= {c.id for c in ok + too_long}
            usable += ok

    dated: list[Candidate] = []
    for c in usable:
        try:
            dated.append(replace(c, upload_date=probe(c.url, c.campaign_id).upload_date))
        except RuntimeError as exc:
            log.warning("cannot probe %s, treating it as oldest: %s", c.id, exc)
            dated.append(c)

    chosen = newest(dated, capacity)
    for c in chosen:
        log.info("new video %s (%s, %.0f min): %s", c.id, c.upload_date or "no date",
                 c.duration_sec / 60, c.title)  # fmt: skip
        db.add_video(conn, c.id, c.campaign_id, c.url, c.title, c.duration_sec, c.upload_date)
    return len(chosen)
