"""discover: pick the next source video to clip -> one row in `videos` (new).

Continuous runs (PLAN.md §17) take one video at a time: the newest unseen video across
the active campaigns' sources, which walks back into the backlog as newer ones are done,
but never further than pipeline.max_age_days.

Channel listings are newest-first and cheap but carry no upload dates. So per source only
its first unseen usable entry (its "frontier") is probed for a date; the newest frontier
wins. A frontier older than max_age_days means everything after it is older too, so that
source is done. Videos over pipeline.max_source_minutes are recorded as skipped_too_long.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, replace
from datetime import date, timedelta

from clipper import db
from clipper.config import Config
from clipper.stages.fetch import probe, ytdlp

log = logging.getLogger(__name__)

LIST_DEPTH = 60  # newest entries looked at per source (about a month of uploads)
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


def frontier(entries: list[Candidate], known: set[str],
             max_minutes: float) -> tuple[Candidate | None, list[Candidate]]:  # fmt: skip
    """The first unseen entry short enough to clip, and the too-long ones passed on the way."""
    too_long: list[Candidate] = []
    for c in entries:
        if c.id in known:
            continue
        if c.duration_sec > max_minutes * 60:
            too_long.append(c)
            continue
        return c, too_long
    return None, too_long


def too_old(upload_date: str, max_age_days: float, today: date) -> bool:
    if not upload_date:
        return False  # unknown: let it through, it is the best we have
    uploaded = date(int(upload_date[:4]), int(upload_date[4:6]), int(upload_date[6:8]))
    return today - uploaded > timedelta(days=max_age_days)


def discover_next(cfg: Config, conn: sqlite3.Connection, today: date | None = None) -> str | None:
    """Add the next video to clip (status new) and return its id; None if every source
    has nothing unseen within max_age_days."""
    p = cfg.pipeline
    today = today or date.today()
    known = db.known_video_ids(conn)
    fronts: list[Candidate] = []
    for campaign in cfg.campaigns:
        if not campaign.active:
            continue
        for source in campaign.sources:
            try:
                entries = list_source(source, campaign.id)
            except RuntimeError as exc:
                log.warning("cannot list %s: %s", source, exc)
                continue
            front, too_long = frontier(entries, known, p.max_source_minutes)
            for c in too_long:
                log.info("skipping %s (%.0f min > %g): %s", c.id, c.duration_sec / 60,
                         p.max_source_minutes, c.title)  # fmt: skip
                db.add_video(conn, c.id, c.campaign_id, c.url, c.title, c.duration_sec,
                             status=db.SKIPPED_TOO_LONG)  # fmt: skip
                known.add(c.id)
            if front is None:
                continue
            try:
                front = replace(front, upload_date=probe(front.url, campaign.id).upload_date)
            except RuntimeError as exc:
                log.warning("cannot probe %s, treating it as oldest: %s", front.id, exc)
            if too_old(front.upload_date, p.max_age_days, today):
                log.info("%s: nothing unseen newer than %g days", source, p.max_age_days)
                continue
            fronts.append(front)
            known.add(front.id)  # the same video listed by two sources counts once
    if not fronts:
        return None
    # newest; undated ones count as oldest
    best = max(fronts, key=lambda c: c.upload_date or "0")  # first one wins a tie
    log.info("next video %s (%s, %.0f min): %s", best.id, best.upload_date or "no date",
             best.duration_sec / 60, best.title)  # fmt: skip
    db.add_video(conn, best.id, best.campaign_id, best.url, best.title, best.duration_sec,
                 best.upload_date)  # fmt: skip
    return best.id
