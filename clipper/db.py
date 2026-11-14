"""SQLite state (data/clipper.db): schema + small helpers."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns(id TEXT PRIMARY KEY, name TEXT, rate_per_1k REAL, active INT);
CREATE TABLE IF NOT EXISTS videos(
    id TEXT PRIMARY KEY, campaign_id TEXT, url TEXT, title TEXT, duration_sec REAL,
    status TEXT, error TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS clips(
    id TEXT PRIMARY KEY, video_id TEXT, start_sec REAL, end_sec REAL, hook_text TEXT, title TEXT,
    reason TEXT, score REAL, status TEXT, file_path TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS posts(
    id INTEGER PRIMARY KEY, clip_id TEXT, platform TEXT, url TEXT, posted_at TEXT, views INT,
    views_checked_at TEXT, submitted_to_campaign INT, campaign_status TEXT, earnings REAL);
CREATE TABLE IF NOT EXISTS runs(
    id INTEGER PRIMARY KEY, stage TEXT, item_id TEXT, started_at TEXT, ended_at TEXT,
    ok INT, error TEXT);
CREATE TABLE IF NOT EXISTS llm_calls(
    id INTEGER PRIMARY KEY, video_id TEXT, model TEXT, input_tokens INT, output_tokens INT,
    cost_usd REAL, exit_code INT, created_at TEXT);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(data_dir / "clipper.db", isolation_level=None)  # autocommit
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def stage_run(conn: sqlite3.Connection, stage: str, item_id: str) -> Iterator[None]:
    """Log one stage run (start, end, ok, error) to `runs`. Re-raises failures."""
    started = datetime.now(UTC)
    log.info("[%s] start %s", stage, item_id)
    cur = conn.execute(
        "INSERT INTO runs(stage, item_id, started_at) VALUES (?, ?, ?)",
        (stage, item_id, started.isoformat(timespec="seconds")),
    )
    ok, error = 1, None
    try:
        yield
    except BaseException as exc:
        ok, error = 0, f"{type(exc).__name__}: {exc}"
        raise
    finally:
        secs = (datetime.now(UTC) - started).total_seconds()
        conn.execute(
            "UPDATE runs SET ended_at = ?, ok = ?, error = ? WHERE id = ?",
            (now(), ok, error, cur.lastrowid),
        )
        log.info("[%s] %s %s in %.1f s", stage, "done" if ok else "FAILED", item_id, secs)


def log_llm_call(
    conn: sqlite3.Connection,
    video_id: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: float | None,
    exit_code: int,
) -> None:
    conn.execute(
        "INSERT INTO llm_calls(video_id, model, input_tokens, output_tokens, cost_usd, exit_code,"
        " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (video_id, model, input_tokens, output_tokens, cost_usd, exit_code, now()),
    )


def upsert_video(
    conn: sqlite3.Connection,
    video_id: str,
    campaign_id: str,
    url: str,
    title: str,
    duration_sec: float,
    status: str,
) -> None:
    conn.execute(
        "INSERT INTO videos(id, campaign_id, url, title, duration_sec, status, created_at,"
        " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET status = excluded.status, error = NULL,"
        " updated_at = excluded.updated_at",
        (video_id, campaign_id, url, title, duration_sec, status, now(), now()),
    )


def set_video_status(
    conn: sqlite3.Connection, video_id: str, status: str, error: str | None = None
) -> None:
    conn.execute(
        "UPDATE videos SET status = ?, error = ?, updated_at = ? WHERE id = ?",
        (status, error, now(), video_id),
    )


def upsert_clip(
    conn: sqlite3.Connection,
    clip_id: str,
    video_id: str,
    start_sec: float,
    end_sec: float,
    hook_text: str,
    title: str,
    reason: str,
    score: float,
    status: str,
    file_path: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO clips(id, video_id, start_sec, end_sec, hook_text, title, reason, score,"
        " status, file_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(id) DO UPDATE SET start_sec = excluded.start_sec,"
        " end_sec = excluded.end_sec, hook_text = excluded.hook_text, title = excluded.title,"
        " reason = excluded.reason, score = excluded.score, status = excluded.status,"
        " file_path = excluded.file_path",
        (clip_id, video_id, start_sec, end_sec, hook_text, title, reason, score, status,
         file_path, now()),
    )  # fmt: skip
