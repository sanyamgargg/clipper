"""SQLite state (data/clipper.db): schema, status machine and small helpers.

Video status machine:
    new -> audio_downloaded -> transcribed -> picked -> rendered -> done
A stage takes videos in one status and advances them to the next. A failure parks the
video in `failed` (with error text, attempts and the status it failed in); the next run
puts it back, up to `max_retries` times. `skipped_too_long` is terminal.
Clips: picked -> rendered, or failed (retried the same way).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

PIPELINE = ("new", "audio_downloaded", "transcribed", "picked", "rendered", "done")
FAILED = "failed"
SKIPPED_TOO_LONG = "skipped_too_long"

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns(id TEXT PRIMARY KEY, name TEXT, rate_per_1k REAL, active INT);
CREATE TABLE IF NOT EXISTS videos(
    id TEXT PRIMARY KEY, campaign_id TEXT, url TEXT, title TEXT, duration_sec REAL,
    status TEXT, error TEXT, created_at TEXT, updated_at TEXT,
    attempts INTEGER DEFAULT 0, failed_status TEXT, upload_date TEXT);
CREATE TABLE IF NOT EXISTS clips(
    id TEXT PRIMARY KEY, video_id TEXT, start_sec REAL, end_sec REAL, hook_text TEXT, title TEXT,
    reason TEXT, score REAL, status TEXT, file_path TEXT, created_at TEXT,
    layout TEXT DEFAULT 'crop', attempts INTEGER DEFAULT 0,
    telegram_message_id INTEGER, links_prompt_id INTEGER, delivered_at TEXT, done_at TEXT,
    message_deleted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
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

# columns added after a table first shipped: (table, column, definition)
MIGRATIONS = (
    ("clips", "layout", "TEXT DEFAULT 'crop'"),
    ("clips", "attempts", "INTEGER DEFAULT 0"),
    ("videos", "attempts", "INTEGER DEFAULT 0"),
    ("videos", "failed_status", "TEXT"),
    ("videos", "upload_date", "TEXT"),
    ("clips", "telegram_message_id", "INTEGER"),
    ("clips", "links_prompt_id", "INTEGER"),
    ("clips", "delivered_at", "TEXT"),
    ("clips", "done_at", "TEXT"),
    ("clips", "message_deleted", "INTEGER DEFAULT 0"),
)

# Clip statuses: picked -> rendered -> delivered (sent to Telegram) -> posted | rejected.
# `rerender` = asked for again (e.g. Fit layout); failed as for videos.
CLIP_COLUMNS = frozenset(
    {"status", "layout", "file_path", "telegram_message_id", "links_prompt_id",
     "delivered_at", "done_at", "message_deleted"}
)  # fmt: skip


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def next_status(status: str) -> str:
    """The status after `status` in the pipeline. ValueError if there is none."""
    if status not in PIPELINE[:-1]:
        raise ValueError(f"no status after {status!r}")
    return PIPELINE[PIPELINE.index(status) + 1]


def connect(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    # autocommit; wait for the lock when the Telegram bot and a run write at the same time
    conn = sqlite3.connect(data_dir / "clipper.db", isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    for table, column, definition in MIGRATIONS:
        if column not in {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    return conn


# --- run / llm logging ------------------------------------------------------


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


# --- campaigns ---------------------------------------------------------------


def sync_campaigns(conn: sqlite3.Connection, campaigns: Iterable) -> None:
    """Mirror config campaigns (id, name, rate_per_1k, active) into the table."""
    for c in campaigns:
        conn.execute(
            "INSERT INTO campaigns(id, name, rate_per_1k, active) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET name = excluded.name,"
            " rate_per_1k = excluded.rate_per_1k, active = excluded.active",
            (c.id, c.name, c.rate_per_1k, int(c.active)),
        )


# --- videos ------------------------------------------------------------------


def known_video_ids(conn: sqlite3.Connection) -> set[str]:
    return {row["id"] for row in conn.execute("SELECT id FROM videos")}


def add_video(
    conn: sqlite3.Connection,
    video_id: str,
    campaign_id: str,
    url: str,
    title: str,
    duration_sec: float,
    upload_date: str = "",
    status: str = "new",
) -> bool:
    """Insert a video unless its id is already known. Returns True if it was inserted."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO videos(id, campaign_id, url, title, duration_sec, upload_date,"
        " status, attempts, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
        (video_id, campaign_id, url, title, duration_sec, upload_date, status, now(), now()),
    )
    return cur.rowcount == 1


def get_video(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()


def videos_in(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    """Videos waiting in `status`, oldest first."""
    return conn.execute(
        "SELECT * FROM videos WHERE status = ? ORDER BY created_at, rowid", (status,)
    ).fetchall()


def advance_video(conn: sqlite3.Connection, video_id: str) -> str:
    """Move a video to the next pipeline status; clears its error and attempts."""
    row = get_video(conn, video_id)
    if row is None:
        raise KeyError(video_id)
    new = next_status(row["status"])
    conn.execute(
        "UPDATE videos SET status = ?, error = NULL, attempts = 0, failed_status = NULL,"
        " updated_at = ? WHERE id = ?",
        (new, now(), video_id),
    )
    return new


def fail_video(conn: sqlite3.Connection, video_id: str, error: str) -> None:
    """Park a video in `failed`, remembering the status it failed in."""
    conn.execute(
        "UPDATE videos SET failed_status = status, status = ?, error = ?,"
        " attempts = COALESCE(attempts, 0) + 1, updated_at = ? WHERE id = ? AND status != ?",
        (FAILED, error, now(), video_id, FAILED),
    )


def requeue_failed(conn: sqlite3.Connection, max_retries: int) -> int:
    """Put failed videos and clips back in line if they have retries left.
    Returns how many items were requeued."""
    videos = conn.execute(
        "UPDATE videos SET status = failed_status, failed_status = NULL, updated_at = ?"
        " WHERE status = ? AND failed_status IS NOT NULL AND attempts <= ?",
        (now(), FAILED, max_retries),
    ).rowcount
    clips = conn.execute(
        "UPDATE clips SET status = 'picked' WHERE status = ? AND attempts <= ?",
        (FAILED, max_retries),
    ).rowcount
    return videos + clips


def in_progress_count(conn: sqlite3.Connection, max_retries: int) -> int:
    """Videos still on their way to `done`, including failed ones that will be retried."""
    marks = ",".join("?" * len(PIPELINE[:-1]))
    return conn.execute(
        f"SELECT COUNT(*) FROM videos WHERE status IN ({marks})"
        " OR (status = ? AND failed_status IS NOT NULL AND attempts <= ?)",
        (*PIPELINE[:-1], FAILED, max_retries),
    ).fetchone()[0]


# --- clips -------------------------------------------------------------------


def add_clip(
    conn: sqlite3.Connection,
    clip_id: str,
    video_id: str,
    start_sec: float,
    end_sec: float,
    hook_text: str,
    title: str,
    reason: str,
    score: float,
    layout: str = "crop",
) -> None:
    """Insert a picked clip unless it already exists (never resets a rendered clip)."""
    conn.execute(
        "INSERT OR IGNORE INTO clips(id, video_id, start_sec, end_sec, hook_text, title, reason,"
        " score, status, layout, attempts, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'picked', ?, 0, ?)",
        (clip_id, video_id, start_sec, end_sec, hook_text, title, reason, score, layout, now()),
    )


def clip_rows(conn: sqlite3.Connection, video_id: str) -> dict[str, sqlite3.Row]:
    rows = conn.execute("SELECT * FROM clips WHERE video_id = ? ORDER BY id", (video_id,))
    return {row["id"]: row for row in rows}


def mark_clip_rendered(conn: sqlite3.Connection, clip_id: str, file_path: str, layout: str) -> None:
    conn.execute(
        "UPDATE clips SET status = 'rendered', file_path = ?, layout = ? WHERE id = ?",
        (file_path, layout, clip_id),
    )


def fail_clip(conn: sqlite3.Connection, clip_id: str) -> None:
    conn.execute(
        "UPDATE clips SET status = ?, attempts = COALESCE(attempts, 0) + 1 WHERE id = ?",
        (FAILED, clip_id),
    )


def get_clip(conn: sqlite3.Connection, clip_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()


def clips_in(conn: sqlite3.Connection, status: str, limit: int | None = None) -> list[sqlite3.Row]:
    """Clips in `status`, oldest first."""
    sql = "SELECT * FROM clips WHERE status = ? ORDER BY created_at, id"
    params: tuple = (status,)
    if limit is not None:
        sql += " LIMIT ?"
        params += (limit,)
    return conn.execute(sql, params).fetchall()


def update_clip(conn: sqlite3.Connection, clip_id: str, **fields: object) -> None:
    unknown = set(fields) - CLIP_COLUMNS
    if unknown:
        raise ValueError(f"not clip columns: {sorted(unknown)}")
    sets = ", ".join(f"{name} = ?" for name in fields)
    conn.execute(f"UPDATE clips SET {sets} WHERE id = ?", (*fields.values(), clip_id))


def clip_for_message(conn: sqlite3.Connection, message_id: int) -> sqlite3.Row | None:
    """The clip whose Telegram video message, or links prompt, has this id."""
    return conn.execute(
        "SELECT * FROM clips WHERE telegram_message_id = ? OR links_prompt_id = ?",
        (message_id, message_id),
    ).fetchone()


def add_post(conn: sqlite3.Connection, clip_id: str, platform: str, url: str) -> bool:
    """Record where a clip was posted. Returns False if this URL was already known."""
    if conn.execute("SELECT 1 FROM posts WHERE clip_id = ? AND url = ?", (clip_id, url)).fetchone():
        return False
    conn.execute(
        "INSERT INTO posts(clip_id, platform, url, posted_at, submitted_to_campaign)"
        " VALUES (?, ?, ?, ?, 0)",
        (clip_id, platform, url, now()),
    )
    return True


def posts_for(conn: sqlite3.Connection, clip_id: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM posts WHERE clip_id = ? ORDER BY id", (clip_id,)).fetchall()


def get_setting(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings(key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
