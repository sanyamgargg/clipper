"""Status report: videos and clips by status, and this month's Claude cost.

python -m clipper.status
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from clipper import db
from clipper.config import load_config

VIDEO_ORDER = (*db.PIPELINE, db.FAILED, db.SKIPPED_TOO_LONG)
CLIP_ORDER = ("picked", "rendered", "rerender", "delivered", "posted", "rejected", db.FAILED)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows, strict=True)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    return lines + [fmt.format(*row).rstrip() for row in rows]


def _counts(conn: sqlite3.Connection, table: str, order: tuple[str, ...]) -> list[list[str]]:
    counts = {
        row["status"]: row["n"]
        for row in conn.execute(f"SELECT status, COUNT(*) AS n FROM {table} GROUP BY status")
    }
    known = [s for s in order if s in counts]
    return [[s, str(counts[s])] for s in known + sorted(set(counts) - set(order))]


def report(conn: sqlite3.Connection, month: str | None = None) -> str:
    """`month` is 'YYYY-MM' (UTC); default: the current month."""
    month = month or datetime.now(UTC).strftime("%Y-%m")
    lines = ["VIDEOS BY STATUS"]
    lines += _table(["status", "count"], _counts(conn, "videos", VIDEO_ORDER)) or []

    videos = conn.execute(
        "SELECT v.*, (SELECT COUNT(*) FROM clips c WHERE c.video_id = v.id) AS clips,"
        " (SELECT COUNT(*) FROM clips c WHERE c.video_id = v.id AND c.status NOT IN"
        " ('picked', 'failed')) AS rendered"
        " FROM videos v WHERE v.status != ? ORDER BY v.created_at, v.rowid",
        (db.SKIPPED_TOO_LONG,),
    ).fetchall()
    if videos:
        rows = [
            [
                v["id"],
                v["campaign_id"] or "",
                v["status"] + (f" (at {v['failed_status']})" if v["failed_status"] else ""),
                str(v["attempts"] or 0),
                f"{v['rendered']}/{v['clips']}",
                f"{(v['duration_sec'] or 0) / 60:.0f}m",
                (v["title"] or "")[:44],
            ]
            for v in videos
        ]
        lines += ["", "VIDEOS"]
        lines += _table(["id", "campaign", "status", "tries", "clips", "len", "title"], rows)
        errors = [f"  {v['id']}: {v['error']}" for v in videos if v["error"]]
        if errors:
            lines += ["", "ERRORS", *errors]

    lines += ["", "CLIPS BY STATUS"]
    lines += _table(["status", "count"], _counts(conn, "clips", CLIP_ORDER))

    cost = conn.execute(
        "SELECT COUNT(*) AS calls, COALESCE(SUM(input_tokens), 0) AS tin,"
        " COALESCE(SUM(output_tokens), 0) AS tout, COALESCE(SUM(cost_usd), 0) AS usd,"
        " COALESCE(SUM(exit_code != 0), 0) AS failed FROM llm_calls WHERE created_at LIKE ?",
        (month + "%",),
    ).fetchone()
    summary = (
        f"CLAUDE {month}: {cost['calls']} calls ({cost['failed']} failed), "
        f"{cost['tin']:,} in / {cost['tout']:,} out tokens, ${cost['usd']:.2f}"
    )
    lines += ["", summary]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clipper.status", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="path to config.yaml")
    args = parser.parse_args(argv)
    conn = db.connect(load_config(args.config).data_dir)
    try:
        sys.stdout.write(report(conn))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
