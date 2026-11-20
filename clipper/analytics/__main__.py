"""Analytics, lite (read-only). The Telegram bot process runs the periodic `tick` itself;
these commands are for running things by hand.

python -m clipper.analytics poll [--all]           YouTube posts that are due (--all: every one)
python -m clipper.analytics digest|weekly [--send] print (or send) a report now
python -m clipper.analytics export                 write data/analytics/clips.csv

The YouTube API key lives in the Keychain:
    security add-generic-password -s clipper-youtube-key -a youtube -w <key>
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from clipper import db
from clipper.analytics.core import AnalyticsError, poll_youtube
from clipper.analytics.reports import daily_digest, export_csv, mvr_alert_text, weekly_report
from clipper.config import REPO_ROOT, load_config
from clipper.telegram.api import TelegramError
from clipper.telegram.inbox import open_inbox

log = logging.getLogger("clipper.analytics")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clipper.analytics", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("poll", help="check YouTube views now")
    p.add_argument("--all", action="store_true", help="every YouTube post, due or not")
    for name in ("digest", "weekly"):
        sub.add_parser(name).add_argument("--send", action="store_true", help="send to Telegram")
    ex = sub.add_parser("export", help="write data/analytics/clips.csv")
    ex.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    conn = db.connect(cfg.data_dir)
    now = datetime.now(UTC)
    try:
        if args.command == "export":
            out = args.out or cfg.dir("analytics") / "clips.csv"
            print(f"wrote {export_csv(cfg, conn, out)} row(s) to {out}")
            return 0
        ix = open_inbox(cfg, conn)
        if args.command == "poll":
            for alert in poll_youtube(cfg, conn, now, force=args.all):
                if ix:
                    ix.say(mvr_alert_text(alert))
                print(f"MVR: {alert.clip_id} {alert.views:,} views")
            rows = conn.execute(
                "SELECT clip_id, views FROM posts WHERE platform = 'youtube_shorts' ORDER BY posted_at"
            ).fetchall()
            for r in rows:
                print(f"{r['clip_id']}: {r['views'] if r['views'] is not None else '–'} views")
            return 0
        text = (daily_digest if args.command == "digest" else weekly_report)(cfg, conn, now)
        if args.send and ix:
            ix.say(text)
        else:
            print(text)
        return 0
    except (AnalyticsError, TelegramError) as exc:
        log.error("%s", exc)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
