"""Post stats: YouTube public statistics via an API key, Instagram counts entered by hand.

YouTube schedule per post (by age): < 48 h every 2 h, 2-14 days every 12 h, 14-45 days
daily, then stop. Each check appends a `post_stats` row and updates posts.views.
Crossing the campaign's MVR (TJR: 5,000 views) sets posts.mvr_reached_at once and returns
an alert, for YouTube and manual counts alike.

The API key is read from the Keychain (or env CLIPPER_YOUTUBE_KEY). It is a URL query
parameter, so URLs are never logged and request errors keep only the exception type.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import requests

from clipper import db
from clipper.config import Campaign, Config

log = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)  # its debug lines contain the URL

VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
KEY_ENV = "CLIPPER_YOUTUBE_KEY"
YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
IG_PATH = re.compile(r"^/(?:[^/]+/)?(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)")
BATCH = 50  # ids per videos.list call (1 quota unit)
YOUTUBE, INSTAGRAM = "youtube_shorts", "instagram_reels"


class AnalyticsError(RuntimeError):
    pass


# --- ids, schedule, money --------------------------------------------------------------


def youtube_id(url: str) -> str | None:
    """The video id in a YouTube watch / shorts / youtu.be URL."""
    u = urlparse(url.strip())
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        vid = u.path.strip("/").split("/")[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        parts = [p for p in u.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live"):
            vid = parts[1]
        else:
            vid = (parse_qs(u.query).get("v") or [""])[0]
    else:
        return None
    return vid if YT_ID.match(vid) else None


def instagram_code(url: str) -> str | None:
    """The shortcode in an Instagram reel / post URL."""
    u = urlparse(url.strip())
    if not (u.hostname or "").lower().endswith("instagram.com"):
        return None
    m = IG_PATH.match(u.path)
    return m.group(1) if m else None


def same_post(a: str, b: str) -> bool:
    """Two URLs of the same post (share links carry extra query parameters)."""
    if a == b:
        return True
    ya, yb = youtube_id(a), youtube_id(b)
    if ya or yb:
        return ya == yb
    ia, ib = instagram_code(a), instagram_code(b)
    return bool(ia) and ia == ib


def poll_interval(age: timedelta, stop_after_days: float = 45) -> timedelta | None:
    """How often a post this old is checked; None = no longer."""
    if age < timedelta(hours=48):
        return timedelta(hours=2)
    if age < timedelta(days=14):
        return timedelta(hours=12)
    if age < timedelta(days=stop_after_days):
        return timedelta(hours=24)
    return None


def is_due(posted_at: datetime, last_checked: datetime | None, now: datetime,
           stop_after_days: float = 45) -> bool:  # fmt: skip
    every = poll_interval(now - posted_at, stop_after_days)
    if every is None:
        return False
    # 5 min slack so a tick a little early does not skip a whole interval
    return last_checked is None or now - last_checked >= every - timedelta(minutes=5)


def earnings_usd(views: int | None, campaign: Campaign) -> float:
    """Estimated payout: nothing below the MVR, then views/1000 × rate, capped."""
    if not views or views < campaign.mvr_views:
        return 0.0
    usd = views / 1000 * campaign.rate_per_1k
    cap = campaign.max_payout_usd
    return round(min(usd, cap) if cap else usd, 2)


def ts(value: str | None) -> datetime | None:
    if not value:
        return None
    t = datetime.fromisoformat(value)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def post_campaign(cfg: Config, conn: sqlite3.Connection, post: sqlite3.Row) -> Campaign | None:
    row = conn.execute(
        "SELECT v.campaign_id FROM clips c JOIN videos v ON v.id = c.video_id WHERE c.id = ?",
        (post["clip_id"],),
    ).fetchone()
    try:
        return cfg.campaign(row["campaign_id"]) if row else None
    except KeyError:
        return None


# --- YouTube ---------------------------------------------------------------------------


def youtube_key(cfg: Config) -> str | None:
    """The API key from env CLIPPER_YOUTUBE_KEY, else the Keychain; None if not set up."""
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        return key
    service = cfg.analytics.youtube_keychain_service
    try:
        proc = subprocess.run(["security", "find-generic-password", "-s", service, "-w"],
                              capture_output=True, text=True, check=False)  # fmt: skip
    except FileNotFoundError:  # not macOS
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def fetch_youtube(key: str, ids: list[str], session: requests.Session | None = None,
                  sleep: Callable[[float], None] = time.sleep) -> dict[str, dict[str, int]]:  # fmt: skip
    """views / likes / comments per video id (public statistics, batched by 50).
    Retries 429 / 5xx / network errors twice; ids YouTube does not return are left out."""
    session = session or requests.Session()
    out: dict[str, dict[str, int]] = {}
    for i in range(0, len(ids), BATCH):
        params = {"part": "statistics", "id": ",".join(ids[i : i + BATCH]), "key": key}
        for attempt in range(3):
            try:
                resp = session.get(VIDEOS_URL, params=params, timeout=30)
            except requests.RequestException as exc:  # str(exc) has the URL, i.e. the key
                error = AnalyticsError(f"YouTube: network error ({type(exc).__name__})")
                retry = True
            else:
                if resp.status_code == 200:
                    for item in resp.json().get("items", []):
                        s = item.get("statistics", {})
                        out[item["id"]] = {
                            "views": int(s.get("viewCount", 0) or 0),
                            "likes": int(s.get("likeCount", 0) or 0),
                            "comments": int(s.get("commentCount", 0) or 0),
                        }
                    break
                try:
                    reason = resp.json()["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    reason = ""
                error = AnalyticsError(f"YouTube: HTTP {resp.status_code} {reason}".strip())
                retry = resp.status_code == 429 or resp.status_code >= 500
            if not retry or attempt == 2:
                raise error
            sleep(2.0 ** (attempt + 1))
    return out


# --- recording --------------------------------------------------------------------------


@dataclass
class Alert:
    post_id: int
    clip_id: str
    title: str
    platform: str
    views: int
    mvr: int
    url: str


def record_views(cfg: Config, conn: sqlite3.Connection, post: sqlite3.Row, views: int,
                 now: datetime, source: str, likes: int | None = None,
                 comments: int | None = None) -> Alert | None:  # fmt: skip
    """Store a count; returns an MVR alert the first time the post crosses the MVR."""
    checked = now.isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO post_stats(post_id, checked_at, views, likes, comments, source)"
        " VALUES (?, ?, ?, ?, ?, ?)", (post["id"], checked, views, likes, comments, source),
    )  # fmt: skip
    campaign = post_campaign(cfg, conn, post)
    earnings = earnings_usd(views, campaign) if campaign else None
    conn.execute("UPDATE posts SET views = ?, views_checked_at = ?, earnings = ? WHERE id = ?",
                 (views, checked, earnings, post["id"]))  # fmt: skip
    mvr = campaign.mvr_views if campaign else 0
    if not mvr or views < mvr:
        return None
    first = conn.execute(
        "UPDATE posts SET mvr_reached_at = ? WHERE id = ? AND mvr_reached_at IS NULL",
        (checked, post["id"]),
    ).rowcount
    if not first:
        return None
    clip = db.get_clip(conn, post["clip_id"])
    title = (clip["title"] if clip else None) or post["clip_id"]
    return Alert(post["id"], post["clip_id"], title, post["platform"], views, mvr, post["url"])


def poll_youtube(cfg: Config, conn: sqlite3.Connection, now: datetime | None = None,
                 key: str | None = None, fetch: Callable[..., dict] = fetch_youtube,
                 force: bool = False) -> list[Alert]:  # fmt: skip
    """Check the YouTube posts that are due; returns new MVR alerts."""
    now = now or datetime.now(UTC)
    posts = conn.execute(
        "SELECT p.*, (SELECT MAX(checked_at) FROM post_stats s WHERE s.post_id = p.id"
        " AND s.source = 'youtube_api') AS last_checked FROM posts p WHERE p.platform = ?"
        " ORDER BY p.posted_at", (YOUTUBE,),
    ).fetchall()  # fmt: skip
    due: dict[str, sqlite3.Row] = {}
    for post in posts:
        posted = ts(post["posted_at"])
        if posted is None or not (force or is_due(posted, ts(post["last_checked"]), now,
                                                  cfg.analytics.stop_after_days)):  # fmt: skip
            continue
        vid = post["platform_video_id"] or youtube_id(post["url"] or "")
        if vid is None:
            log.warning("post %s: no YouTube id in %s", post["id"], post["url"])
            continue
        if not post["platform_video_id"]:
            conn.execute("UPDATE posts SET platform_video_id = ? WHERE id = ?", (vid, post["id"]))
        due[vid] = post
    if not due:
        return []
    key = key or youtube_key(cfg)
    if not key:
        log.warning("no YouTube API key: security add-generic-password -s %s -a youtube -w",
                    cfg.analytics.youtube_keychain_service)  # fmt: skip
        return []
    stats = fetch(key, list(due))
    alerts = []
    for vid, post in due.items():
        s = stats.get(vid)
        if s is None:
            log.warning("YouTube returned nothing for %s (private or deleted?)", vid)
            continue
        alert = record_views(cfg, conn, post, s["views"], now, "youtube_api", s["likes"],
                             s["comments"])  # fmt: skip
        if alert:
            alerts.append(alert)
    log.info("youtube: checked %d post(s), %d MVR alert(s)", len(stats), len(alerts))
    return alerts


COUNT_RE = re.compile(r"^([\d.,]+)([km]?)$")


def parse_count(text: str) -> int | None:
    """'12345', '12,345', '12.3k', '1.2M' -> int; None if it is not a count."""
    m = COUNT_RE.match(text.strip().lower())
    if not m:
        return None
    number, unit = m.group(1).replace(",", ""), m.group(2)
    try:
        if unit:
            return round(float(number) * (1000 if unit == "k" else 1_000_000))
        return int(number) if number.isdigit() else None
    except ValueError:
        return None


def find_post(conn: sqlite3.Connection, ref: str, prefer: str = INSTAGRAM) -> sqlite3.Row | None:
    """A post by URL (matched by video id / shortcode, so share links work) or by clip id
    (its `prefer` platform post, else its only post)."""
    ref = ref.strip()
    if ref.startswith("http"):
        for row in conn.execute("SELECT * FROM posts ORDER BY id DESC"):
            if same_post(row["url"] or "", ref):
                return row
        return None
    rows = conn.execute("SELECT * FROM posts WHERE clip_id = ? ORDER BY id", (ref,)).fetchall()
    preferred = [r for r in rows if r["platform"] == prefer]
    if preferred:
        return preferred[-1]
    return rows[0] if len(rows) == 1 else None


# --- tick (called by the bot process) ---------------------------------------------------


def report_due(now_local: datetime, at: str, last_sent: str | None,
               weekday: int | None = None) -> bool:  # fmt: skip
    """True once a day (or once a week on `weekday`) from local time `at` (HH:MM) on."""
    hour, minute = (int(x) for x in at.split(":"))
    if weekday is not None and now_local.weekday() != weekday:
        return False
    if (now_local.hour, now_local.minute) < (hour, minute):
        return False
    return last_sent != now_local.date().isoformat()


def tick(cfg: Config, conn: sqlite3.Connection, say: Callable[[str], Any],
         now: datetime | None = None, fetch: Callable[..., dict] = fetch_youtube,
         key: str | None = None) -> int:  # fmt: skip
    """Poll YouTube, send MVR alerts, and the daily / weekly report when due. Never
    raises: analytics must not disturb the bot or the pipeline. Returns alerts sent."""
    from clipper.analytics.reports import daily_digest, mvr_alert_text, weekly_report

    now = now or datetime.now(UTC)
    sent = 0
    try:
        for alert in poll_youtube(cfg, conn, now, key=key, fetch=fetch):
            say(mvr_alert_text(alert))
            sent += 1
    except Exception as exc:  # noqa: BLE001 - report, never crash the caller
        log.warning("analytics poll failed: %s", exc)
    a = cfg.analytics
    local = now.astimezone(ZoneInfo(a.timezone))
    reports = (("digest_sent", None, daily_digest), ("weekly_sent", a.weekly_day, weekly_report))
    for setting, weekday, build in reports:
        if not report_due(local, a.digest_time, db.get_setting(conn, setting), weekday):
            continue
        try:
            say(build(cfg, conn, now))
            db.set_setting(conn, setting, local.date().isoformat())
        except Exception as exc:  # noqa: BLE001
            log.warning("could not send %s: %s", setting, exc)
    return sent
