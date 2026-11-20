"""Telegram reports (HTML) and the CSV export, from the counts in post_stats.

- MVR alert: "🎯 <title> hit 5,000 views on Instagram — check approval on Content Rewards"
- daily digest: yesterday's posts with their latest views, totals, estimated earnings,
  and the Instagram posts that need a /views count
- weekly report: top / bottom 3, median views at 48 h by length bucket, story flag and
  layout, estimated vs paid earnings, a reminder to check audience countries
"""

from __future__ import annotations

import csv
import html
import sqlite3
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from clipper.analytics.core import INSTAGRAM, Alert, earnings_usd, post_campaign, ts
from clipper.config import Config
from clipper.stages.pick import Clip, load_clips

LABEL = {"youtube_shorts": "YouTube", "instagram_reels": "Instagram", "tiktok": "TikTok"}
esc = html.escape


@dataclass
class PostView:
    id: int
    clip_id: str
    title: str
    platform: str  # youtube_shorts | instagram_reels | ...
    url: str
    posted_at: datetime
    duration_sec: float | None
    layout: str
    story: bool
    hook: str
    score: float | None
    views: int | None  # latest count
    views_48h: int | None
    last_manual: datetime | None
    earnings: float
    paid_usd: float | None
    approved: bool

    @property
    def label(self) -> str:
        return LABEL.get(self.platform, self.platform)


def views_at(conn: sqlite3.Connection, post_id: int, posted_at: datetime,
             hours: float) -> int | None:  # fmt: skip
    """Views `hours` after posting: the last count up to then (1 h grace)."""
    cutoff = (posted_at + timedelta(hours=hours + 1)).isoformat(timespec="seconds")
    row = conn.execute(
        "SELECT views FROM post_stats WHERE post_id = ? AND checked_at <= ? AND views IS NOT NULL"
        " ORDER BY checked_at DESC LIMIT 1", (post_id, cutoff),
    ).fetchone()  # fmt: skip
    return row["views"] if row else None


def length_bucket(sec: float | None) -> str:
    if sec is None:
        return "?"
    if sec < 25:
        return "<25 s"
    return "25–35 s" if sec <= 35 else ">35 s"


def post_views(cfg: Config, conn: sqlite3.Connection) -> list[PostView]:
    rows = conn.execute(
        "SELECT p.*, c.title AS clip_title, c.layout AS clip_layout, c.hook_text, c.score,"
        " c.video_id, (SELECT MAX(s.checked_at) FROM post_stats s WHERE s.post_id = p.id"
        " AND s.source = 'manual') AS last_manual"
        " FROM posts p LEFT JOIN clips c ON c.id = p.clip_id ORDER BY p.posted_at, p.id"
    ).fetchall()
    picks: dict[str, dict[str, Clip]] = {}
    out = []
    for r in rows:
        posted = ts(r["posted_at"])
        if posted is None:
            continue
        vid = r["video_id"]
        if vid and vid not in picks:
            path = cfg.dir("transcripts") / f"{vid}.clips.json"
            picks[vid] = {c.id: c for c in load_clips(path)} if path.exists() else {}
        clip = picks.get(vid, {}).get(r["clip_id"])
        campaign = post_campaign(cfg, conn, r)
        out.append(PostView(
            id=r["id"], clip_id=r["clip_id"], title=r["clip_title"] or r["clip_id"],
            platform=r["platform"] or "other", url=r["url"] or "", posted_at=posted,
            duration_sec=r["duration_sec"], layout=r["clip_layout"] or "?",
            story=bool(clip and clip.story), hook=r["hook_text"] or "", score=r["score"],
            views=r["views"], views_48h=views_at(conn, r["id"], posted, 48),
            last_manual=ts(r["last_manual"]),
            earnings=earnings_usd(r["views"], campaign) if campaign else 0.0,
            paid_usd=r["paid_usd"], approved=bool(r["approved"]),
        ))  # fmt: skip
    return out


def _n(v: float | None) -> str:
    return "–" if v is None else f"{v:,.0f}"


def mvr_alert_text(alert: Alert, marketplace: str = "Content Rewards") -> str:
    platform = LABEL.get(alert.platform, alert.platform)
    return (f"🎯 <b>{esc(alert.title)}</b> hit {alert.mvr:,} views on {platform} — check "
            f"approval on {esc(marketplace)}\n{esc(alert.url)}")  # fmt: skip


def needs_count(cfg: Config, posts: list[PostView], now: datetime) -> list[PostView]:
    """Live Instagram posts without a /views count in the last 24 h."""
    live = timedelta(days=cfg.analytics.stop_after_days)
    return [p for p in posts if p.platform == INSTAGRAM and now - p.posted_at < live
            and (p.last_manual is None or now - p.last_manual > timedelta(hours=24))]  # fmt: skip


def daily_digest(cfg: Config, conn: sqlite3.Connection, now: datetime) -> str:
    tz = ZoneInfo(cfg.analytics.timezone)
    start = datetime.combine(now.astimezone(tz).date() - timedelta(days=1), time(), tz)
    end = start + timedelta(days=1)
    posts = post_views(cfg, conn)
    yday = [p for p in posts if start <= p.posted_at < end]
    lines = [f"<b>📊 Daily digest — {start:%a %d %b}</b>"]
    if yday:
        lines.append(f"Posted yesterday ({len(yday)}):")
        lines += [f"• {esc(p.title)} · {p.label} · {_n(p.views)} views" for p in yday]
    else:
        lines.append("Nothing posted yesterday.")
    lines += [
        "",
        f"All posts: {_n(sum(p.views or 0 for p in posts))} views",
        f"Estimated earnings (posts over MVR): ${sum(p.earnings for p in posts):,.2f}",
    ]
    over = [p for p in posts if p.earnings > 0]
    if over:
        lines.append(f"Over MVR: {len(over)} post(s), {sum(p.approved for p in over)} approved")
    missing = needs_count(cfg, posts, now)
    if missing:
        lines += [
            "",
            "📝 Instagram views to enter (/views &lt;link or clip id&gt; &lt;number&gt;):",
        ]
        lines += [f"• {esc(p.title)} ({esc(p.clip_id)})" for p in missing]
    return "\n".join(lines)


def _median_by(posts: list[PostView], key: Callable[[PostView], str]) -> list[str]:
    groups: dict[str, list[int]] = {}
    for p in posts:
        if p.views_48h is not None:
            groups.setdefault(key(p), []).append(p.views_48h)
    return [f"  {esc(name)}: median {_n(statistics.median(v))} (n={len(v)})"
            for name, v in sorted(groups.items())]  # fmt: skip


def weekly_report(cfg: Config, conn: sqlite3.Connection, now: datetime) -> str:
    posts = post_views(cfg, conn)
    week = [p for p in posts if now - p.posted_at < timedelta(days=7)]
    tz = ZoneInfo(cfg.analytics.timezone)
    lines = [f"<b>📈 Weekly report — week to {now.astimezone(tz):%d %b}</b>",
             f"{len(week)} post(s) this week, {len(posts)} in total."]  # fmt: skip

    def best(p: PostView) -> int:
        return p.views_48h if p.views_48h is not None else (p.views or 0)

    ranked = sorted((p for p in week if p.views is not None), key=lambda p: -best(p))
    if ranked:
        lines += ["", "<b>Top 3</b> (views at 48 h, else latest):"]
        lines += [f"• {esc(p.title)} · {p.label} · {_n(best(p))}" for p in ranked[:3]]
        if len(ranked) > 3:
            lines += ["<b>Bottom 3</b>:"]
            lines += [f"• {esc(p.title)} · {p.label} · {_n(best(p))}" for p in ranked[-3:]]
    groups: tuple[tuple[str, Callable[[PostView], str]], ...] = (
        ("By length", lambda p: length_bucket(p.duration_sec)),
        ("Story vs not", lambda p: "story" if p.story else "not story"),
        ("By layout", lambda p: p.layout),
    )
    for title, key in groups:
        if rows := _median_by(posts, key):
            lines += ["", f"<b>{title}</b> (all posts, views at 48 h):", *rows]
    est = sum(p.earnings for p in posts)
    paid = sum(p.paid_usd or 0 for p in posts)
    reminder = (
        "🌍 Check audience countries this week (YouTube Studio → Audience, Instagram insights)."
    )
    lines += ["", f"Estimated earnings ${est:,.2f} · paid (entered with /paid) ${paid:,.2f}", "",
              reminder]  # fmt: skip
    for c in cfg.campaigns:
        target = (c.rules.model_extra or {}).get("target_audience")
        if c.active and target:
            lines.append(f"  {esc(c.name)}: {esc(str(target))}")
    return "\n".join(lines)


def stats_summary(cfg: Config, conn: sqlite3.Connection, now: datetime, days: int = 7) -> str:
    posts = [p for p in post_views(cfg, conn) if now - p.posted_at < timedelta(days=days)]
    if not posts:
        return f"No posts in the last {days} days."
    total = sum(p.views or 0 for p in posts)
    lines = [f"<b>Last {days} days: {len(posts)} post(s), {_n(total)} views</b>"]
    for p in sorted(posts, key=lambda p: -(p.views or 0)):
        mvr = " 🎯" if p.earnings > 0 else ""
        lines.append(f"• {esc(p.title)} · {p.label} · {_n(p.views)}{mvr} ({esc(p.clip_id)})")
    return "\n".join(lines)


def clip_summary(cfg: Config, conn: sqlite3.Connection, clip_id: str) -> str:
    posts = [p for p in post_views(cfg, conn) if p.clip_id == clip_id]
    if not posts:
        return f"No posts recorded for {esc(clip_id)}."
    first = posts[0]
    lines = [
        f"<b>{esc(first.title)}</b> ({esc(clip_id)})",
        f"{_n(first.duration_sec)} s · {first.layout}{' · story' if first.story else ''}",
    ]
    for p in posts:
        extra = [f"48 h {_n(p.views_48h)}"]
        if p.earnings:
            extra.append(f"est ${p.earnings:,.2f}")
        if p.approved:
            extra.append("approved")
        if p.paid_usd:
            extra.append(f"paid ${p.paid_usd:,.2f}")
        lines.append(f"• {p.label}: {_n(p.views)} views ({', '.join(extra)})")
        lines.append(f"  {esc(p.url)}")
    return "\n".join(lines)


EXPORT_FIELDS = ("post_id", "clip_id", "platform", "posted_at", "title", "hook", "length_sec",
                 "story", "layout", "score", "views_48h", "views", "earnings", "approved",
                 "paid_usd", "url")  # fmt: skip


def export_csv(cfg: Config, conn: sqlite3.Connection, path: Path) -> int:
    """One row per post (hook, title, length, story, layout, score, views at 48 h)."""
    posts = post_views(cfg, conn)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(EXPORT_FIELDS)
        for p in posts:
            w.writerow([p.id, p.clip_id, p.label, p.posted_at.isoformat(), p.title, p.hook,
                        p.duration_sec, int(p.story), p.layout, p.score, p.views_48h, p.views,
                        p.earnings, int(p.approved), p.paid_usd, p.url])  # fmt: skip
    return len(posts)
