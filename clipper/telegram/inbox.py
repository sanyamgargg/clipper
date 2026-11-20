"""Telegram clip inbox (PLAN.md §14): deliver rendered clips to one private chat and
handle the Done / Reject / layout buttons, post-link replies and commands.

Clip statuses: rendered -> delivered -> posted | rejected; a layout button sets
`rerender` and the next pipeline run renders and delivers the clip again.
"""

from __future__ import annotations

import html
import logging
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from clipper import db, post_text, proc
from clipper.analytics import core as analytics
from clipper.analytics import reports
from clipper.config import Campaign, Config
from clipper.render.ffmpeg import encoder_args, probe_video
from clipper.stages.pick import Clip, load_clips, save_clips
from clipper.status import report
from clipper.telegram.api import BotAPI, TelegramError, get_token

log = logging.getLogger(__name__)

CAPTION_LIMIT = 1024  # Telegram media captions
MESSAGE_LIMIT = 4096
HARD_UPLOAD_LIMIT = 50e6  # Bot API upload limit
AUDIO_KBPS = 160
OFFSET_KEY = "telegram_offset"
BOT_LOCK = "telegram-bot.lock"  # held by `python -m clipper.telegram bot`
URL_RE = re.compile(r"https?://[^\s<>\"']+")
PLATFORMS = (
    ("youtube_shorts", re.compile(r"(^|\.)(youtube\.com|youtu\.be)$")),
    ("instagram_reels", re.compile(r"(^|\.)instagram\.com$")),
    ("tiktok", re.compile(r"(^|\.)tiktok\.com$")),
)
LABELS = {
    "youtube_shorts": "YouTube",
    "instagram_reels": "Instagram",
    "tiktok": "TikTok",
    "other": "other",
}
NO_BUTTONS: dict[str, Any] = {"inline_keyboard": []}
HELP = (
    "Clips arrive here when they are rendered.\n"
    "• <b>✅ Done</b> after posting: then reply with the post links.\n"
    "• <b>❌ Reject</b> deletes the clip.\n"
    "• <b>🔁 Framed / Crop / Fit</b> re-renders it in that layout on the next run.\n"
    "• Reply to a clip with its post links at any time.\n"
    "/pending: clips waiting  ·  /status: pipeline status and Claude cost\n"
    "/stats: last 7 days  ·  /clip &lt;id&gt;: one clip's numbers\n"
    "/views &lt;link or clip id&gt; &lt;n&gt;: Instagram views from the app\n"
    "/approved &lt;link&gt;  ·  /paid &lt;link&gt; &lt;usd&gt;: Content Rewards results"
)


class Api(Protocol):
    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        files: dict[str, tuple[str, Path, str]] | None = None,
        timeout: float = 60,
        retries: int = 3,
    ) -> Any: ...


@dataclass
class Inbox:
    cfg: Config
    conn: sqlite3.Connection
    api: Api
    chat_id: int

    def clip(self, clip_id: str) -> tuple[Clip, Campaign]:
        video_id = clip_id.rsplit("_", 1)[0]
        clips = load_clips(self.cfg.dir("transcripts") / f"{video_id}.clips.json")
        clip = next((c for c in clips if c.id == clip_id), None)
        video = db.get_video(self.conn, video_id)
        if clip is None or video is None:
            raise KeyError(f"unknown clip {clip_id}")
        return clip, self.cfg.campaign(video["campaign_id"])

    def say(self, text: str, **extra: Any) -> Any:
        return self.api.call(
            "sendMessage",
            {"chat_id": self.chat_id, "text": text[:MESSAGE_LIMIT], "parse_mode": "HTML",
             "link_preview_options": {"is_disabled": True}, **extra},
        )  # fmt: skip


def open_inbox(cfg: Config, conn: sqlite3.Connection) -> Inbox | None:
    """The inbox if Telegram is enabled, linked and has a token; else None (logged)."""
    t = cfg.telegram
    if not t.enabled:
        return None
    if t.chat_id is None:
        log.warning("telegram is enabled but not linked: run python -m clipper.telegram link")
        return None
    try:
        token = get_token(t.keychain_service)
    except TelegramError as exc:
        log.warning("%s", exc)
        return None
    return Inbox(cfg, conn, BotAPI(token), t.chat_id)


# --- formatting -----------------------------------------------------------------


def find_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?)]}") for u in URL_RE.findall(text)]


def platform_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return next((name for name, rx in PLATFORMS if rx.search(host)), "other")


def submit_minutes(campaign: Campaign) -> int:
    return int((campaign.rules.model_extra or {}).get("submit_within_minutes", 30))


def submit_url(campaign: Campaign) -> str:
    extra = campaign.model_extra or {}
    return str(extra.get("submit_url") or extra.get("campaign_url") or "")


def clip_caption(cfg: Config, clip: Clip, campaign: Campaign, note: str = "") -> str:
    """Video caption (HTML): title, campaign, score and length; the post caption with its
    footer and hashtags in one <code> block (one tap copies it all); the YouTube title in
    another; where to submit; and an optional status note."""
    esc = html.escape

    def build(head: str, title: str) -> str:
        lines = [
            f"<b>{esc(head)}</b>",
            f"{esc(campaign.name)} · score {clip.score:.2f} · {clip.end_sec - clip.start_sec:.0f} s",
            "",
            "Caption + hashtags (tap to copy):",
            f"<code>{esc(post_text.post_caption(cfg, campaign, title, clip.hashtags))}</code>",
            "",
            "YouTube title (tap to copy):",
            f"<code>{esc(post_text.youtube_title(title))}</code>",
        ]
        url = submit_url(campaign)
        if url:
            lines += ["", f"Submit within {submit_minutes(campaign)} min of posting:", esc(url)]
        if note:
            lines += ["", note]
        return "\n".join(lines)

    def cut(title: str, over: int) -> str:
        return title[: max(0, len(title) - over - 1)].rstrip() + "…"

    text = build(clip.title, clip.title)
    if len(text) > CAPTION_LIMIT:  # shorten the bold heading first, then the copied title
        text = build(cut(clip.title, len(text) - CAPTION_LIMIT), clip.title)
    title = clip.title
    while len(text) > CAPTION_LIMIT and title:  # hashtags and the note always stay whole
        title = cut(title, len(text) - CAPTION_LIMIT).removesuffix("…")
        text = build("…", title + "…")
    return text[:CAPTION_LIMIT]


def refresh_captions(ix: Inbox) -> tuple[int, int]:
    """Edit the caption of every clip message still waiting (delivered), e.g. after the
    caption format changed. Returns (edited, failed)."""
    edited = failed = 0
    for row in db.clips_in(ix.conn, "delivered"):
        if not row["telegram_message_id"]:
            continue
        try:
            clip, campaign = ix.clip(row["id"])
            ix.api.call(
                "editMessageCaption",
                {"chat_id": ix.chat_id, "message_id": row["telegram_message_id"],
                 "caption": clip_caption(ix.cfg, clip, campaign), "parse_mode": "HTML",
                 "reply_markup": buttons(clip.id, clip.layout)},
            )  # fmt: skip
            edited += 1
        except (TelegramError, KeyError) as exc:
            if isinstance(exc, TelegramError) and "not modified" in str(exc):
                continue
            log.warning("could not refresh %s: %s", row["id"], exc)
            failed += 1
    return edited, failed


LAYOUTS = ("framed", "crop", "fit")


def buttons(clip_id: str, layout: str) -> dict[str, Any]:
    """Done / Reject, and a re-render button for each other layout."""
    others = [name for name in LAYOUTS if name != layout]
    return {
        "inline_keyboard": [
            [
                {"text": "✅ Done", "callback_data": f"done:{clip_id}"},
                {"text": "❌ Reject", "callback_data": f"reject:{clip_id}"},
            ],
            [
                {"text": f"🔁 {name.title()}", "callback_data": f"{name}:{clip_id}"}
                for name in others
            ],
        ]
    }


# --- delivery ---------------------------------------------------------------------


def upload_bitrate_kbps(size_bytes: int, duration: float, limit_bytes: float) -> int | None:
    """Video bitrate (kbps) that brings a clip under `limit_bytes`, or None if it fits."""
    if size_bytes <= limit_bytes or duration <= 0:
        return None
    total_kbps = limit_bytes * 8 * 0.95 / duration / 1000  # 5% container headroom
    return max(500, int(total_kbps - AUDIO_KBPS))


def prepare_upload(cfg: Config, video: Path, duration: float, work: Path) -> Path:
    """The file to upload: `video` itself, or a smaller re-encode if it is too big."""
    kbps = upload_bitrate_kbps(video.stat().st_size, duration, cfg.telegram.max_upload_mb * 1e6)
    if kbps is None:
        return video
    out = work / video.name
    log.info("%s is %.1f MB: re-encoding at %d kbps for Telegram",
             video.name, video.stat().st_size / 1e6, kbps)  # fmt: skip
    result = proc.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         *encoder_args(cfg.machine.video_encoder, f"{kbps}k"),
         "-c:a", "copy", "-movflags", "+faststart", str(out)]
    )  # fmt: skip
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"re-encode failed: {result.stderr.strip()[-400:]}")
    if out.stat().st_size > HARD_UPLOAD_LIMIT:
        raise RuntimeError(f"{video.name} is still {out.stat().st_size / 1e6:.1f} MB")
    return out


def make_thumbnail(jpg: Path, work: Path) -> Path | None:
    """Telegram thumbnails must be JPEG, at most 320 px and 200 kB."""
    if not jpg.exists():
        return None
    out = work / "thumb.jpg"
    result = proc.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(jpg), "-vf",
         "scale=320:320:force_original_aspect_ratio=decrease", "-q:v", "4", str(out)]
    )  # fmt: skip
    if result.returncode != 0 or not out.exists() or out.stat().st_size > 200_000:
        return None
    return out


def _delete_message(ix: Inbox, message_id: int) -> None:
    try:
        ix.api.call("deleteMessage", {"chat_id": ix.chat_id, "message_id": message_id}, retries=1)
    except TelegramError as exc:  # e.g. older than 48 h: bots may not delete those
        log.warning("could not delete message %s: %s", message_id, exc)


def send_clip(ix: Inbox, row: sqlite3.Row) -> int:
    """Send one rendered clip; returns the message id."""
    clip, campaign = ix.clip(row["id"])
    clips_dir = ix.cfg.dir("clips")
    video = clips_dir / f"{clip.id}.mp4"
    if not video.exists():
        raise RuntimeError(f"{video.name} is missing")
    width, height, duration = probe_video(video)
    work = ix.cfg.dir("work") / f"telegram-{clip.id}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        files = {"video": (video.name, prepare_upload(ix.cfg, video, duration, work), "video/mp4")}
        thumb = make_thumbnail(clips_dir / f"{clip.id}.jpg", work)
        if thumb:
            files["thumbnail"] = ("thumb.jpg", thumb, "image/jpeg")
        result = ix.api.call(
            "sendVideo",
            {
                "chat_id": ix.chat_id,
                "caption": clip_caption(ix.cfg, clip, campaign),
                "parse_mode": "HTML",
                "supports_streaming": True,
                "width": width,
                "height": height,
                "duration": round(duration),
                "reply_markup": buttons(clip.id, clip.layout),
            },
            files=files,
            timeout=600,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    message_id = int(result["message_id"])
    old = row["telegram_message_id"]
    db.update_clip(ix.conn, clip.id, status="delivered", telegram_message_id=message_id,
                   delivered_at=db.now(), links_prompt_id=None, done_at=None,
                   message_deleted=0)  # fmt: skip
    if old and old != message_id:
        _delete_message(ix, old)  # a re-render replaces the previous message
    if ix.cfg.telegram.delete_local_after_send:
        video.unlink(missing_ok=True)
    log.info("delivered %s to Telegram (message %d)", clip.id, message_id)
    return message_id


def deliver(ix: Inbox, limit: int) -> tuple[int, int]:
    """Send up to `limit` rendered clips, oldest first. A clip that fails stays
    `rendered` for the next run. Returns (sent, failed)."""
    sent = failed = 0
    for row in db.clips_in(ix.conn, "rendered", limit):
        try:
            send_clip(ix, row)
            sent += 1
        except Exception as exc:  # noqa: BLE001 - one bad clip must not block the others
            log.error("could not deliver %s: %s", row["id"], exc)
            failed += 1
    return sent, failed


# --- updates ----------------------------------------------------------------------


def poll(ix: Inbox, timeout: int = 0) -> int:
    """Fetch and handle pending updates (long poll up to `timeout` s), then tidy up.
    Returns the number of updates handled."""
    offset = db.get_setting(ix.conn, OFFSET_KEY)
    updates = ix.api.call(
        "getUpdates",
        {"offset": int(offset) if offset else None, "timeout": timeout,
         "allowed_updates": ["message", "callback_query"]},
        timeout=timeout + 15, retries=1,
    )  # fmt: skip
    for update in updates:
        try:
            handle_update(ix, update)
        except Exception:
            log.exception("could not handle update %s", update.get("update_id"))
        db.set_setting(ix.conn, OFFSET_KEY, str(update["update_id"] + 1))
    housekeeping(ix)
    return len(updates)


def handle_update(ix: Inbox, update: dict[str, Any]) -> None:
    if cq := update.get("callback_query"):
        message = cq.get("message") or {}
        if (message.get("chat") or {}).get("id") != ix.chat_id:
            log.warning("ignoring a button press from another chat")
            return
        answer = on_button(ix, cq.get("data", ""))
        ix.api.call("answerCallbackQuery", {"callback_query_id": cq["id"], "text": answer[:200]})
        return
    message = update.get("message")
    if not message:
        return
    if (message.get("chat") or {}).get("id") != ix.chat_id:
        log.warning("ignoring a message from another chat")
        return
    text = message.get("text") or message.get("caption") or ""
    if text.startswith("/"):
        command, *args = text.split()
        on_command(ix, command.split("@")[0].lower(), args)
    elif urls := find_urls(text):
        on_links(ix, message, urls)
    else:
        ix.say("Reply to a clip with its post links, or see /help.")


def on_button(ix: Inbox, data: str) -> str:
    action, _, clip_id = data.partition(":")
    row = db.get_clip(ix.conn, clip_id)
    if row is None:
        return "Unknown clip"
    if action == "done":
        return on_done(ix, row)
    if action == "reject":
        return on_reject(ix, row)
    if action in LAYOUTS:
        return on_layout(ix, row, action)
    return "Unknown button"


def _local_time(minutes: int = 0) -> str:
    return (datetime.now().astimezone() + timedelta(minutes=minutes)).strftime("%H:%M")


def on_done(ix: Inbox, row: sqlite3.Row) -> str:
    if row["status"] != "delivered":
        return f"Already {row['status']}"
    clip, campaign = ix.clip(row["id"])
    note = f"✅ Posted {_local_time()} — submit by {_local_time(submit_minutes(campaign))}"
    db.update_clip(ix.conn, clip.id, status="posted", done_at=db.now())
    ix.api.call(
        "editMessageCaption",
        {"chat_id": ix.chat_id, "message_id": row["telegram_message_id"],
         "caption": clip_caption(ix.cfg, clip, campaign, note), "parse_mode": "HTML",
         "reply_markup": NO_BUTTONS},
    )  # fmt: skip
    if not db.posts_for(ix.conn, clip.id):
        prompt = ix.say(
            f"Paste the post links (YouTube / Instagram) for <b>{html.escape(clip.title)}</b> "
            "as a reply to this message.",
            reply_markup={"force_reply": True, "input_field_placeholder": "Post links"},
        )
        db.update_clip(ix.conn, clip.id, links_prompt_id=int(prompt["message_id"]))
    return "Marked as posted"


def _delete_local(ix: Inbox, clip_id: str) -> None:
    for suffix in (".mp4", ".jpg", ".json"):
        (ix.cfg.dir("clips") / f"{clip_id}{suffix}").unlink(missing_ok=True)


def on_reject(ix: Inbox, row: sqlite3.Row) -> str:
    if row["status"] in ("posted", "rejected"):
        return f"Already {row['status']}"
    db.update_clip(ix.conn, row["id"], status="rejected", done_at=db.now(), message_deleted=1)
    if row["telegram_message_id"]:
        _delete_message(ix, row["telegram_message_id"])
    _delete_local(ix, row["id"])
    return "Rejected and deleted"


def on_layout(ix: Inbox, row: sqlite3.Row, layout: str) -> str:
    if row["status"] != "delivered":
        return f"Cannot change a clip that is {row['status']}"
    clip, campaign = ix.clip(row["id"])
    path = ix.cfg.dir("transcripts") / f"{clip.id.rsplit('_', 1)[0]}.clips.json"
    clips = load_clips(path)
    for c in clips:
        if c.id == clip.id:
            c.layout = layout
    save_clips(clips, path)
    db.update_clip(ix.conn, clip.id, status="rerender", layout=layout)
    clip.layout = layout
    note = f"🔁 Re-rendering with {layout} layout on the next run; this message will be replaced."
    ix.api.call(
        "editMessageCaption",
        {"chat_id": ix.chat_id, "message_id": row["telegram_message_id"],
         "caption": clip_caption(ix.cfg, clip, campaign, note),
         "parse_mode": "HTML", "reply_markup": NO_BUTTONS},
    )  # fmt: skip
    return f"Will re-render with {layout} layout"


def on_links(ix: Inbox, message: dict[str, Any], urls: list[str]) -> None:
    reply_to = message.get("reply_to_message")
    target = db.clip_for_message(ix.conn, reply_to["message_id"]) if reply_to else None
    if target is None and not reply_to:
        # not a reply: the most recent "paste the links" prompt still waiting
        target = ix.conn.execute(
            "SELECT * FROM clips c WHERE links_prompt_id IS NOT NULL AND NOT EXISTS"
            " (SELECT 1 FROM posts p WHERE p.clip_id = c.id) ORDER BY done_at DESC LIMIT 1"
        ).fetchone()
    if target is None:
        ix.say("Which clip is this for? Reply to the clip's message with its post links.")
        return
    added = [platform_of(u) for u in urls if db.add_post(ix.conn, target["id"], platform_of(u), u)]
    title = html.escape(target["title"] or target["id"])
    if added:
        kinds = ", ".join(LABELS[p] for p in added)
        ix.say(f"Saved {len(added)} link(s) for <b>{title}</b>: {kinds}")
    else:
        ix.say(f"Those links are already saved for <b>{title}</b>.")


def on_stats_command(ix: Inbox, command: str, args: list[str]) -> bool:
    """/stats, /clip, /views, /approved, /paid (PLAN.md §15). False if not one of them."""
    now = datetime.now(UTC)
    if command == "/stats":
        ix.say(reports.stats_summary(ix.cfg, ix.conn, now))
        return True
    if command == "/clip":
        ix.say(
            reports.clip_summary(ix.cfg, ix.conn, args[0])
            if args
            else "Usage: /clip &lt;clip id&gt;"
        )
        return True
    if command not in ("/views", "/approved", "/paid"):
        return False
    need = 1 if command == "/approved" else 2
    usage = {"/views": "/views &lt;post link or clip id&gt; &lt;views, e.g. 12345 or 12.3k&gt;",
             "/approved": "/approved &lt;post link&gt;",
             "/paid": "/paid &lt;post link&gt; &lt;usd&gt;"}[command]  # fmt: skip
    if len(args) < need:
        ix.say(f"Usage: {usage}")
        return True
    post = analytics.find_post(ix.conn, args[0])
    if post is None:
        ix.say(f"No recorded post matches {html.escape(args[0])}. Reply to the clip with its link "
               "first.")  # fmt: skip
        return True
    title = html.escape((db.get_clip(ix.conn, post["clip_id"]) or {"title": None})["title"]
                        or post["clip_id"])  # fmt: skip
    label = LABELS.get(post["platform"], post["platform"])
    if command == "/approved":
        ix.conn.execute("UPDATE posts SET approved = 1 WHERE id = ?", (post["id"],))
        ix.say(f"✅ Approved: <b>{title}</b> ({label})")
        return True
    if command == "/paid":
        try:
            usd = float(args[1].lstrip("$").replace(",", ""))
        except ValueError:
            ix.say(f"Usage: {usage}")
            return True
        ix.conn.execute("UPDATE posts SET paid_usd = ? WHERE id = ?", (usd, post["id"]))
        ix.say(f"💵 Paid ${usd:,.2f}: <b>{title}</b> ({label})")
        return True
    views = analytics.parse_count(args[1])
    if views is None:
        ix.say(f"Usage: {usage}")
        return True
    alert = analytics.record_views(ix.cfg, ix.conn, post, views, now, "manual")
    ix.say(f"Saved {views:,} views for <b>{title}</b> ({label}).")
    if alert:
        ix.say(reports.mvr_alert_text(alert))
    return True


def on_command(ix: Inbox, command: str, args: list[str] | None = None) -> None:
    if on_stats_command(ix, command, args or []):
        return
    if command == "/start":
        ix.say("✅ This chat is linked. New clips will arrive here.\n\n" + HELP)
    elif command == "/help":
        ix.say(HELP)
    elif command == "/status":
        ix.say(f"<pre>{html.escape(report(ix.conn))[: MESSAGE_LIMIT - 20]}</pre>")
    elif command == "/pending":
        rows = db.clips_in(ix.conn, "delivered")
        if not rows:
            ix.say("Nothing waiting. 🎉")
            return
        lines = [f"<b>{len(rows)} clip(s) waiting:</b>"]
        for row in rows:
            lines.append(f"• {html.escape(row['title'] or row['id'])} ({row['id']})")
        ix.say("\n".join(lines))
    else:
        ix.say("Unknown command. /help")


def housekeeping(ix: Inbox) -> int:
    """Delete video messages of clips posted more than delete_after_done_hours ago."""
    cutoff = datetime.now(UTC) - timedelta(hours=ix.cfg.telegram.delete_after_done_hours)
    rows = ix.conn.execute(
        "SELECT * FROM clips WHERE status = 'posted' AND message_deleted = 0"
        " AND telegram_message_id IS NOT NULL AND done_at <= ?",
        (cutoff.isoformat(timespec="seconds"),),
    ).fetchall()
    for row in rows:
        _delete_message(ix, row["telegram_message_id"])
        db.update_clip(ix.conn, row["id"], message_deleted=1)
    return len(rows)


# --- linking --------------------------------------------------------------------


def find_start(updates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The private chat that most recently sent /start."""
    for update in reversed(updates):
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        if chat.get("type") == "private" and (message.get("text") or "").startswith("/start"):
            return chat
    return None


def write_chat_id(path: Path, chat_id: int) -> None:
    """Set telegram.chat_id (and enabled: true) in config.yaml, keeping its comments."""
    text = path.read_text()
    block = re.search(r"(?m)^telegram:[^\n]*\n((?:[ \t]+[^\n]*\n|[ \t]*\n)*)", text)
    if block is None:
        text = text.rstrip("\n") + f"\n\ntelegram:\n  enabled: true\n  chat_id: {chat_id}\n"
    else:
        body = block.group(1)

        def set_key(body: str, key: str, value: str) -> str:
            pattern = rf"(?m)^([ \t]+{key}:)[ \t]*[^#\n]*?([ \t]*#[^\n]*)?$"
            if re.search(pattern, body):
                return re.sub(pattern, lambda m: f"{m.group(1)} {value}{m.group(2) or ''}",
                              body, count=1)  # fmt: skip
            return f"  {key}: {value}\n" + body

        body = set_key(set_key(body, "chat_id", str(chat_id)), "enabled", "true")
        text = text[: block.start(1)] + body + text[block.end(1) :]
    path.write_text(text)
