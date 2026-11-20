"""The text that goes with a posted clip: caption + footer + exactly N hashtags, and the
YouTube title. Used by the Telegram inbox: one tap copies the whole post text.

    <clip title>

    Follow for daily TJR clips 📈

    #TJR #daytrading #trading #money #mindset

Hashtags: campaign required_hashtags first (they count toward the N), then the defaults
(captions.default_hashtags, or the campaign's), with up to 2 topic hashtags from Claude
replacing the last defaults.
"""

from __future__ import annotations

import re

from clipper.config import Campaign, Config

TAG_RE = re.compile(r"^#[a-z0-9_]{2,30}$")
MAX_TOPIC_TAGS = 2
YT_TITLE_MAX = 100
# tags Instagram limits/hides or that read as spam: never used
BANNED_TAGS = frozenset({
    "#followforfollow", "#follow4follow", "#f4f", "#like4like", "#l4l", "#likeforlike",
    "#followback", "#followme", "#tagsforlikes", "#instalike", "#likes", "#follow",
    "#fyp", "#foryou", "#foryoupage", "#viral", "#explore", "#explorepage", "#trending",
    "#reels", "#reelsinstagram", "#shorts", "#youtubeshorts", "#instagood", "#tiktok",
    "#getrichquick", "#easymoney", "#freemoney", "#guaranteed", "#signals",
})  # fmt: skip


def _norm(tag: str) -> str:
    tag = tag.strip()
    return tag if tag.startswith("#") else f"#{tag}"


def clean_topic_tags(tags: list[str] | None) -> list[str]:
    """Claude's topic hashtags: lowercase, no spaces, not banned, at most 2."""
    out: list[str] = []
    for raw in tags or []:
        tag = _norm(str(raw)).lower()
        if TAG_RE.match(tag) and tag not in BANNED_TAGS and tag not in out:
            out.append(tag)
    return out[:MAX_TOPIC_TAGS]


def hashtags(cfg: Config, campaign: Campaign, topic: list[str] | None = None) -> list[str]:
    """Exactly captions.hashtag_count tags (fewer only if config lists too few): required
    first, then the defaults, with the last defaults replaced by topic tags."""
    n = cfg.captions.hashtag_count
    extra = campaign.captions.default_hashtags
    defaults = extra if extra is not None else cfg.captions.default_hashtags
    tags: list[str] = []
    seen: set[str] = set()

    def add(tag: str) -> None:
        if len(tags) < n and tag.lower() not in seen:
            seen.add(tag.lower())
            tags.append(tag)

    for t in campaign.rules.required_hashtags:
        add(_norm(t))
    topics = [t for t in clean_topic_tags(topic) if t.lower() not in seen]
    defaults_n = [_norm(t) for t in defaults]
    keep = max(0, n - len(tags) - len(topics))
    for t in defaults_n:
        if keep <= 0:
            break
        before = len(tags)
        add(t)
        keep -= len(tags) - before
    for t in topics:
        add(t)
    for t in defaults_n:  # topic tags that duplicated a default leave room: top up
        add(t)
    return tags


def footer(cfg: Config, campaign: Campaign) -> str:
    value = campaign.captions.caption_footer
    return (cfg.captions.caption_footer if value is None else value).strip()


def post_caption(cfg: Config, campaign: Campaign, title: str,
                 topic: list[str] | None = None) -> str:  # fmt: skip
    """Title, footer, hashtags, separated by blank lines (the tap-to-copy text)."""
    head = title.strip()
    if campaign.rules.required_caption_text.strip():
        head += " " + campaign.rules.required_caption_text.strip()
    parts = [head]
    if foot := footer(cfg, campaign):
        parts.append(foot)
    if tags := hashtags(cfg, campaign, topic):
        parts.append(" ".join(tags))
    return "\n\n".join(parts)


def youtube_title(title: str) -> str:
    """Clip title + " #shorts", at most 100 characters."""
    suffix = " #shorts"
    return title.strip()[: YT_TITLE_MAX - len(suffix)].rstrip() + suffix
