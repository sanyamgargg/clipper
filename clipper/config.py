"""Load and validate config.yaml. All machine-specific choices live there."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parent.parent


class Machine(BaseModel):
    whisper_backend: str = "whisper_cpp"
    whisper_model: str = "large-v3-turbo-q5_0"
    video_encoder: str = "libx264"
    data_dir: Path = Path("./data")
    min_free_gb: float = 5.0  # refuse to download below this much free disk


class Pipeline(BaseModel):
    videos_per_night: int = 2
    max_source_minutes: float = 120  # longer videos are skipped (status skipped_too_long)
    max_retries: int = 3  # a failed video/clip is retried on up to this many later runs
    candidates_per_video: int = 5
    clip_min_sec: float = 18
    clip_max_sec: float = 35
    clip_story_max_sec: float = 45  # only for clips Claude marks "story": true
    clip_target_min_sec: float = 20  # length the prompt asks Claude to prefer
    clip_target_max_sec: float = 30


class Render(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    video_bitrate: str = "6M"  # keeps 45 s clips under Telegram's 50 MB bot upload limit
    loudness_lufs: float = -14
    fonts_dir: Path = Path("assets/fonts")  # bundled fonts, handed to libass
    caption_font: str = "Montserrat Black"  # a font in fonts_dir (else a system font)
    caption_font_size: int = 90  # em height in px
    caption_y_frac: float = 0.66  # caption centre, as a fraction of frame height
    caption_y_frac_fit: float = 0.72  # fit layout: captions sit below the video band
    caption_face_gap: float = 0.04  # caption y >= face bottom + this (fraction of height)
    caption_max_y_frac: float = 0.78  # but never below this
    hook_style: Literal["box", "outline", "titlecard"] = "titlecard"
    hook_font: str = "Montserrat SemiBold"  # titlecard text, a font in fonts_dir
    titlecard_font_size: int = 64  # em height in px; smaller only if a line would not fit
    titlecard_top_frac: float = 0.08  # bar top, as a fraction of frame height
    titlecard_opacity: float = 0.7  # of the black bar
    hook_sec: float = 3.0  # box / outline only; a title card stays for the whole clip
    hook_font_size: int = 150  # largest em height in px; shrinks to fit width / face
    hook_min_font_size: int = 90  # never smaller, even if it then overlaps the head
    hook_fallback_top_frac: float = 0.09  # hook top when no face is found
    hook_top_limit_frac: float = 0.05  # hook never starts higher than this
    default_layout: Literal["framed", "crop", "fit"] = "framed"  # per clip, can be changed
    # framed layout: dark canvas, headline + context line, square video, captions in it
    framed_video_px: int = 1080  # square side; taken at source resolution (<= this)
    framed_background: str = "#0C0C0C"
    framed_headline_px: int = 64  # caption_font (Montserrat Black); shrinks to fit 2 lines
    framed_accent_color: str = "#FFD600"  # Claude's accent_word, and the spoken word
    framed_context_px: int = 36  # hook_font (Montserrat SemiBold)
    framed_context_color: str = "#969696"
    framed_caption_px: int = 64
    framed_caption_outline: int = 6
    framed_caption_above_bottom: int = 200  # caption centre above the square's bottom
    framed_nudge_up: int = 40  # the whole block, from vertically centred
    face_model: str = "blaze_face_short_range"  # data/models/<name>.tflite
    person_model: str = "efficientdet_lite0"  # fallback when no face: object detector


class Claude(BaseModel):
    # Keychain item holding the `claude setup-token` token; run_nightly.sh exports it as
    # CLAUDE_CODE_OAUTH_TOKEN
    keychain_service: str = "clipper-claude-oauth"
    model: str = "sonnet"
    max_turns: int = 3
    timeout_sec: int = 300


class Telegram(BaseModel):
    enabled: bool = False
    chat_id: int | None = None  # set by `python -m clipper.telegram link`
    keychain_service: str = "clipper-telegram"  # bot token; env CLIPPER_TELEGRAM_TOKEN wins
    max_per_run: int = 10
    max_upload_mb: float = 45  # bot uploads max out at 50 MB; bigger clips are re-encoded
    delete_local_after_send: bool = True  # keep the jpg + json, drop the mp4
    delete_after_done_hours: float = 24  # then delete the video message from the chat


class Captions(BaseModel):
    """The tap-to-copy post caption: title, footer, then exactly `hashtag_count` hashtags
    (campaign required_hashtags first, then these, up to 2 topic tags from Claude last)."""

    default_hashtags: list[str] = Field(default_factory=list)
    hashtag_count: int = 5
    caption_footer: str = ""


class CampaignCaptions(BaseModel):
    default_hashtags: list[str] | None = None  # None: use captions.default_hashtags
    caption_footer: str | None = None  # None: use captions.caption_footer


class Analytics(BaseModel):
    """M5 lite (PLAN.md §15): YouTube views via an API key, Instagram views via /views."""

    enabled: bool = True
    youtube_keychain_service: str = "clipper-youtube-key"  # env CLIPPER_YOUTUBE_KEY wins
    timezone: str = "Asia/Kolkata"
    digest_time: str = "09:00"  # daily digest, local time
    weekly_day: int = 0  # weekly report on Monday (0) at digest_time
    stop_after_days: float = 45  # posts older than this are no longer polled
    tick_minutes: float = 10  # how often the bot process checks what is due


class Rules(BaseModel):
    # Campaigns add their own fields (mvr_views, submit_within_minutes, ...): keep them.
    model_config = ConfigDict(extra="allow")

    min_sec: float = 15
    max_sec: float = 60
    required_hashtags: list[str] = Field(default_factory=list)
    required_caption_text: str = ""
    must_feature: str = ""
    no_negative_portrayal: bool = False
    look_for: str = ""
    notes: str = ""


class Campaign(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    rate_per_1k: float = 0.0
    platforms: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    rules: Rules = Field(default_factory=Rules)
    vocabulary: list[str] = Field(default_factory=list)  # whisper prompt: names, jargon
    caption_fixes: dict[str, str] = Field(default_factory=dict)  # misheard -> right, captions
    captions: CampaignCaptions = Field(default_factory=CampaignCaptions)
    active: bool = True

    @property
    def mvr_views(self) -> int:
        """Views a post needs before it earns (campaign `mvr_views`, or rules.mvr_views)."""
        value = (self.model_extra or {}).get("mvr_views")
        value = value if value is not None else (self.rules.model_extra or {}).get("mvr_views")
        return int(value or 0)

    @property
    def max_payout_usd(self) -> float | None:
        value = (self.model_extra or {}).get("max_payout_usd")
        return float(value) if value else None


class Config(BaseModel):
    machine: Machine = Field(default_factory=Machine)
    pipeline: Pipeline = Field(default_factory=Pipeline)
    render: Render = Field(default_factory=Render)
    telegram: Telegram = Field(default_factory=Telegram)
    claude: Claude = Field(default_factory=Claude)
    campaigns: list[Campaign] = Field(default_factory=list)
    captions: Captions = Field(default_factory=Captions)
    analytics: Analytics = Field(default_factory=Analytics)

    @property
    def data_dir(self) -> Path:
        d = self.machine.data_dir
        return d if d.is_absolute() else (REPO_ROOT / d).resolve()

    @property
    def fonts_dir(self) -> Path:
        d = self.render.fonts_dir
        return d if d.is_absolute() else REPO_ROOT / d

    def dir(self, name: str) -> Path:
        p = self.data_dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def campaign(self, campaign_id: str | None = None) -> Campaign:
        """The named campaign, or the first active one."""
        for c in self.campaigns:
            if (campaign_id and c.id == campaign_id) or (not campaign_id and c.active):
                return c
        raise KeyError(f"campaign not found: {campaign_id or '(any active)'}")

    def clip_bounds(self, campaign: Campaign, story: bool = False) -> tuple[float, float]:
        """Clip length limits: pipeline settings narrowed to the campaign's rules. Stories
        (Claude's `"story": true`) may run up to clip_story_max_sec."""
        p = self.pipeline
        lo = max(p.clip_min_sec, campaign.rules.min_sec)
        hi = max(p.clip_max_sec, p.clip_story_max_sec) if story else p.clip_max_sec
        return lo, min(hi, campaign.rules.max_sec)


def load_config(path: Path | None = None) -> Config:
    path = path or REPO_ROOT / "config.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return Config.model_validate(raw)
