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
    clip_min_sec: float = 20
    clip_max_sec: float = 45


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
    hook_style: Literal["box", "outline"] = "outline"
    hook_sec: float = 3.0
    hook_font_size: int = 150  # largest em height in px; shrinks to fit width / face
    hook_min_font_size: int = 90  # never smaller, even if it then overlaps the head
    hook_fallback_top_frac: float = 0.09  # hook top when no face is found
    hook_top_limit_frac: float = 0.05  # hook never starts higher than this
    default_layout: Literal["crop", "fit"] = "crop"  # per clip, can be changed later
    face_model: str = "blaze_face_short_range"  # data/models/<name>.tflite
    person_model: str = "efficientdet_lite0"  # fallback when no face: object detector


class Claude(BaseModel):
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
    active: bool = True


class Config(BaseModel):
    machine: Machine = Field(default_factory=Machine)
    pipeline: Pipeline = Field(default_factory=Pipeline)
    render: Render = Field(default_factory=Render)
    telegram: Telegram = Field(default_factory=Telegram)
    claude: Claude = Field(default_factory=Claude)
    campaigns: list[Campaign] = Field(default_factory=list)

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

    def clip_bounds(self, campaign: Campaign) -> tuple[float, float]:
        """Clip length limits: pipeline defaults narrowed to the campaign's rules."""
        lo = max(self.pipeline.clip_min_sec, campaign.rules.min_sec)
        hi = min(self.pipeline.clip_max_sec, campaign.rules.max_sec)
        return lo, hi


def load_config(path: Path | None = None) -> Config:
    path = path or REPO_ROOT / "config.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return Config.model_validate(raw)
