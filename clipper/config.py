"""Load and validate config.yaml. All machine-specific choices live there."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    candidates_per_video: int = 5
    clip_min_sec: float = 20
    clip_max_sec: float = 45


class Render(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: int = 30
    video_bitrate: str = "9M"
    loudness_lufs: float = -14
    caption_font: str = "Arial"  # must be installed on this machine
    caption_font_size: int = 84
    caption_y_frac: float = 0.66  # caption centre, as a fraction of frame height
    hook_sec: float = 2.5
    face_model: str = "blaze_face_short_range"  # data/models/<name>.tflite


class Claude(BaseModel):
    model: str = "sonnet"
    max_turns: int = 3
    timeout_sec: int = 300


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
    active: bool = True


class Config(BaseModel):
    machine: Machine = Field(default_factory=Machine)
    pipeline: Pipeline = Field(default_factory=Pipeline)
    render: Render = Field(default_factory=Render)
    claude: Claude = Field(default_factory=Claude)
    campaigns: list[Campaign] = Field(default_factory=list)

    @property
    def data_dir(self) -> Path:
        d = self.machine.data_dir
        return d if d.is_absolute() else (REPO_ROOT / d).resolve()

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
