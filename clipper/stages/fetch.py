"""fetch: audio-only download for transcription, then only the picked segments as video.

Contract:
  fetch_audio   -> data/sources/<video_id>.wav (16 kHz mono) + <video_id>.info.json
  fetch_segment -> data/sources/<video_id>_<nn>.mp4 (frame-accurate cut, t=0 is clip start)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from clipper.config import Config
from clipper.disk import ensure_free_space

log = logging.getLogger(__name__)

YT_DLP = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress"]
WAV_BYTES_PER_SEC = 32_000  # 16 kHz mono s16
SEGMENT_GB_PER_SEC = 1.5e-3  # ~12 Mbps worst case for a 1080p cut


@dataclass
class VideoMeta:
    id: str
    url: str
    title: str
    duration_sec: float
    upload_date: str  # YYYYMMDD
    campaign_id: str

    @classmethod
    def load(cls, path: Path) -> VideoMeta:
        return cls(**json.loads(path.read_text()))


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[2]} failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
    return proc.stdout


def probe(url: str, campaign_id: str) -> VideoMeta:
    info = json.loads(_run([*YT_DLP, "--dump-json", "--skip-download", url]))
    return VideoMeta(
        id=info["id"],
        url=info.get("webpage_url") or url,
        title=info.get("title", ""),
        duration_sec=float(info.get("duration") or 0),
        upload_date=info.get("upload_date", ""),
        campaign_id=campaign_id,
    )


def fetch_audio(cfg: Config, meta: VideoMeta) -> Path:
    """Download audio only, as 16 kHz mono wav (what whisper.cpp wants). Skips if present."""
    sources = cfg.dir("sources")
    url = meta.url
    wav = sources / f"{meta.id}.wav"
    info_path = sources / f"{meta.id}.info.json"
    info_path.write_text(json.dumps(asdict(meta), indent=2))
    if wav.exists() and wav.stat().st_size > 0:
        log.info("audio already present: %s", wav.name)
        return wav

    # wav plus the compressed download that yt-dlp keeps until conversion finishes
    needed_gb = meta.duration_sec * WAV_BYTES_PER_SEC * 1.5 / 1e9
    ensure_free_space(sources, cfg.machine.min_free_gb, needed_gb)

    log.info("downloading audio: %s (%s, %.0f s)", meta.id, meta.title, meta.duration_sec)
    _run(
        [
            *YT_DLP,
            "-f", "bestaudio/best",
            "-x", "--audio-format", "wav",
            "--postprocessor-args", "ExtractAudio:-ar 16000 -ac 1",
            "-o", str(sources / "%(id)s.%(ext)s"),
            url,
        ]
    )  # fmt: skip
    if not wav.exists():
        raise RuntimeError(f"yt-dlp finished but {wav} is missing")
    return wav


def fetch_segment(cfg: Config, meta: VideoMeta, index: int, start: float, end: float) -> Path:
    """Download only [start, end] of the video (max 1080p mp4). Skips if present."""
    sources = cfg.dir("sources")
    out = sources / f"{meta.id}_{index:02d}.mp4"
    if out.exists() and out.stat().st_size > 0:
        log.info("segment already present: %s", out.name)
        return out
    ensure_free_space(sources, cfg.machine.min_free_gb, (end - start) * SEGMENT_GB_PER_SEC)

    log.info("downloading segment %s %.2f-%.2f", out.name, start, end)
    _run(
        [
            *YT_DLP,
            "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
            "--merge-output-format", "mp4",
            "--download-sections", f"*{start:.3f}-{end:.3f}",
            "--force-keyframes-at-cuts",  # frame-accurate, so captions stay in sync
            "-o", str(out),
            meta.url,
        ]
    )  # fmt: skip
    if not out.exists():
        raise RuntimeError(f"yt-dlp finished but {out} is missing")
    return out
