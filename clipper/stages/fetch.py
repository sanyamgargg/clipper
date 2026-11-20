"""fetch: audio-only download for transcription, then only the picked segments as video.

Contract:
  fetch_audio   -> data/sources/<video_id>.wav (16 kHz mono) + <video_id>.info.json
  fetch_segment -> data/sources/<video_id>_<nn>.mp4 (frame-accurate cut, t=0 is clip start)
Both download to a `.tmp.` name and rename when complete, so a killed download never
leaves a file that looks finished.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from clipper import proc
from clipper.config import Config
from clipper.disk import ensure_free_space

log = logging.getLogger(__name__)

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


def ytdlp(*args: str) -> str:
    """Run yt-dlp, return its stdout. RuntimeError with the tail of stderr on failure."""
    result = proc.run([sys.executable, "-m", "yt_dlp", "--no-progress", *args])
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed ({result.returncode}): {result.stderr.strip()[-800:]}")
    return result.stdout


def probe(url: str, campaign_id: str) -> VideoMeta:
    info = json.loads(ytdlp("--no-playlist", "--dump-json", "--skip-download", url))
    return VideoMeta(
        id=info["id"],
        url=info.get("webpage_url") or url,
        title=info.get("title", ""),
        duration_sec=float(info.get("duration") or 0),
        upload_date=info.get("upload_date") or "",
        campaign_id=campaign_id,
    )


def fetch_audio(cfg: Config, meta: VideoMeta) -> Path:
    """Download audio only, as 16 kHz mono wav (what whisper.cpp wants). Skips if present."""
    sources = cfg.dir("sources")
    wav = sources / f"{meta.id}.wav"
    (sources / f"{meta.id}.info.json").write_text(json.dumps(asdict(meta), indent=2))
    if wav.exists() and wav.stat().st_size > 0:
        log.info("audio already present: %s", wav.name)
        return wav

    # wav plus the compressed download that yt-dlp keeps until conversion finishes
    needed_gb = meta.duration_sec * WAV_BYTES_PER_SEC * 1.5 / 1e9
    ensure_free_space(sources, cfg.machine.min_free_gb, needed_gb)

    log.info("downloading audio: %s (%s, %.0f s)", meta.id, meta.title, meta.duration_sec)
    tmp = sources / f"{meta.id}.tmp.wav"
    ytdlp(
        "--no-playlist",
        "-f", "bestaudio/best",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", "ExtractAudio:-ar 16000 -ac 1",
        "-o", str(sources / f"{meta.id}.tmp.%(ext)s"),
        meta.url,
    )  # fmt: skip
    if not tmp.exists():
        raise RuntimeError(f"yt-dlp finished but {tmp} is missing")
    tmp.replace(wav)
    return wav


SEGMENT_MIN_FRACTION = 0.8  # a segment shorter than this share of the clip is broken


def segment_problem(path: Path, expected_sec: float) -> str | None:
    """Why a downloaded segment is unusable (no video, no audio, cut short), or None.
    yt-dlp's section downloads occasionally produce such files."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=False,
    )  # fmt: skip
    try:
        info = json.loads(proc.stdout or "{}")
    except ValueError:
        info = {}
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    if "video" not in kinds:
        return "no video stream"
    if "audio" not in kinds:
        return "no audio stream"
    duration = float((info.get("format") or {}).get("duration") or 0)
    if duration < expected_sec * SEGMENT_MIN_FRACTION:
        return f"only {duration:.1f} s of {expected_sec:.1f} s"
    return None


def fetch_segment(cfg: Config, meta: VideoMeta, index: int, start: float, end: float) -> Path:
    """Download only [start, end] of the video (max 1080p mp4). A segment already on disk
    is reused only if it checks out; a broken download is retried once."""
    sources = cfg.dir("sources")
    out = sources / f"{meta.id}_{index:02d}.mp4"
    if out.exists():
        problem = segment_problem(out, end - start)
        if problem is None:
            log.info("segment already present: %s", out.name)
            return out
        log.warning("discarding broken segment %s (%s)", out.name, problem)
        out.unlink()
    ensure_free_space(sources, cfg.machine.min_free_gb, (end - start) * SEGMENT_GB_PER_SEC)

    tmp = sources / f"{meta.id}_{index:02d}.tmp.mp4"
    problem = ""
    for attempt in (1, 2):
        log.info("downloading segment %s %.2f-%.2f", out.name, start, end)
        tmp.unlink(missing_ok=True)
        ytdlp(
            "--no-playlist",
            "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b",
            "--merge-output-format", "mp4",
            "--download-sections", f"*{start:.3f}-{end:.3f}",
            "--force-keyframes-at-cuts",  # frame-accurate, so captions stay in sync
            "-o", str(tmp),
            meta.url,
        )  # fmt: skip
        if not tmp.exists():
            raise RuntimeError(f"yt-dlp finished but {tmp} is missing")
        problem = segment_problem(tmp, end - start) or ""
        if not problem:
            tmp.replace(out)
            return out
        log.warning("segment %s is broken (%s), try %d of 2", out.name, problem, attempt)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"segment {out.name} keeps coming out broken: {problem}")
