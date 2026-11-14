"""ffmpeg command builder. Encoder and output settings come from config."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clipper.config import Render
from clipper.render.facetrack import Crop


def run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")


def probe_video(path: Path) -> tuple[int, int, float]:
    """(width, height, duration_sec) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    info = json.loads(out)
    stream = info["streams"][0]
    return int(stream["width"]), int(stream["height"]), float(info["format"]["duration"])


def encoder_args(encoder: str, bitrate: str) -> list[str]:
    args = ["-c:v", encoder, "-b:v", bitrate, "-pix_fmt", "yuv420p"]
    if encoder == "libx264":
        args += ["-preset", "veryfast", "-maxrate", bitrate, "-bufsize", "20M"]
    elif encoder == "h264_qsv":
        args[-1] = "nv12"
    return args


def render_cmd(
    src: Path,
    out: Path,
    crop: Crop,
    r: Render,
    encoder: str,
    cmd_file: str | None,
    ass_file: str,
) -> list[str]:
    """Crop (moving x via sendcmd) -> scale -> burn captions; normalize loudness.
    `cmd_file` and `ass_file` are bare file names: run ffmpeg with cwd = their directory,
    which avoids filter-path escaping."""
    x0 = crop.xs[0] if crop.xs else 0
    vf = [f"fps={r.fps}"]
    if cmd_file:
        vf.append(f"sendcmd=f={cmd_file}")
    vf += [
        f"crop={crop.w}:{crop.h}:{x0}:{crop.y}",
        f"scale={r.width}:{r.height}:flags=lanczos",
        "setsar=1",
        f"ass={ass_file}",
    ]
    return [
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", ",".join(vf),
        "-af", f"loudnorm=I={r.loudness_lufs:g}:TP=-1.5:LRA=11",
        *encoder_args(encoder, r.video_bitrate),
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out),
    ]  # fmt: skip


def thumbnail_cmd(clip: Path, out: Path, at_sec: float) -> list[str]:
    return ["ffmpeg", "-y", "-v", "error", "-ss", f"{at_sec:.2f}", "-i", str(clip),
            "-frames:v", "1", "-q:v", "3", str(out)]  # fmt: skip
