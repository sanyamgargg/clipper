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


def video_filter(r: Render, crop: Crop | None, cmd_file: str | None, ass_file: str) -> str:
    """filter_complex graph from [0:v] to [v].

    crop layout (`crop` given): crop window, moved by sendcmd, scaled to the output size.
    fit layout (`crop` is None): the whole frame scaled to fit, centred over a blurred,
    slightly darkened copy of itself that fills the output."""
    if crop is None:
        bw, bh = r.width // 4, r.height // 4  # blur at quarter size: cheap, and softer
        chain = (
            f"[0:v]fps={r.fps},split[a][b];"
            f"[a]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
            f"boxblur=8:2,scale={r.width}:{r.height},eq=brightness=-0.08[bg];"
            f"[b]scale={r.width}:{r.height}:force_original_aspect_ratio=decrease:"
            f"force_divisible_by=2:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
    else:
        send = f"sendcmd=f={cmd_file}," if cmd_file else ""
        x0 = crop.xs[0] if crop.xs else 0
        chain = (
            f"[0:v]fps={r.fps},{send}crop={crop.w}:{crop.h}:{x0}:{crop.y},"
            f"scale={r.width}:{r.height}:flags=lanczos"
        )
    return f"{chain},setsar=1,ass={ass_file}[v]"


def render_cmd(
    src: Path,
    out: Path,
    crop: Crop | None,
    r: Render,
    encoder: str,
    cmd_file: str | None,
    ass_file: str,
) -> list[str]:
    """Lay out the video (see video_filter), burn captions, normalize loudness.
    `cmd_file` and `ass_file` are bare file names: run ffmpeg with cwd = their directory,
    which avoids filter-path escaping."""
    return [
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-filter_complex", video_filter(r, crop, cmd_file, ass_file),
        "-map", "[v]", "-map", "0:a:0",
        "-af", f"loudnorm=I={r.loudness_lufs:g}:TP=-1.5:LRA=11",
        *encoder_args(encoder, r.video_bitrate),
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-movflags", "+faststart",
        str(out),
    ]  # fmt: skip


def thumbnail_cmd(clip: Path, out: Path, at_sec: float) -> list[str]:
    return ["ffmpeg", "-y", "-v", "error", "-ss", f"{at_sec:.2f}", "-i", str(clip),
            "-frames:v", "1", "-q:v", "3", str(out)]  # fmt: skip
