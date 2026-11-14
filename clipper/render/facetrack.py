"""Face tracking -> smoothed crop-x path for a vertical crop.

detect_faces() is the only heavy part (ffmpeg pipe + MediaPipe); everything else is pure.
"""

from __future__ import annotations

import logging
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SAMPLE_FPS = 5
DETECT_W = 480  # frames are scaled down to this width for detection
MIN_SCORE = 0.6
SAME_FACE = 0.12  # faces within this x distance (fraction of width) are the same person
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/{name}/float16/latest/"
    "{name}.tflite"
)


@dataclass(frozen=True)
class Face:
    cx: float  # centre x, 0..1 of frame width
    size: float  # box width, 0..1 of frame width
    score: float


@dataclass(frozen=True)
class Crop:
    w: int
    h: int
    y: int
    xs: list[int]  # crop x per output frame


def ensure_model(models_dir: Path, name: str) -> Path:
    path = models_dir / f"{name}.tflite"
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading face model %s", name)
        urllib.request.urlretrieve(MODEL_URL.format(name=name), path)
    return path


def detect_faces(video: Path, model: Path, src_w: int, src_h: int) -> list[list[Face]]:
    """Faces per sampled frame (SAMPLE_FPS). The detector is freed before returning."""
    import mediapipe as mp  # heavy: import only when rendering
    import numpy as np
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    w = DETECT_W
    h = max(2, round(src_h * w / src_w / 2) * 2)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={SAMPLE_FPS},scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )  # fmt: skip
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        min_detection_confidence=MIN_SCORE,
    )
    frames: list[list[Face]] = []
    frame_bytes = w * h * 3
    assert proc.stdout is not None
    with vision.FaceDetector.create_from_options(options) as detector:
        while len(buf := proc.stdout.read(frame_bytes)) == frame_bytes:
            rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            frames.append(
                [
                    Face(
                        cx=(d.bounding_box.origin_x + d.bounding_box.width / 2) / w,
                        size=d.bounding_box.width / w,
                        score=d.categories[0].score,
                    )
                    for d in result.detections
                ]
            )
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg frame sampling failed for {video}")
    return frames


def track_primary(
    frames: list[list[Face]], sample_fps: float = SAMPLE_FPS, switch_sec: float = 1.5
) -> list[float | None]:
    """Pick one face centre per frame. Follows the current person; switches to a bigger
    face only after it has been the biggest for `switch_sec` (no jittery switching), or
    at once when the current person is gone (scene cut)."""
    out: list[float | None] = []
    current: float | None = None
    pending = 0
    for faces in frames:
        if not faces:
            out.append(None)
            continue
        biggest = max(faces, key=lambda f: f.size * f.score)
        if current is None:
            current = biggest.cx
        else:
            near = min(faces, key=lambda f: abs(f.cx - current))
            if abs(near.cx - current) > SAME_FACE:  # current person left the frame
                current, pending = biggest.cx, 0
            elif abs(biggest.cx - near.cx) <= SAME_FACE:  # current person is the main one
                current, pending = near.cx, 0
            else:
                pending += 1
                if pending >= switch_sec * sample_fps:
                    current, pending = biggest.cx, 0
                else:
                    current = near.cx
        out.append(current)
    return out


def smooth_path(
    xs: list[float | None],
    alpha: float = 0.25,
    deadband: float = 0.04,
    snap: float = 0.2,
) -> list[float]:
    """Camera path from raw face centres: gaps are held, small movement is ignored
    (deadband), larger movement is eased (EMA), and big jumps cut instantly."""
    known = [x for x in xs if x is not None]
    if not known:
        return [0.5] * len(xs)
    last = known[0]
    filled: list[float] = []
    for x in xs:
        last = x if x is not None else last
        filled.append(last)

    ema = cam = filled[0]
    out: list[float] = []
    for x in filled:
        if abs(x - ema) > snap:
            ema = cam = x
        else:
            ema += alpha * (x - ema)
            if abs(ema - cam) > deadband:
                cam += 0.35 * (ema - cam)
        out.append(cam)
    return out


def crop_path(
    centers: list[float],
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
    n_frames: int,
    fps: float,
    sample_fps: float = SAMPLE_FPS,
) -> Crop:
    """Crop window with the output aspect ratio and its x position for every output frame
    (linear interpolation between samples, clamped to the frame)."""
    crop_w = min(src_w, int(src_h * out_w / out_h) // 2 * 2)
    crop_h = min(src_h, int(crop_w * out_h / out_w) // 2 * 2)
    max_x = src_w - crop_w
    centers = centers or [0.5]
    xs: list[int] = []
    for n in range(n_frames):
        pos = n / fps * sample_fps
        i = min(int(pos), len(centers) - 1)
        j = min(i + 1, len(centers) - 1)
        c = centers[i] + (centers[j] - centers[i]) * min(1.0, pos - i)
        xs.append(int(min(max_x, max(0, round(c * src_w - crop_w / 2)))))
    return Crop(w=crop_w, h=crop_h, y=(src_h - crop_h) // 2, xs=xs)


def sendcmd_script(crop: Crop, fps: float) -> str:
    """ffmpeg sendcmd file moving the crop filter's x; one line per change."""
    lines: list[str] = []
    prev: int | None = crop.xs[0] if crop.xs else None
    for n, x in enumerate(crop.xs):
        if n and x != prev:
            lines.append(f"{n / fps:.4f} crop x {x};")
            prev = x
    return "\n".join(lines) + "\n"
