"""Face tracking -> smoothed crop-x path for a vertical crop, plus where the face is
(for hook and caption placement).

detect_faces() is the only heavy part (ffmpeg pipe + MediaPipe); everything else is pure.
Frames without a face fall back to MediaPipe's object detector (class "person"), so wide
shots still follow the speaker; only if neither is found does the crop hold/centre.
"""

from __future__ import annotations

import logging
import re
import shutil
import ssl
import subprocess
import urllib.request
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from clipper import proc

log = logging.getLogger(__name__)

SAMPLE_FPS = 5
DETECT_W = 480  # frames are scaled down to this width for detection
MIN_SCORE = 0.6  # face detector confidence
PERSON_SCORE = 0.4  # object detector confidence for "person"
SAME_FACE = 0.12  # faces within this x distance (fraction of width) are the same person
SHOT_SIZE_RATIO = 1.8  # tracked subject this much bigger/smaller from one sample: a cut
SHOT_JUMP = 0.2  # ... or moved this far (fraction of width): a cut, e.g. to another speaker
SCENE_THRESHOLD = 0.3  # ffmpeg scene-change score that counts as a camera cut
# BlazeFace boxes end around the mouth; the chin (and beard) is about this much of the
# box height lower. Measured on TJR close-ups: box bottom 0.65, chin 0.71 of the frame.
CHIN_EXTEND = 0.25
MODEL_URLS = {
    "face": "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "{name}/float16/latest/{name}.tflite",
    "object": "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "{name}/int8/latest/{name}.tflite",
}

Kind = Literal["face", "person"]


@dataclass(frozen=True)
class Face:
    """A detected face, or a person when no face was found in that frame."""

    cx: float  # centre x, 0..1 of frame width
    size: float  # box width, 0..1 of frame width
    score: float
    top: float = 0.0  # box top, 0..1 of frame height
    bottom: float = 0.0  # box bottom, 0..1 of frame height
    kind: Kind = "face"
    eye_y: float | None = None  # mean of the two eye keypoints, 0..1 of frame height


@dataclass(frozen=True)
class Crop:
    w: int
    h: int
    y: int
    xs: list[int]  # crop x per output frame
    ys: list[int] | None = None  # crop y per output frame, if it moves (else `y`)


def ensure_model(models_dir: Path, name: str, kind: Literal["face", "object"] = "face") -> Path:
    path = models_dir / f"{name}.tflite"
    if not path.exists():
        import certifi  # python.org builds ship without CA certificates

        models_dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s model %s", kind, name)
        tmp = path.with_suffix(".tmp")
        context = ssl.create_default_context(cafile=certifi.where())
        url = MODEL_URLS[kind].format(name=name)
        with urllib.request.urlopen(url, context=context, timeout=60) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(path)
    return path


def _box(detection: Any, w: int, h: int, kind: Kind) -> Face:
    b = detection.bounding_box
    keypoints = getattr(detection, "keypoints", None) or []
    eye_y = (
        (keypoints[0].y + keypoints[1].y) / 2 if kind == "face" and len(keypoints) >= 2 else None
    )
    return Face(
        cx=(b.origin_x + b.width / 2) / w,
        size=b.width / w,
        score=detection.categories[0].score,
        top=b.origin_y / h,
        bottom=(b.origin_y + b.height) / h,
        kind=kind,
        eye_y=eye_y,
    )


def detect_faces(
    video: Path, model: Path, src_w: int, src_h: int, person_model: Path | None = None
) -> list[list[Face]]:
    """Faces per sampled frame (SAMPLE_FPS); frames without a face get the people the
    object detector finds (if `person_model` is given). Detectors are freed on return."""
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
    face_options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        min_detection_confidence=MIN_SCORE,
    )
    frames: list[list[Face]] = []
    frame_bytes = w * h * 3
    assert proc.stdout is not None
    with ExitStack() as stack:
        faces_det = stack.enter_context(vision.FaceDetector.create_from_options(face_options))
        people_det = None  # created on the first frame that needs it
        while len(buf := proc.stdout.read(frame_bytes)) == frame_bytes:
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            )
            found = [_box(d, w, h, "face") for d in faces_det.detect(image).detections]
            if not found and person_model is not None:
                if people_det is None:
                    people_det = stack.enter_context(
                        vision.ObjectDetector.create_from_options(
                            vision.ObjectDetectorOptions(
                                base_options=mp_python.BaseOptions(
                                    model_asset_path=str(person_model)
                                ),
                                score_threshold=PERSON_SCORE,
                                category_allowlist=["person"],
                                max_results=5,
                            )
                        )
                    )
                found = [_box(d, w, h, "person") for d in people_det.detect(image).detections]
            frames.append(found)
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg frame sampling failed for {video}")
    return frames


def track_primary(
    frames: list[list[Face]], sample_fps: float = SAMPLE_FPS, switch_sec: float = 1.5
) -> list[float | None]:
    """Pick one centre per frame. Follows the current person; switches to a bigger one
    only after it has been the biggest for `switch_sec` (no jittery switching), or at
    once when the current person is gone (scene cut). Largest = biggest box x score."""
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


def tracked(frames: list[list[Face]], track: list[float | None]) -> list[Face | None]:
    """The detection being followed in each sampled frame (None where nothing is)."""
    return [
        min(faces, key=lambda f: abs(f.cx - x)) if faces and x is not None else None
        for faces, x in zip(frames, track, strict=False)
    ]


def shot_cuts(subjects: list[Face | None]) -> set[int]:
    """Samples where the followed subject changes abruptly (face <-> person, size by
    SHOT_SIZE_RATIO, or position by SHOT_JUMP): almost always a camera cut. The crop
    jumps there instead of easing or panning across."""
    cuts: set[int] = set()
    prev: Face | None = None
    for i, s in enumerate(subjects):
        if s is None:
            continue
        if prev is not None:
            ratio = max(s.size, prev.size) / max(1e-6, min(s.size, prev.size))
            if s.kind != prev.kind or ratio >= SHOT_SIZE_RATIO or abs(s.cx - prev.cx) > SHOT_JUMP:
                cuts.add(i)
        prev = s
    return cuts


def scene_cuts(video: Path, threshold: float = SCENE_THRESHOLD) -> list[float]:
    """Times (s) of camera cuts in the video, from ffmpeg's scene-change score."""
    result = proc.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf",
         f"scale=256:-2,select='gt(scene,{threshold})',metadata=print:file=-", "-f", "null", "-"]
    )  # fmt: skip
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg scene detection failed: {result.stderr.strip()[-400:]}")
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", result.stdout)]


def smooth_path(
    xs: list[float | None],
    alpha: float = 0.25,
    deadband: float = 0.04,
    snap: float = SHOT_JUMP,
    cuts: frozenset[int] | set[int] = frozenset(),
) -> list[float]:
    """Camera path from raw centres: gaps are held, small movement is ignored (deadband),
    larger movement is eased (EMA), and big jumps or shot `cuts` cut instantly."""
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
    for i, x in enumerate(filled):
        if i in cuts or abs(x - ema) > snap:
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
    cuts: frozenset[int] | set[int] = frozenset(),
    cut_times: list[float] | tuple[float, ...] = (),
    max_side: int | None = None,
    y_fracs: list[float] | None = None,
) -> Crop:
    """Crop window with the output aspect ratio and its position for every output frame:
    linear interpolation between samples, clamped to the frame. At a shot cut (sample j
    in `cuts`) the crop does not interpolate but switches from sample j-1 to j, at the
    camera cut in `cut_times` between them if there is one, else at sample j.
    `max_side` caps a square crop (framed layout: 1080 source px, never upscaled for
    1080p); `y_fracs` then places it vertically (0 = top .. 1 = bottom of the room)."""
    crop_w = min(src_w, int(src_h * out_w / out_h) // 2 * 2)
    if max_side:
        crop_w = min(crop_w, max_side // 2 * 2)
    crop_h = min(src_h, int(crop_w * out_h / out_w) // 2 * 2)
    max_x, room = src_w - crop_w, src_h - crop_h
    xs = [
        int(min(max_x, max(0, round(c * src_w - crop_w / 2))))
        for c in _per_frame(centers or [0.5], n_frames, fps, sample_fps, cuts, cut_times)
    ]
    ys = None
    if y_fracs and room > 0:
        ys = [
            int(min(room, max(0, round(f * room))))
            for f in _per_frame(y_fracs, n_frames, fps, sample_fps, cuts, cut_times)
        ]
    return Crop(w=crop_w, h=crop_h, y=ys[0] if ys else room // 2, xs=xs, ys=ys)


def _per_frame(
    samples: list[float],
    n_frames: int,
    fps: float,
    sample_fps: float,
    cuts: frozenset[int] | set[int],
    cut_times: list[float] | tuple[float, ...],
) -> list[float]:
    """Sample values (SAMPLE_FPS) to one value per output frame; no easing across cuts."""
    last = len(samples) - 1
    switch: dict[int, float] = {}
    for j in cuts:
        lo, hi = (j - 1) / sample_fps, (j + 0.5) / sample_fps
        found = [t for t in cut_times if lo < t <= hi]
        switch[j] = found[0] if found else j / sample_fps
    out: list[float] = []
    for n in range(n_frames):
        t = n / fps
        pos = t * sample_fps
        i = min(int(pos), last)
        j = min(i + 1, last)
        if j in cuts and j != i:  # before sample j: old shot until the switch
            v = samples[j] if t >= switch[j] else samples[i]
        elif i in cuts and i > 0 and t < switch[i]:  # camera cut comes just after sample i
            v = samples[i - 1]
        else:
            v = samples[i] + (samples[j] - samples[i]) * min(1.0, pos - i)
        out.append(v)
    return out


HEAD_EXTEND = 0.35  # hair above the face box top, as a fraction of the box height
EYE_FRAC = 0.38  # framed layout: eyes at this fraction of the square's height


def square_y_fracs(
    subjects: list[Face | None], src_h: int, side: int, eye_frac: float = EYE_FRAC
) -> list[float | None]:
    """Vertical place of a `side`-px square per sample, as a fraction of the room
    (src_h - side): eyes at `eye_frac` of the square, but never cutting the top of the
    head. All 0 when the square already spans the full height (1080p sources)."""
    room = src_h - side
    if room <= 0:
        return [0.0] * len(subjects)
    out: list[float | None] = []
    for s in subjects:
        if s is None or s.kind != "face":
            out.append(None)
            continue
        box = s.bottom - s.top
        eye = (s.eye_y if s.eye_y is not None else s.top + 0.4 * box) * src_h
        head_top = (s.top - HEAD_EXTEND * box) * src_h
        y = min(eye - eye_frac * side, head_top)
        out.append(min(1.0, max(0.0, y / room)))
    return out


def sendcmd_script(crop: Crop, fps: float) -> str:
    """ffmpeg sendcmd file moving the crop filter's x (and y); one line per change."""
    lines: list[str] = []
    ys = crop.ys or [crop.y] * len(crop.xs)
    prev = (crop.xs[0], ys[0]) if crop.xs else None
    for n, (x, y) in enumerate(zip(crop.xs, ys, strict=True)):
        if n and prev and (x, y) != prev:
            cmds = []
            if x != prev[0]:
                cmds.append(f"crop x {x}")
            if y != prev[1]:
                cmds.append(f"crop y {y}")
            lines.append(f"{n / fps:.4f} {', '.join(cmds)};")
            prev = (x, y)
    return "\n".join(lines) + "\n"


def face_top(subjects: list[Face | None], n: int) -> float | None:
    """Highest top edge (0..1 of frame height) of the followed face over the first `n`
    samples, i.e. while the hook is on screen. People (no face found) do not count."""
    tops = [s.top for s in subjects[:n] if s is not None and s.kind == "face"]
    return min(tops) if tops else None


def face_bottoms(subjects: list[Face | None]) -> list[float | None]:
    """Estimated chin (0..1 of frame height) of the followed face per sample, or None.
    The detector's box stops around the mouth, so it is extended by CHIN_EXTEND."""
    return [
        s.bottom + CHIN_EXTEND * (s.bottom - s.top) if s is not None and s.kind == "face" else None
        for s in subjects
    ]


def to_output_y(
    y: float, src_w: int, src_h: int, out_w: int, out_h: int, crop: Crop | None
) -> float:
    """A source y (0..1 of height) in output pixels, for the crop layout (`crop`) or the
    fit layout (`crop` is None: frame scaled to fit and centred)."""
    if crop is None:
        scale = min(out_w / src_w, out_h / src_h)
        return (out_h - src_h * scale) / 2 + y * src_h * scale
    return (y * src_h - crop.y) * out_h / crop.h
