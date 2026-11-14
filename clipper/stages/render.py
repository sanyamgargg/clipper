"""render: one segment mp4 + clip + words -> data/clips/<clip_id>.mp4 (+ .jpg, .json)."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from clipper.config import Config
from clipper.disk import ensure_free_space
from clipper.render import facetrack
from clipper.render.captions import build_ass
from clipper.render.ffmpeg import probe_video, render_cmd, run, thumbnail_cmd
from clipper.stages.pick import Clip
from clipper.timing import Word, words_between

log = logging.getLogger(__name__)


def _face_crop(
    cfg: Config, clip: Clip, segment: Path, work: Path
) -> tuple[facetrack.Crop, str | None]:
    """Face-tracked crop path for the `crop` layout, and its sendcmd file name (if it moves)."""
    r = cfg.render
    src_w, src_h, duration = probe_video(segment)
    model = facetrack.ensure_model(cfg.data_dir / "models", r.face_model)
    # the detector is freed before ffmpeg starts: one heavy step at a time
    faces = facetrack.detect_faces(segment, model, src_w, src_h)
    log.info("%s: face in %d/%d sampled frames", clip.id, sum(1 for f in faces if f), len(faces))
    centers = facetrack.smooth_path(facetrack.track_primary(faces))
    n_frames = int(duration * r.fps) + 1
    crop = facetrack.crop_path(centers, src_w, src_h, r.width, r.height, n_frames, r.fps)
    script = facetrack.sendcmd_script(crop, r.fps)
    if not script.strip():
        return crop, None
    (work / "crop.cmd").write_text(script)
    return crop, "crop.cmd"


def render_clip(cfg: Config, clip: Clip, segment: Path, words: list[Word]) -> Path:
    """`segment` holds exactly [clip.start_sec, clip.end_sec] of the source (t=0 = clip
    start); `words` are the full-video words. `clip.layout` picks crop or fit.
    Returns the rendered clip path."""
    r = cfg.render
    if clip.layout not in ("crop", "fit"):
        raise ValueError(f"unknown layout {clip.layout!r} for {clip.id}")
    fit = clip.layout == "fit"
    clips_dir = cfg.dir("clips")
    out = clips_dir / f"{clip.id}.mp4"
    ensure_free_space(clips_dir, cfg.machine.min_free_gb, 0.2)
    work = cfg.dir("work") / clip.id
    work.mkdir(parents=True, exist_ok=True)
    try:
        duration = probe_video(segment)[2]
        crop, cmd_file = (None, None) if fit else _face_crop(cfg, clip, segment, work)

        ass = build_ass(
            words_between(words, clip.start_sec, clip.end_sec),
            clip.hook_text,
            width=r.width,
            height=r.height,
            font=r.caption_font,
            font_size=r.caption_font_size,
            y_frac=r.caption_y_frac_fit if fit else r.caption_y_frac,
            hook_sec=r.hook_sec,
            hook_size=r.hook_font_size,
            hook_y_frac=r.hook_y_frac,
        )
        (work / "captions.ass").write_text(ass)

        # encode to a temp name, so a crash never leaves a half-written clip in place
        tmp = work / "out.mp4"
        run(
            render_cmd(segment, tmp, crop, r, cfg.machine.video_encoder, cmd_file, "captions.ass"),
            cwd=work,
        )
        # thumbnail: a frame of this clip, after the hook text is gone
        run(thumbnail_cmd(tmp, clips_dir / f"{clip.id}.jpg", min(duration / 3, r.hook_sec + 4)))
        tmp.replace(out)
        (clips_dir / f"{clip.id}.json").write_text(json.dumps(asdict(clip), indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    log.info(
        "rendered %s (%s, %.1f s, %.1f MB)",
        out.name, clip.layout, duration, out.stat().st_size / 1e6,
    )  # fmt: skip
    return out
