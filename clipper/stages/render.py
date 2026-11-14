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


def render_clip(cfg: Config, clip: Clip, segment: Path, words: list[Word]) -> Path:
    """`segment` holds exactly [clip.start_sec, clip.end_sec] of the source (t=0 = clip
    start); `words` are the full-video words. Returns the rendered clip path."""
    r = cfg.render
    clips_dir = cfg.dir("clips")
    out = clips_dir / f"{clip.id}.mp4"
    ensure_free_space(clips_dir, cfg.machine.min_free_gb, 0.2)
    work = cfg.dir("work") / clip.id
    work.mkdir(parents=True, exist_ok=True)
    try:
        src_w, src_h, duration = probe_video(segment)

        # 1. face tracking (detector is freed before ffmpeg starts: one heavy step at a time)
        model = facetrack.ensure_model(cfg.data_dir / "models", r.face_model)
        faces = facetrack.detect_faces(segment, model, src_w, src_h)
        found = sum(1 for f in faces if f)
        log.info("%s: face in %d/%d sampled frames", clip.id, found, len(faces))
        centers = facetrack.smooth_path(facetrack.track_primary(faces))
        n_frames = int(duration * r.fps) + 1
        crop = facetrack.crop_path(centers, src_w, src_h, r.width, r.height, n_frames, r.fps)
        script = facetrack.sendcmd_script(crop, r.fps)
        cmd_file = None
        if script.strip():
            cmd_file = "crop.cmd"
            (work / cmd_file).write_text(script)

        # 2. captions + hook
        ass = build_ass(
            words_between(words, clip.start_sec, clip.end_sec),
            clip.hook_text,
            width=r.width,
            height=r.height,
            font=r.caption_font,
            font_size=r.caption_font_size,
            y_frac=r.caption_y_frac,
            hook_sec=r.hook_sec,
        )
        (work / "captions.ass").write_text(ass)

        # 3. encode to a temp name, so a crash never leaves a half-written clip in place
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
    log.info("rendered %s (%.1f s, %.1f MB)", out.name, duration, out.stat().st_size / 1e6)
    return out
