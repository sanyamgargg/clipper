"""render: one segment mp4 + clip + words -> data/clips/<clip_id>.mp4 (+ .jpg, .json)."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from clipper.config import Campaign, Config
from clipper.disk import ensure_free_space
from clipper.render import facetrack
from clipper.render.captions import ESTIMATED, CaptionStyle, FontMetrics, build_ass, caption_words
from clipper.render.ffmpeg import probe_video, render_cmd, run, thumbnail_cmd
from clipper.stages.pick import Clip
from clipper.timing import Word, words_between

log = logging.getLogger(__name__)

FONT_SUFFIXES = (".ttf", ".otf")


def _caption_style(cfg: Config, fit: bool) -> CaptionStyle:
    r = cfg.render
    return CaptionStyle(
        width=r.width,
        height=r.height,
        font=r.caption_font,
        caption_px=r.caption_font_size,
        caption_y_frac=r.caption_y_frac_fit if fit else r.caption_y_frac,
        hook_style=r.hook_style,
        hook_sec=r.hook_sec,
        hook_max_px=r.hook_font_size,
        hook_min_px=r.hook_min_font_size,
        hook_fallback_top_frac=r.hook_fallback_top_frac,
        hook_top_limit_frac=r.hook_top_limit_frac,
    )


def _fonts(cfg: Config, work: Path) -> tuple[FontMetrics, str | None]:
    """Copy the bundled fonts into the work dir (so the filter path needs no escaping)
    and measure the caption font. Falls back to estimated metrics and system fonts."""
    src = cfg.fonts_dir
    files = [p for p in sorted(src.glob("*")) if p.suffix.lower() in FONT_SUFFIXES]
    fonts_arg = None
    if files:
        (work / "fonts").mkdir(exist_ok=True)
        for p in files:
            shutil.copy2(p, work / "fonts" / p.name)
        fonts_arg = "fonts"
    metrics = FontMetrics.find(src, cfg.render.caption_font)
    if metrics is None:
        log.warning("font %r not found in %s: libass picks a system font, sizes are estimated",
                    cfg.render.caption_font, src)  # fmt: skip
        metrics = ESTIMATED
    return metrics, fonts_arg


def render_clip(
    cfg: Config, clip: Clip, segment: Path, words: list[Word], campaign: Campaign
) -> Path:
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
        src_w, src_h, duration = probe_video(segment)

        # 1. faces: the crop path (crop layout) and where the hook goes (both layouts).
        #    The detector is freed before ffmpeg starts: one heavy step at a time.
        model = facetrack.ensure_model(cfg.data_dir / "models", r.face_model)
        faces = facetrack.detect_faces(segment, model, src_w, src_h)
        log.info("%s: face in %d/%d sampled frames", clip.id, sum(1 for f in faces if f),
                 len(faces))  # fmt: skip
        track = facetrack.track_primary(faces)
        crop, cmd_file = None, None
        if not fit:
            n_frames = int(duration * r.fps) + 1
            centers = facetrack.smooth_path(track)
            crop = facetrack.crop_path(centers, src_w, src_h, r.width, r.height, n_frames, r.fps)
            script = facetrack.sendcmd_script(crop, r.fps)
            if script.strip():
                cmd_file = "crop.cmd"
                (work / cmd_file).write_text(script)
        top = facetrack.face_top(faces, track, int(r.hook_sec * facetrack.SAMPLE_FPS) + 1)
        face_y = (
            None
            if top is None
            else facetrack.to_output_y(top, src_w, src_h, r.width, r.height, crop)
        )

        # 2. captions + hook
        metrics, fonts_arg = _fonts(cfg, work)
        cap_words = caption_words(
            words_between(words, clip.start_sec, clip.end_sec), campaign.caption_fixes
        )
        ass = build_ass(cap_words, clip.hook_text, _caption_style(cfg, fit), metrics, face_y)
        (work / "captions.ass").write_text(ass)

        # 3. encode to a temp name, so a crash never leaves a half-written clip in place
        tmp = work / "out.mp4"
        cmd = render_cmd(segment, tmp, crop, r, cfg.machine.video_encoder, cmd_file,
                         "captions.ass", fonts_arg)  # fmt: skip
        run(cmd, cwd=work)
        # thumbnail: a frame of this clip, after the hook text is gone
        run(thumbnail_cmd(tmp, clips_dir / f"{clip.id}.jpg", min(duration / 3, r.hook_sec + 4)))
        tmp.replace(out)
        (clips_dir / f"{clip.id}.json").write_text(json.dumps(asdict(clip), indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    log.info(
        "rendered %s (%s, hook %s, %.1f s, %.1f MB)",
        out.name, clip.layout, r.hook_style, duration, out.stat().st_size / 1e6,
    )  # fmt: skip
    return out
