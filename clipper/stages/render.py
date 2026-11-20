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
from clipper.render.captions import (
    ESTIMATED,
    CaptionStyle,
    FontMetrics,
    FramedStyle,
    build_ass,
    build_framed_ass,
    caption_words,
    layout_framed,
)
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
        caption_face_gap=r.caption_face_gap,
        caption_max_y_frac=r.caption_max_y_frac,
        hook_style=r.hook_style,
        hook_font=r.hook_font,
        titlecard_px=r.titlecard_font_size,
        titlecard_top_frac=r.titlecard_top_frac,
        titlecard_opacity=r.titlecard_opacity,
        hook_sec=r.hook_sec,
        hook_max_px=r.hook_font_size,
        hook_min_px=r.hook_min_font_size,
        hook_fallback_top_frac=r.hook_fallback_top_frac,
        hook_top_limit_frac=r.hook_top_limit_frac,
    )


def _framed_style(cfg: Config) -> FramedStyle:
    r = cfg.render
    return FramedStyle(
        width=r.width,
        height=r.height,
        video_px=r.framed_video_px,
        background=r.framed_background,
        headline_font=r.caption_font,
        headline_px=r.framed_headline_px,
        accent_color=r.framed_accent_color,
        context_font=r.hook_font,
        context_px=r.framed_context_px,
        context_color=r.framed_context_color,
        caption_px=r.framed_caption_px,
        caption_outline=r.framed_caption_outline,
        caption_above_bottom=r.framed_caption_above_bottom,
        nudge_up=r.framed_nudge_up,
    )


def _metrics(fonts_dir: Path, family: str) -> FontMetrics:
    metrics = FontMetrics.find(fonts_dir, family)
    if metrics is None:
        log.warning("font %r not found in %s: libass picks a system font, sizes are estimated",
                    family, fonts_dir)  # fmt: skip
        return ESTIMATED
    return metrics


def _fonts(cfg: Config, work: Path) -> tuple[FontMetrics, FontMetrics, str | None]:
    """Copy the bundled fonts into the work dir (so the filter path needs no escaping)
    and measure the caption and hook fonts. Falls back to estimates and system fonts."""
    src = cfg.fonts_dir
    files = [p for p in sorted(src.glob("*")) if p.suffix.lower() in FONT_SUFFIXES]
    fonts_arg = None
    if files:
        (work / "fonts").mkdir(exist_ok=True)
        for p in files:
            shutil.copy2(p, work / "fonts" / p.name)
        fonts_arg = "fonts"
    return _metrics(src, cfg.render.caption_font), _metrics(src, cfg.render.hook_font), fonts_arg


def render_clip(
    cfg: Config, clip: Clip, segment: Path, words: list[Word], campaign: Campaign
) -> Path:
    """`segment` holds exactly [clip.start_sec, clip.end_sec] of the source (t=0 = clip
    start); `words` are the full-video words. `clip.layout` picks framed, crop or fit.
    Returns the rendered clip path."""
    r = cfg.render
    if clip.layout not in ("framed", "crop", "fit"):
        raise ValueError(f"unknown layout {clip.layout!r} for {clip.id}")
    fit = clip.layout == "fit"
    framed = clip.layout == "framed"
    clips_dir = cfg.dir("clips")
    out = clips_dir / f"{clip.id}.mp4"
    ensure_free_space(clips_dir, cfg.machine.min_free_gb, 0.2)
    work = cfg.dir("work") / clip.id
    work.mkdir(parents=True, exist_ok=True)
    try:
        src_w, src_h, duration = probe_video(segment)

        # 1. faces: the crop path (crop layout) and where the hook goes (both layouts).
        #    The detector is freed before ffmpeg starts: one heavy step at a time.
        models = cfg.data_dir / "models"
        face_model = facetrack.ensure_model(models, r.face_model, "face")
        person_model = facetrack.ensure_model(models, r.person_model, "object")
        frames = facetrack.detect_faces(segment, face_model, src_w, src_h, person_model)
        n_face = sum(any(f.kind == "face" for f in fs) for fs in frames)
        n_person = sum(any(f.kind == "person" for f in fs) for fs in frames)
        track = facetrack.track_primary(frames)
        subjects = facetrack.tracked(frames, track)
        cuts = facetrack.shot_cuts(subjects)
        log.info("%s: %d samples: face %d, person only %d, nothing %d; shot cuts at %s",
                 clip.id, len(frames), n_face, n_person, len(frames) - n_face - n_person,
                 [round(i / facetrack.SAMPLE_FPS, 1) for i in sorted(cuts)])  # fmt: skip
        crop, cmd_file = None, None
        if not fit:
            n_frames = int(duration * r.fps) + 1
            centers = facetrack.smooth_path(track, cuts=cuts)
            cut_times = facetrack.scene_cuts(segment) if cuts else []
            if framed:  # square at source resolution, eyes at ~38% where there is room
                side = min(r.framed_video_px, src_w, src_h) // 2 * 2
                y_fracs = facetrack.smooth_path(
                    facetrack.square_y_fracs(subjects, src_h, side), cuts=cuts
                )
                crop = facetrack.crop_path(centers, src_w, src_h, 1, 1, n_frames, r.fps,
                                           cuts=cuts, cut_times=cut_times, max_side=side,
                                           y_fracs=y_fracs)  # fmt: skip
            else:
                crop = facetrack.crop_path(centers, src_w, src_h, r.width, r.height, n_frames,
                                           r.fps, cuts=cuts, cut_times=cut_times)  # fmt: skip
            if cut_times:
                log.info("%s: camera cuts at %s s", clip.id, [round(t, 2) for t in cut_times])
            script = facetrack.sendcmd_script(crop, r.fps)
            if script.strip():
                cmd_file = "crop.cmd"
                (work / cmd_file).write_text(script)

        def out_y(y: float) -> float:
            return facetrack.to_output_y(y, src_w, src_h, r.width, r.height, crop)

        top = facetrack.face_top(subjects, int(r.hook_sec * facetrack.SAMPLE_FPS) + 1)
        face_y = None if top is None else out_y(top)
        bottoms = [None if b is None else out_y(b) for b in facetrack.face_bottoms(subjects)]

        # 2. captions + hook
        metrics, hook_metrics, fonts_arg = _fonts(cfg, work)
        cap_words = caption_words(
            words_between(words, clip.start_sec, clip.end_sec), campaign.caption_fixes
        )
        frame = None
        if framed:
            fs = _framed_style(cfg)
            layout = layout_framed(clip.hook_text, clip.context_line, metrics, hook_metrics, fs)
            ass = build_framed_ass(cap_words, clip.accent_word, clip.context_line, layout, fs,
                                   metrics, hook_metrics, duration)  # fmt: skip
            frame = (layout.video_top, fs.video_px, fs.background)
        else:
            ass = build_ass(cap_words, clip.hook_text, _caption_style(cfg, fit), metrics, face_y,
                            bottoms, facetrack.SAMPLE_FPS, hook_metrics, duration)  # fmt: skip
        (work / "captions.ass").write_text(ass)

        # 3. encode to a temp name, so a crash never leaves a half-written clip in place
        tmp = work / "out.mp4"
        cmd = render_cmd(segment, tmp, crop, r, cfg.machine.video_encoder, cmd_file,
                         "captions.ass", fonts_arg, frame)  # fmt: skip
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
