"""pick: compact transcript + campaign rules -> clips (the only Claude step).

Contract: writes data/transcripts/<video_id>.clips.json and rows in `clips`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from clipper import db
from clipper.config import REPO_ROOT, Campaign, Config
from clipper.llm.claude_cli import ClaudeCLIError, extract_json, run_claude
from clipper.timing import Word, load_words, parse_ts, snap_clip, words_between

log = logging.getLogger(__name__)

PROMPT_PATH = REPO_ROOT / "prompts" / "pick_clips.md"
HOOK_RULES_PATH = REPO_ROOT / "prompts" / "hook_rules.md"
REHOOK_PATH = REPO_ROOT / "prompts" / "regenerate_hook.md"
HOOK_WORDS = (6, 10)


def fill(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def hook_problems(hook: str) -> list[str]:
    """Hook rules that can be checked mechanically (prompts/hook_rules.md)."""
    words = hook.split()
    problems = []
    lo, hi = HOOK_WORDS
    if not lo <= len(words) <= hi:
        problems.append(f"{len(words)} words instead of {lo}-{hi}")
    shouting = [w for w in words if sum(c.isalpha() for c in w) >= 4 and w.upper() == w]
    if shouting:
        problems.append(f"ALL CAPS words: {', '.join(shouting)}")
    if any(c in hook for c in '!#"“”'):
        problems.append("no exclamation marks, hashtags or quotation marks")
    return problems


class PickedClip(BaseModel):
    start: str | float
    end: str | float
    hook_text: str = Field(min_length=1)
    title: str = Field(min_length=1)
    reason: str = ""
    score: float = 0.0
    story: bool = False  # a story may run up to pipeline.clip_story_max_sec


class PickResponse(BaseModel):
    clips: list[PickedClip] = Field(min_length=1)


@dataclass
class Clip:
    id: str
    video_id: str
    start_sec: float
    end_sec: float
    hook_text: str
    title: str
    caption: str  # title + required campaign hashtags/text, ready to paste when posting
    reason: str
    score: float
    layout: str = "crop"  # crop (face-tracked 9:16 crop) | fit (full frame on a blurred fill)
    story: bool = False  # Claude marked it a story: allowed up to clip_story_max_sec


def load_clips(path: Path) -> list[Clip]:
    return [Clip(**d) for d in json.loads(path.read_text())]


def save_clips(clips: list[Clip], path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps([asdict(c) for c in clips], indent=2))
    tmp.replace(path)  # atomic: a crash never leaves a half-written picks file


def campaign_rules_text(campaign: Campaign) -> str:
    r = campaign.rules
    lines = [f"Campaign: {campaign.name}"]
    if r.look_for:
        lines.append(f"Look for: {r.look_for}")
    if r.must_feature:
        lines.append(f"Must feature: {r.must_feature}")
    if r.no_negative_portrayal:
        lines.append("Never pick moments that portray the creator or brand negatively.")
    lines.append("Language: English only (hook_text and title too).")
    if r.notes:
        lines.append(f"Notes: {r.notes}")
    return "\n".join(f"- {line}" for line in lines)


def build_prompt(cfg: Config, campaign: Campaign, compact: str) -> str:
    lo, hi = cfg.clip_bounds(campaign)
    story_hi = cfg.clip_bounds(campaign, story=True)[1]
    p = cfg.pipeline
    values = {
        "CANDIDATES": str(p.candidates_per_video),
        "MIN_SEC": f"{lo:g}",
        "MAX_SEC": f"{hi:g}",
        "TARGET_MIN_SEC": f"{max(lo, p.clip_target_min_sec):g}",
        "TARGET_MAX_SEC": f"{min(hi, p.clip_target_max_sec):g}",
        "STORY_MAX_SEC": f"{story_hi:g}",
        "HOOK_RULES": HOOK_RULES_PATH.read_text().strip(),
        "CAMPAIGN_RULES": campaign_rules_text(campaign),
        "TRANSCRIPT": compact.strip(),
    }
    return fill(PROMPT_PATH.read_text(), values)


class HookReply(BaseModel):
    hook_text: str = Field(min_length=1)


def _parse_hook(text: str) -> str:
    hook = HookReply.model_validate(extract_json(text)).hook_text
    return " ".join(hook.strip().strip("\"“”'").split()).rstrip(".")


def regenerate_hook(
    cfg: Config,
    conn: sqlite3.Connection,
    clip: Clip,
    campaign: Campaign,
    words: list[Word],
    video_title: str,
) -> str:
    """A new hook for an existing clip, written from what is said in it (one Claude call,
    plus one retry if the reply breaks the hook rules). Raises ClaudeCLIError/ValueError."""
    said = " ".join(w.w for w in words_between(words, clip.start_sec, clip.end_sec))
    prompt = fill(
        REHOOK_PATH.read_text(),
        {
            "HOOK_RULES": HOOK_RULES_PATH.read_text().strip(),
            "CAMPAIGN": campaign.name,
            "VIDEO_TITLE": video_title or "(unknown)",
            "CLIP_TITLE": clip.title,
            "OLD_HOOK": clip.hook_text,
            "CLIP_TRANSCRIPT": said,
        },
    )
    reply = _ask(cfg, conn, clip.video_id, prompt)
    try:
        hook = _parse_hook(reply)
    except (ValueError, ValidationError):
        hook = ""
    if problems := (hook_problems(hook) if hook else ["no valid JSON"]):
        log.warning("%s: hook %r breaks the rules (%s); asking once more",
                    clip.id, hook, "; ".join(problems))  # fmt: skip
        fix = (
            f"Your hook {hook!r} breaks these rules: {'; '.join(problems)}.\n"
            'Return ONLY the corrected JSON object {"hook_text": "..."}.\n\n' + prompt
        )
        retry = _parse_hook(_ask(cfg, conn, clip.video_id, fix))
        if not hook or len(hook_problems(retry)) <= len(problems):
            hook = retry
        if left := hook_problems(hook):
            log.warning("%s: using hook %r anyway (%s)", clip.id, hook, "; ".join(left))
    return hook


def parse_response(text: str) -> PickResponse:
    """Raises ValueError (incl. pydantic ValidationError) if the reply is not valid."""
    return PickResponse.model_validate(extract_json(text))


def make_caption(title: str, campaign: Campaign) -> str:
    parts = [title.strip(), campaign.rules.required_caption_text.strip()]
    parts += [h for h in campaign.rules.required_hashtags if h.lower() not in title.lower()]
    return " ".join(p for p in parts if p)


def resolve_clips(
    picked: list[PickedClip],
    words: list[Word],
    video_id: str,
    campaign: Campaign,
    min_sec: float,
    max_sec: float,
    story_max_sec: float | None = None,
) -> list[Clip]:
    """Snap picks to sentence boundaries; drop bad timestamps, wrong lengths and overlaps.
    Picks marked `story` may run up to `story_max_sec`; others are trimmed to `max_sec`."""
    clips: list[Clip] = []
    for p in sorted(picked, key=lambda c: c.score, reverse=True):
        limit = max(max_sec, story_max_sec or max_sec) if p.story else max_sec
        try:
            span = snap_clip(words, parse_ts(p.start), parse_ts(p.end), min_sec, limit)
        except ValueError:
            span = None
        if span is None:
            log.warning(
                "dropping pick %s-%s (%r): cannot fit length rules", p.start, p.end, p.title
            )
            continue
        s, e = span
        # back-to-back clips share a few tenths of padding: that is not an overlap
        if any(min(e, c.end_sec) - max(s, c.start_sec) > 0.5 for c in clips):
            log.warning("dropping pick %s-%s (%r): overlaps a better clip", p.start, p.end, p.title)
            continue
        clips.append(
            Clip(
                id=f"{video_id}_{len(clips) + 1:02d}",
                video_id=video_id,
                start_sec=s,
                end_sec=e,
                hook_text=p.hook_text.strip(),
                title=p.title.strip(),
                caption=make_caption(p.title, campaign),
                reason=p.reason.strip(),
                score=p.score,
                story=p.story,
            )
        )
    return clips


def _ask(cfg: Config, conn: sqlite3.Connection, video_id: str, prompt: str) -> str:
    """One logged Claude call. ClaudeCLIError propagates (caller stops the stage)."""
    try:
        result = run_claude(cfg, prompt, cwd=cfg.dir("logs"))
    except ClaudeCLIError as exc:
        r = exc.result
        db.log_llm_call(conn, video_id, r.model, r.input_tokens, r.output_tokens, r.cost_usd,
                        r.exit_code)  # fmt: skip
        raise
    db.log_llm_call(conn, video_id, result.model, result.input_tokens, result.output_tokens,
                    result.cost_usd, result.exit_code)  # fmt: skip
    log.info("claude: %s in / %s out tokens, cost $%s", result.input_tokens, result.output_tokens,
             result.cost_usd)  # fmt: skip
    return result.text


def pick(
    cfg: Config,
    conn: sqlite3.Connection,
    video_id: str,
    campaign: Campaign,
    words_path: Path,
    compact_path: Path,
) -> list[Clip]:
    """Returns the picked clips. Skips Claude if <video_id>.clips.json already exists."""
    clips_path = cfg.dir("transcripts") / f"{video_id}.clips.json"
    if clips_path.exists():
        log.info("picks already present: %s", clips_path.name)
        clips = load_clips(clips_path)
        _add_rows(conn, clips)  # in case a crash came between the file and the rows
        return clips

    text = _ask(cfg, conn, video_id, build_prompt(cfg, campaign, compact_path.read_text()))
    try:
        response = parse_response(text)
    except (ValueError, ValidationError) as exc:
        log.warning("invalid JSON from claude, retrying once: %s", exc)
        fix = (
            "Your previous reply was not valid for the required JSON schema.\n"
            f"Error: {exc}\n\n"
            'Return ONLY the corrected JSON object {"clips": [{"start", "end", "hook_text", '
            '"title", "reason", "score", "story"}]}, no prose, no code fences.\n\n'
            f"Previous reply:\n{text}"
        )
        response = parse_response(_ask(cfg, conn, video_id, fix))

    lo, hi = cfg.clip_bounds(campaign)
    story_hi = cfg.clip_bounds(campaign, story=True)[1]
    clips = resolve_clips(response.clips, load_words(words_path), video_id, campaign, lo, hi,
                          story_hi)  # fmt: skip
    if stories := sum(c.story for c in clips):
        log.info("%d of %d clips are stories (up to %g s)", stories, len(clips), story_hi)
    if not clips:
        raise RuntimeError("claude returned no usable clips")
    for c in clips:
        c.layout = cfg.render.default_layout
    save_clips(clips, clips_path)
    _add_rows(conn, clips)
    log.info("picked %d clips", len(clips))
    return clips


def _add_rows(conn: sqlite3.Connection, clips: list[Clip]) -> None:
    for c in clips:
        db.add_clip(conn, c.id, c.video_id, c.start_sec, c.end_sec, c.hook_text, c.title,
                    c.reason, c.score, c.layout)  # fmt: skip
