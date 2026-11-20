"""pick: compact transcript + campaign rules -> clips (the only Claude step).

Contract: writes data/transcripts/<video_id>.clips.json and rows in `clips`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from clipper import db
from clipper.config import REPO_ROOT, Campaign, Config
from clipper.llm.claude_cli import ClaudeCLIError, extract_json, run_claude
from clipper.post_text import clean_topic_tags
from clipper.render.captions import accent_spans
from clipper.timing import Word, load_words, parse_ts, snap_clip, words_between

log = logging.getLogger(__name__)

PROMPT_PATH = REPO_ROOT / "prompts" / "pick_clips.md"
HOOK_RULES_PATH = REPO_ROOT / "prompts" / "hook_rules.md"
REHOOK_PATH = REPO_ROOT / "prompts" / "regenerate_hook.md"
HOOK_WORDS = (5, 9)
CONTEXT_MAX_WORDS = 6


def fill(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def hook_problems(hook: str, accent: str | None = None, context: str | None = None) -> list[str]:
    """Headline rules that can be checked mechanically (prompts/hook_rules.md). `accent`
    and `context` are checked when given (None = not part of this check)."""
    words = hook.split()
    problems = []
    lo, hi = HOOK_WORDS
    if not lo <= len(words) <= hi:
        problems.append(f"{len(words)} words instead of {lo}-{hi}")
    if ":" not in hook:
        problems.append('no "setup: punch" colon')
    shouting = [w for w in words if sum(c.isalpha() for c in w) >= 4 and w.upper() == w]
    if shouting:
        problems.append(f"ALL CAPS words: {', '.join(shouting)}")
    if any(c in hook for c in '!#"“”'):
        problems.append("no exclamation marks, hashtags or quotation marks")
    if accent is not None and not accent_spans(words, accent):
        problems.append("accent_word must be copied exactly from hook_text")
    if context is not None and not 1 <= len(context.split()) <= CONTEXT_MAX_WORDS:
        problems.append(f"context_line must be 1-{CONTEXT_MAX_WORDS} words")
    return problems


class PickedClip(BaseModel):
    start: str | float
    end: str | float
    hook_text: str = Field(min_length=1)
    accent_word: str = ""  # the key word/number of hook_text, shown in the accent colour
    context_line: str = ""  # who/what, max 6 words, under the headline
    title: str = Field(min_length=1)
    reason: str = ""
    score: float = 0.0
    story: bool = False  # a story may run up to pipeline.clip_story_max_sec
    hashtags: list[str] = Field(default_factory=list)  # up to 2 topic tags, cleaned later


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
    layout: str = "crop"  # framed (square on a canvas) | crop (9:16) | fit (blurred fill)
    story: bool = False  # Claude marked it a story: allowed up to clip_story_max_sec
    accent_word: str = ""  # framed headline: the word/number shown in the accent colour
    context_line: str = ""  # framed: who/what under the headline
    hashtags: list[str] = field(default_factory=list)  # Claude's topic tags (max 2, cleaned)


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


class Headline(BaseModel):
    hook_text: str = Field(min_length=1)
    accent_word: str = ""
    context_line: str = ""

    def problems(self) -> list[str]:
        return hook_problems(self.hook_text, self.accent_word, self.context_line)


def _tidy(text: str) -> str:
    return " ".join(text.strip().strip("\"“”'").split())


def _parse_headline(text: str) -> Headline:
    h = Headline.model_validate(extract_json(text))
    return Headline(
        hook_text=_tidy(h.hook_text).rstrip("."),
        accent_word=_tidy(h.accent_word),
        context_line=_tidy(h.context_line).rstrip("."),
    )


def regenerate_headline(
    cfg: Config,
    conn: sqlite3.Connection,
    clip: Clip,
    campaign: Campaign,
    words: list[Word],
    video_title: str,
) -> Headline:
    """A new hook, accent word and context line for an existing clip, written from what is
    said in it (one Claude call, plus one retry if the reply breaks the rules). Raises
    ClaudeCLIError/ValueError."""
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
        head: Headline | None = _parse_headline(reply)
    except (ValueError, ValidationError):
        head = None
    if problems := (head.problems() if head else ["no valid JSON"]):
        log.warning("%s: headline %s breaks the rules (%s); asking once more",
                    clip.id, head and head.model_dump(), "; ".join(problems))  # fmt: skip
        fix = (
            f"Your reply {head and head.model_dump_json()} breaks these rules: "
            f"{'; '.join(problems)}.\nReturn ONLY the corrected JSON object "
            '{"hook_text": "...", "accent_word": "...", "context_line": "..."}.\n\n' + prompt
        )
        retry = _parse_headline(_ask(cfg, conn, clip.video_id, fix))
        if head is None or len(retry.problems()) <= len(problems):
            head = retry
        if left := head.problems():
            log.warning("%s: using %s anyway (%s)", clip.id, head.model_dump(), "; ".join(left))
    return head


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
                accent_word=_tidy(p.accent_word),
                context_line=_tidy(p.context_line),
                hashtags=clean_topic_tags(p.hashtags),
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
