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
from clipper.timing import Word, load_words, parse_ts, snap_clip

log = logging.getLogger(__name__)

PROMPT_PATH = REPO_ROOT / "prompts" / "pick_clips.md"


class PickedClip(BaseModel):
    start: str | float
    end: str | float
    hook_text: str = Field(min_length=1)
    title: str = Field(min_length=1)
    reason: str = ""
    score: float = 0.0


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


def load_clips(path: Path) -> list[Clip]:
    return [Clip(**d) for d in json.loads(path.read_text())]


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
    values = {
        "CANDIDATES": str(cfg.pipeline.candidates_per_video),
        "MIN_SEC": f"{lo:g}",
        "MAX_SEC": f"{hi:g}",
        "CAMPAIGN_RULES": campaign_rules_text(campaign),
        "TRANSCRIPT": compact.strip(),
    }
    prompt = PROMPT_PATH.read_text()
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", value)
    return prompt


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
) -> list[Clip]:
    """Snap picks to sentence boundaries; drop bad timestamps, wrong lengths and overlaps."""
    clips: list[Clip] = []
    for p in sorted(picked, key=lambda c: c.score, reverse=True):
        try:
            span = snap_clip(words, parse_ts(p.start), parse_ts(p.end), min_sec, max_sec)
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
        return load_clips(clips_path)

    text = _ask(cfg, conn, video_id, build_prompt(cfg, campaign, compact_path.read_text()))
    try:
        response = parse_response(text)
    except (ValueError, ValidationError) as exc:
        log.warning("invalid JSON from claude, retrying once: %s", exc)
        fix = (
            "Your previous reply was not valid for the required JSON schema.\n"
            f"Error: {exc}\n\n"
            'Return ONLY the corrected JSON object {"clips": [{"start", "end", "hook_text", '
            '"title", "reason", "score"}]}, no prose, no code fences.\n\n'
            f"Previous reply:\n{text}"
        )
        response = parse_response(_ask(cfg, conn, video_id, fix))

    lo, hi = cfg.clip_bounds(campaign)
    clips = resolve_clips(response.clips, load_words(words_path), video_id, campaign, lo, hi)
    if not clips:
        raise RuntimeError("claude returned no usable clips")
    clips_path.write_text(json.dumps([asdict(c) for c in clips], indent=2))
    for c in clips:
        db.upsert_clip(conn, c.id, c.video_id, c.start_sec, c.end_sec, c.hook_text, c.title,
                       c.reason, c.score, status="picked")  # fmt: skip
    log.info("picked %d clips", len(clips))
    return clips
