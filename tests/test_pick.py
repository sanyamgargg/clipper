import pytest

from clipper.config import Campaign, Config, Rules
from clipper.llm.claude_cli import extract_json
from clipper.stages.pick import PickedClip, build_prompt, parse_response, resolve_clips
from clipper.timing import Word

CAMPAIGN = Campaign(
    id="tjr",
    name="TJR",
    rules=Rules(required_hashtags=["#TJR"], look_for="Funny moments", must_feature="TJR speaks",
                no_negative_portrayal=True),
)  # fmt: skip


def make_words(text: str) -> list[Word]:
    out, t = [], 0.0
    for tok in text.split():
        out.append(Word(tok, round(t, 3), round(t + 0.4, 3)))
        t += 0.5
    return out


# sentences start at 0, 2.5, 5.0, 7.5 s
WORDS = make_words(
    "One two three four five. Six seven eight nine ten? "
    "Eleven twelve thirteen fourteen fifteen! Sixteen seventeen eighteen nineteen twenty."
)


def picked(start: str, end: str, score: float, title: str = "Title") -> PickedClip:
    return PickedClip(start=start, end=end, hook_text="Hook", title=title, score=score)


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"clips": []}\n```') == {"clips": []}
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json('{"clips": [')


def test_parse_response_validates_schema():
    ok = parse_response(
        '{"clips": [{"start": "01:00", "end": "01:30", "hook_text": "h", "title": "t",'
        ' "reason": "r", "score": 0.8}]}'
    )
    assert ok.clips[0].score == 0.8
    for bad in ['{"clips": []}', '{"clips": [{"start": "01:00"}]}', '{"other": 1}']:
        with pytest.raises(ValueError):
            parse_response(bad)


def test_resolve_clips_orders_by_score_drops_overlaps_and_bad_timestamps():
    clips = resolve_clips(
        [
            picked("00:00", "00:05", 0.5, "overlaps the best one"),
            picked("00:02", "00:07", 0.9, "best"),
            picked("00:07", "00:10", 0.1, "adjacent"),
            picked("x:y", "00:10", 0.99, "bad timestamp"),
        ],
        WORDS, "vid", CAMPAIGN, min_sec=2, max_sec=20,
    )  # fmt: skip
    assert [(c.id, c.title) for c in clips] == [("vid_01", "best"), ("vid_02", "adjacent")]
    assert clips[0].start_sec == pytest.approx(2.4) and clips[0].end_sec == pytest.approx(7.5)
    assert all(c.caption.endswith("#TJR") for c in clips)


def test_resolve_clips_drops_clips_outside_length_rules():
    assert resolve_clips([picked("00:00", "00:02", 0.9)], WORDS, "vid", CAMPAIGN, 5, 6) == []


def test_caption_does_not_repeat_hashtag():
    clips = resolve_clips([picked("00:00", "00:05", 0.9, "Wild #tjr take")], WORDS, "vid",
                          CAMPAIGN, 2, 20)  # fmt: skip
    assert clips[0].caption == "Wild #tjr take"


def test_build_prompt_fills_every_placeholder():
    cfg = Config(campaigns=[CAMPAIGN])
    prompt = build_prompt(cfg, CAMPAIGN, "[00:00] Hello there.")
    assert "{{" not in prompt
    assert "Pick 5 clips" in prompt and "20–45 seconds" in prompt
    assert "Look for: Funny moments" in prompt and "Must feature: TJR speaks" in prompt
    assert "portray the creator or brand negatively" in prompt
    assert prompt.rstrip().endswith("[00:00] Hello there.")
