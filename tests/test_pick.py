import pytest

from clipper import db
from clipper.config import Campaign, Config, Machine, Rules
from clipper.llm.claude_cli import LLMResult, extract_json
from clipper.stages import pick as pick_mod
from clipper.stages.pick import (
    Clip,
    PickedClip,
    build_prompt,
    hook_problems,
    parse_response,
    regenerate_headline,
    resolve_clips,
)
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


def test_story_clips_may_run_longer():
    # 4 sentences of 2.5 s; normal clips are capped at 5.5 s, stories at 11 s
    long_pick = {"start": "00:00", "end": "00:10", "hook_text": "h", "title": "t", "score": 0.9}
    normal = resolve_clips([PickedClip(**long_pick)], WORDS, "vid", CAMPAIGN, 2, 5.5, 11)
    story = resolve_clips([PickedClip(**long_pick, story=True)], WORDS, "vid", CAMPAIGN, 2, 5.5, 11)
    assert normal[0].end_sec - normal[0].start_sec <= 5.5 + 0.4  # trimmed to whole sentences
    assert story[0].end_sec - story[0].start_sec > 9  # keeps all four sentences
    assert (normal[0].story, story[0].story) == (False, True)
    assert (
        parse_response(
            '{"clips": [{"start": "0:01", "end": "0:30", "hook_text": "h", "title": "t"}]}'
        )
        .clips[0]
        .story
        is False
    )  # optional in Claude's reply


def test_clip_bounds_story_and_campaign_limits():
    cfg = Config(campaigns=[CAMPAIGN])  # pipeline 18-35, story 45; campaign rules 15-60
    assert cfg.clip_bounds(CAMPAIGN) == (18, 35)
    assert cfg.clip_bounds(CAMPAIGN, story=True) == (18, 45)
    strict = Campaign(id="x", name="x", rules=Rules(min_sec=20, max_sec=40))
    assert cfg.clip_bounds(strict, story=True) == (20, 40)  # the campaign always wins


def test_build_prompt_fills_every_placeholder():
    cfg = Config(campaigns=[CAMPAIGN])
    prompt = build_prompt(cfg, CAMPAIGN, "[00:00] Hello there.")
    assert "{{" not in prompt
    assert "Pick 5 clips" in prompt and "18–35 seconds long, and preferably 20–30 seconds" in prompt
    assert "may run up to 45 seconds" in prompt and '"story": false' in prompt
    assert "most gripping sentence" in prompt and "payoff line" in prompt
    assert "Look for: Funny moments" in prompt and "Must feature: TJR speaks" in prompt
    assert "portray the creator or brand negatively" in prompt
    assert prompt.rstrip().endswith("[00:00] Hello there.")


def test_hook_rules_are_in_the_pick_prompt():
    prompt = build_prompt(Config(campaigns=[CAMPAIGN]), CAMPAIGN, "[00:00] Hi.")
    assert '5–9 words in a "setup: punch" shape' in prompt
    assert "accent_word" in prompt and "context_line" in prompt
    assert '"TJR trades sick: up at 5:30, plunge at 8" (accent_word "5:30")' in prompt
    assert "ALL CAPS (they are highlighted)" not in prompt


@pytest.mark.parametrize(
    ("hook", "problems"),
    [
        ("TJR's $10K challenge: $500 on day one", []),
        ("I just want it more", ['no "setup: punch" colon']),
        ("TJR: NEVER again", ["3 words instead of 5-9", "ALL CAPS words: NEVER"]),
        (
            "TJR's day one: down $56,000 on NQ!",
            ["no exclamation marks, hashtags or quotation marks"],
        ),
    ],
)
def test_hook_problems(hook, problems):
    assert hook_problems(hook) == problems


def test_accent_and_context_checks():
    hook = "TJR's $10K challenge: $500 on day one"
    assert hook_problems(hook, "$500", "TJR · $10K live trading challenge") == []
    assert hook_problems(hook, "$600", "TJR") == [
        "accent_word must be copied exactly from hook_text"
    ]
    assert hook_problems(hook, "day one", "one two three four five six seven") == [
        "context_line must be 1-6 words"
    ]


def test_picked_clips_keep_accent_and_context():
    pick = PickedClip(start="00:00", end="00:05", hook_text="Setup: punch here now", title="t",
                      score=0.9, accent_word=" punch ", context_line="TJR ·  challenge")  # fmt: skip
    clip = resolve_clips([pick], WORDS, "vid", CAMPAIGN, 2, 20)[0]
    assert (clip.accent_word, clip.context_line) == ("punch", "TJR · challenge")


def test_regenerate_headline_uses_the_clip_and_retries_a_bad_reply(monkeypatch, tmp_path):
    cfg = Config(machine=Machine(data_dir=tmp_path, min_free_gb=0), campaigns=[CAMPAIGN])
    conn = db.connect(tmp_path)
    clip = Clip("v_01", "v", 2.4, 7.5, "I just want it more", "Title", "Title #TJR", "r", 0.9)
    replies = iter([
        '{"hook_text": "TJR WANTS IT MORE", "accent_word": "more", "context_line": "TJR"}',
        (
            '{"hook_text": "\\"TJR trades sick: up at 5:30, plunge at 8.\\"", "accent_word": "5:30",'
            ' "context_line": "TJR · a $10M trader\'s day"}'
        ),
    ])  # fmt: skip
    prompts = []

    def fake_claude(cfg, prompt, cwd):
        prompts.append(prompt)
        return LLMResult(next(replies), 0, "sonnet", 100, 10, 0.001)

    monkeypatch.setattr(pick_mod, "run_claude", fake_claude)
    head = regenerate_headline(cfg, conn, clip, CAMPAIGN, WORDS, "What a $10M Trader Does")
    assert head.hook_text == "TJR trades sick: up at 5:30, plunge at 8"  # quotes, stop stripped
    assert (head.accent_word, head.context_line) == ("5:30", "TJR · a $10M trader's day")
    assert "Six seven eight nine ten?" in prompts[0]  # only what is said in the clip
    assert "One two" not in prompts[0] and "What a $10M Trader" in prompts[0]
    assert "breaks these rules" in prompts[1] and "colon" in prompts[1]
    assert conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 2
    conn.close()
