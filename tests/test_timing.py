import pytest

from clipper.timing import (
    Word,
    compact_transcript,
    fmt_ts,
    parse_ts,
    sentences,
    snap_clip,
    words_between,
    words_from_whisper,
)


def make_words(text: str, start: float = 0.0, dur: float = 0.4, gap: float = 0.1) -> list[Word]:
    """Evenly spaced words: each lasts `dur`, followed by `gap` of silence."""
    out, t = [], start
    for tok in text.split():
        out.append(Word(tok, round(t, 3), round(t + dur, 3)))
        t += dur + gap
    return out


# 4 sentences x 5 words, 0.5 s per word slot -> sentences start at 0, 2.5, 5.0, 7.5
WORDS = make_words(
    "One two three four five. Six seven eight nine ten? "
    "Eleven twelve thirteen fourteen fifteen! Sixteen seventeen eighteen nineteen twenty."
)


@pytest.mark.parametrize(
    ("ts", "sec"),
    [("01:05", 65), ("75:30", 4530), ("1:02:03", 3723), ("[00:09]", 9), (12.5, 12.5), ("42", 42)],
)
def test_parse_ts(ts, sec):
    assert parse_ts(ts) == sec


def test_parse_ts_rejects_garbage():
    with pytest.raises(ValueError):
        parse_ts("1:2:3:4")


def test_fmt_ts_roundtrip_and_long_videos():
    assert fmt_ts(65.9) == "01:05"
    assert fmt_ts(4530) == "75:30"
    assert parse_ts(fmt_ts(3599)) == 3599


def test_words_from_whisper_merges_pieces_and_drops_non_speech():
    segs = [
        {"text": " I", "offsets": {"from": 0, "to": 200}},
        {"text": " don", "offsets": {"from": 200, "to": 400}},
        {"text": "'t", "offsets": {"from": 400, "to": 500}},
        {"text": " [MUSIC]", "offsets": {"from": 500, "to": 900}},
        {"text": "", "offsets": {"from": 900, "to": 900}},
        {"text": " know.", "offsets": {"from": 900, "to": 1300}},
    ]
    assert words_from_whisper(segs) == [
        Word("I", 0.0, 0.2),
        Word("don't", 0.2, 0.5),
        Word("know.", 0.9, 1.3),
    ]


def test_sentences_split_on_punctuation():
    assert sentences(WORDS) == [(0, 5), (5, 10), (10, 15), (15, 20)]


def test_sentences_cap_without_punctuation():
    words = make_words(" ".join(["word"] * 90))
    assert sentences(words) == [(0, 40), (40, 80), (80, 90)]


def test_sentences_cap_prefers_last_comma():
    toks = ["word"] * 50
    toks[29] = "word,"
    assert sentences(make_words(" ".join(toks)))[0] == (0, 30)


def test_compact_transcript_lines():
    lines = compact_transcript(make_words("Hi there. Bye now.", start=59.0)).splitlines()
    assert lines == ["[00:59] Hi there.", "[01:00] Bye now."]


def test_snap_clip_snaps_to_sentence_boundaries():
    # rough 2-7 s -> sentences 2 and 3 (2.5 .. 7.4), padded into the surrounding silence
    s, e = snap_clip(WORDS, 2.0, 7.0, min_sec=2, max_sec=20)
    assert s == pytest.approx(2.5 - 0.1)  # gap before "Six" is 0.1 s
    assert e == pytest.approx(7.4 + 0.1)  # gap after "fifteen!" is only 0.1 s
    inside = words_between(WORDS, s, e)
    assert inside[0].w == "Six" and inside[-1].w == "fifteen!"


def test_snap_clip_never_cuts_mid_word():
    for start, end in [(0.3, 4.2), (2.7, 9.9), (4.9, 7.6)]:
        s, e = snap_clip(WORDS, start, end, min_sec=1, max_sec=20)
        for w in WORDS:
            assert not (w.s < s < w.e), f"start {s} cuts {w}"
            assert not (w.s < e < w.e), f"end {e} cuts {w}"


def test_snap_clip_trims_whole_sentences_to_max():
    s, e = snap_clip(WORDS, 0.0, 10.0, min_sec=2, max_sec=5.5)
    assert e - s <= 5.5 + 0.4
    assert words_between(WORDS, s, e)[-1].w == "ten?"


def test_snap_clip_extends_to_min():
    s, e = snap_clip(WORDS, 2.5, 5.0, min_sec=4, max_sec=8)
    assert words_between(WORDS, s, e)[-1].w == "fifteen!"


def test_snap_clip_returns_none_when_impossible():
    assert snap_clip(WORDS, 0.0, 2.5, min_sec=1, max_sec=1.5) is None  # one sentence > max
    assert snap_clip(WORDS, 7.5, 10.0, min_sec=5, max_sec=20) is None  # nothing left to extend
    assert snap_clip([], 0, 10, 1, 20) is None


def test_snap_clip_at_transcript_edges():
    s, e = snap_clip(WORDS, 0.0, 99.0, min_sec=1, max_sec=60)
    assert s == 0.0
    assert e == pytest.approx(WORDS[-1].e + 0.3)


def test_words_between_rebases_to_clip_start():
    out = words_between(WORDS, 2.4, 4.95)
    assert [w.w for w in out] == ["Six", "seven", "eight", "nine", "ten?"]
    assert out[0].s == pytest.approx(0.1)
