import itertools

from clipper.render.captions import (
    HIGHLIGHT,
    WHITE,
    ass_time,
    build_ass,
    caption_events,
    chunk_words,
    display,
    fit_hook,
)
from clipper.timing import Word


def make_words(text: str, dur: float = 0.2, gap: float = 0.05) -> list[Word]:
    out, t = [], 0.0
    for tok in text.split():
        out.append(Word(tok, round(t, 3), round(t + dur, 3)))
        t += dur + gap
    return out


def texts(chunks: list[list[Word]]) -> list[str]:
    return [" ".join(w.w for w in c) for c in chunks]


def test_display_cleans_words():
    assert display("Hello,") == "HELLO"
    assert display("what?") == "WHAT?"
    assert display('"quote."') == "QUOTE"
    assert display("{\\b1}x") == "B1X"


def test_chunks_respect_max_words():
    assert texts(chunk_words(make_words("a b c d e f g h i j"))) == ["a b c d", "e f g h", "i j"]


def test_chunks_respect_max_chars():
    chunks = chunk_words(make_words("extraordinary circumstances happen so it goes"))
    assert texts(chunks) == ["extraordinary", "circumstances", "happen so it goes"]
    for c in chunks[2:]:
        assert len(" ".join(display(w.w) for w in c)) <= 18


def test_chunks_break_at_sentence_end():
    assert texts(chunk_words(make_words("Hello there. Next one"))) == ["Hello there.", "Next one"]


def test_chunks_break_at_pause():
    words = [Word("a", 0.0, 0.2), Word("b", 1.0, 1.2), Word("c", 1.25, 1.4)]
    assert texts(chunk_words(words)) == ["a", "b c"]


def test_chunks_keep_every_word_in_order():
    words = make_words("So first and foremost, is changing your environment. It matters a lot!")
    assert [w for c in chunk_words(words) for w in c] == words


def test_chunks_skip_words_with_nothing_to_show():
    words = [Word("a", 0.0, 0.2), Word(",", 0.2, 0.25), Word("b", 0.25, 0.4)]
    assert texts(chunk_words(words)) == ["a b"]


def test_caption_events_highlight_each_word_in_turn():
    hl = f"{{\\c{HIGHLIGHT}}}%s{{\\c{WHITE}}}"
    chunks = [[Word("a", 0.0, 0.2), Word("b", 0.25, 0.45)], [Word("c.", 0.5, 0.7)]]
    assert caption_events(chunks) == [
        (0.0, 0.25, f"{hl % 'A'} B"),
        (0.25, 0.5, f"A {hl % 'B'}"),  # lingers only until the next chunk starts
        (0.5, 0.7 + 0.2, hl % "C"),
    ]


def test_caption_events_never_overlap():
    events = caption_events(chunk_words(make_words("one two three four five six. Seven eight")))
    for (_, end, _), (start, _, _) in itertools.pairwise(events):
        assert end <= start + 1e-9


def test_ass_time():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(75.239) == "0:01:15.24"
    assert ass_time(3723.5) == "1:02:03.50"


ASS_ARGS = {
    "width": 1080,
    "height": 1920,
    "font": "Arial",
    "font_size": 84,
    "y_frac": 0.66,
    "hook_sec": 2.5,
    "hook_size": 150,
    "hook_y_frac": 0.175,
}


def test_fit_hook_short_text_is_one_line_at_max_size():
    assert fit_hook("Red flag.", 960, 150) == (["Red flag."], 150)


def test_fit_hook_splits_into_two_balanced_lines():
    lines, size = fit_hook("I moved and noticed something", 960, 150)
    assert lines == ["I moved and", "noticed something"]
    assert size == int(960 / (0.6 * 17)) == 94


def test_fit_hook_never_more_than_two_lines_and_has_a_floor():
    lines, size = fit_hook("word " * 30, 960, 150)
    assert len(lines) == 2 and size == 56
    assert fit_hook("", 960, 150) == ([], 150)


def test_build_ass_has_hook_and_positioned_captions():
    ass = build_ass(make_words("Hello there."), "Big {hook}\ntext", **ASS_ARGS)
    assert "PlayResX: 1080" in ass and "Style: Caption,Arial,84," in ass
    # white text, black outline, sized to fit; centred at 17.5% of the height
    assert "Style: Hook,Arial,150,&H00FFFFFF,&H00FFFFFF,&H00000000," in ass
    assert (
        "Dialogue: 0,0:00:00.00,0:00:02.50,Hook,,0,0,0,,"
        "{\\q2\\pos(540,336)\\fad(0,200)}Big hook\\Ntext\n"
    ) in ass
    assert ass.count("Dialogue: 1,") == 2
    assert "{\\pos(540,1267)}" in ass


def test_build_ass_long_hook_wraps_to_two_lines():
    ass = build_ass([], "Rich people asking YOU for help? Red flag.", **ASS_ARGS)
    assert "}Rich people asking YOU\\Nfor help? Red flag.\n" in ass


def test_build_ass_without_hook():
    assert "Dialogue" not in build_ass([], "", **ASS_ARGS)
