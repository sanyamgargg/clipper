import itertools

import pytest

from clipper.config import REPO_ROOT
from clipper.render.captions import (
    HOOK_FACE_GAP,
    YELLOW,
    CaptionStyle,
    FontMetrics,
    apply_fixes,
    ass_time,
    build_ass,
    caption_events,
    caption_words,
    caption_ys,
    chunk_fit,
    chunk_words,
    display,
    hook_markup,
    is_caps_word,
    layout_hook,
    layout_titlecard,
    rounded_rect,
)
from clipper.timing import Word

FONTS = REPO_ROOT / "assets" / "fonts"
# synthetic font: every glyph 0.5 em wide, cell = 1 em, caps 0.7 em
SIMPLE = FontMetrics(upm=1000, cell=1000, ascent=800, cap=700, descent=200, default_advance=500)


@pytest.fixture(scope="module")
def montserrat() -> FontMetrics:
    metrics = FontMetrics.find(FONTS, "Montserrat Black")
    assert metrics is not None, "bundled font missing from assets/fonts"
    return metrics


def make_words(text: str, dur: float = 0.2, gap: float = 0.05) -> list[Word]:
    out, t = [], 0.0
    for tok in text.split():
        out.append(Word(tok, round(t, 3), round(t + dur, 3)))
        t += dur + gap
    return out


def texts(chunks: list[list[Word]]) -> list[str]:
    return [" ".join(w.w for w in c) for c in chunks]


# --- caption words: fillers and fixes ---------------------------------------------


def test_fillers_are_dropped_from_captions_only():
    words = make_words("So um, I uh... think Uhm. umbrella uhh")
    assert [w.w for w in caption_words(words)] == ["So", "I", "think", "umbrella"]
    assert len(words) == 8  # the transcript itself is untouched


def test_fixes_replace_phrases_case_and_punctuation_insensitively():
    words = make_words("met In Santos? and incentos, in Vegas")
    fixed = apply_fixes(words, {"in santos": "Insentos", "incentos": "Insentos"})
    assert [w.w for w in fixed] == ["met", "Insentos?", "and", "Insentos,", "in", "Vegas"]
    phrase = fixed[1]
    assert (phrase.s, phrase.e) == (words[1].s, words[2].e)  # spans both words


def test_fixes_prefer_the_longest_match():
    words = make_words("t j r said tj")
    fixed = apply_fixes(words, {"tj": "TJ", "t j r": "TJR"})
    assert [w.w for w in fixed] == ["TJR", "said", "TJ"]


def test_fixes_without_rules_return_the_words():
    words = make_words("a b")
    assert apply_fixes(words, {}) == words
    assert apply_fixes(words, {"": "x"}) == words


# --- chunks ------------------------------------------------------------------------


def test_display_cleans_words():
    assert display("Hello,") == "HELLO"
    assert display("what?") == "WHAT?"
    assert display('"quote."') == "QUOTE"
    assert display("{\\b1}x") == "B1X"


def test_chunks_respect_max_three_words():
    assert texts(chunk_words(make_words("a b c d e f g"))) == ["a b c", "d e f", "g"]


def test_chunks_respect_max_14_chars():
    chunks = chunk_words(make_words("everything happens fast now"))
    assert texts(chunks) == ["everything", "happens fast", "now"]
    for c in chunks:
        assert len(" ".join(display(w.w) for w in c)) <= 14


def test_chunks_break_at_sentence_end_and_pause():
    assert texts(chunk_words(make_words("Hello there. Next one"))) == ["Hello there.", "Next one"]
    words = [Word("a", 0.0, 0.2), Word("b", 1.0, 1.2), Word("c", 1.25, 1.4)]
    assert texts(chunk_words(words)) == ["a", "b c"]


def test_chunks_keep_every_word_in_order():
    words = make_words("So first and foremost, is changing your environment. It matters a lot!")
    assert [w for c in chunk_words(words) for w in c] == words


def test_chunks_skip_words_with_nothing_to_show():
    words = [Word("a", 0.0, 0.2), Word(",", 0.2, 0.25), Word("b", 0.25, 0.4)]
    assert texts(chunk_words(words)) == ["a b"]


# --- caption events -----------------------------------------------------------------


def test_events_highlight_each_word_and_pop_only_when_the_chunk_appears():
    chunks = [[Word("a", 0.0, 0.2), Word("b", 0.25, 0.45)], [Word("c.", 0.5, 0.7)]]
    pop = "\\fscx85\\fscy85\\t(0,80,\\fscx100\\fscy100)"
    pop_active = "\\fscx95\\fscy95\\t(0,80,\\fscx112\\fscy112)"
    still = "\\fscx100\\fscy100"
    still_active = "\\fscx112\\fscy112"
    assert caption_events(chunks) == [
        (0.0, 0.25, f"{{{pop}}}{{\\c{YELLOW}{pop_active}}}A{{\\c&HFFFFFF&{pop}}} B"),
        (0.25, 0.5, f"{{{still}}}A {{\\c{YELLOW}{still_active}}}B{{\\c&HFFFFFF&{still}}}"),
        (0.5, 0.7 + 0.2, f"{{{pop}}}{{\\c{YELLOW}{pop_active}}}C{{\\c&HFFFFFF&{pop}}}"),
    ]


def test_events_never_overlap():
    events = caption_events(chunk_words(make_words("one two three four five six. Seven eight")))
    for (_, end, _), (start, _, _) in itertools.pairwise(events):
        assert end <= start + 1e-9


def test_chunk_fit_shrinks_only_chunks_that_would_overflow(montserrat):
    normal = make_words("I just want")
    long = make_words("ENTREPRENEURSHIP.")
    assert chunk_fit(normal, montserrat, 90, 1000) == 1.0
    fit = chunk_fit(long, montserrat, 90, 1000)
    assert 0.8 < fit < 1.0
    width = montserrat.width("ENTREPRENEURSHIP", 90, 1) * 1.12 * fit
    assert width == pytest.approx(1000, abs=1)
    event = caption_events([long], montserrat, 90, 1000)[0][2]
    assert f"\\fscx{round(100 * fit)}" in event and "\\fscx100" not in event


def chunks_at(*starts: float, dur: float = 0.8) -> list[list[Word]]:
    """One single-word chunk per start time."""
    return [[Word("w", s, s + dur)] for s in starts]


YS = {"default_y": 1267, "gap": 77, "max_y": 1498, "sample_fps": 5}


def test_caption_y_default_without_a_face():
    assert caption_ys(chunks_at(0, 1, 2), None, **YS) == [1267, 1267, 1267]
    assert caption_ys(chunks_at(0, 1), [None] * 10, **YS) == [1267, 1267]


def test_caption_y_moves_below_a_low_face_and_is_clamped():
    assert caption_ys(chunks_at(0), [1000] * 5, **YS) == [1267]  # face high: default
    assert caption_ys(chunks_at(0), [1300] * 5, **YS) == [1377]  # bottom + 4%
    assert caption_ys(chunks_at(0), [1700] * 5, **YS) == [1498]  # never below 78%


def test_caption_y_uses_the_lowest_face_bottom_during_the_chunk_and_holds_gaps():
    bottoms = [1200, 1200, 1350, None, None, None, None, None, None, None]
    # chunk 0 spans samples 0..4: the 1350 sample counts; chunk 1 holds the last bottom
    assert caption_ys(chunks_at(0, 1.2), bottoms, **YS) == [1427, 1427]


def test_caption_y_is_smoothed_between_chunks():
    smooth = {**YS, "deadband": 38, "step": 58}
    bottoms = [1300] * 5 + [1320] * 5 + [1000] * 15
    ys = caption_ys(chunks_at(0, 1, 2, 3, 4), bottoms, **smooth)
    assert ys[0] == 1377
    assert ys[1] == 1377  # +20 px is inside the deadband: no jump
    assert ys[2:] == [1319, 1267, 1267]  # face moves up: ease back 58 px per chunk
    down = caption_ys(chunks_at(0, 1), [1000] * 5 + [1400] * 5, **smooth)
    assert down == [1267, 1477]  # moving down to clear the face is immediate


def test_ass_time():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(75.239) == "0:01:15.24"
    assert ass_time(3723.5) == "1:02:03.50"


# --- font metrics -------------------------------------------------------------------


def test_bundled_font_is_found_by_any_of_its_names(montserrat):
    assert FontMetrics.find(FONTS, "montserrat black") == montserrat
    assert FontMetrics.find(FONTS, "Montserrat-Black") == montserrat  # PostScript name
    assert FontMetrics.find(FONTS, "Comic Sans") is None
    assert FontMetrics.find(FONTS / "missing", "Montserrat Black") is None


def test_ass_size_converts_em_pixels_to_libass_cell_size(montserrat):
    # libass makes winAscent + winDescent (1562 units) equal Fontsize
    assert (montserrat.upm, montserrat.cell) == (1000, 1562)
    assert montserrat.ass_size(90) == 141
    assert SIMPLE.ass_size(90) == 90
    assert SIMPLE.width("abcd", 100) == 200
    assert SIMPLE.width("abcd", 100, spacing=1) == 204


# --- hook ---------------------------------------------------------------------------

H = 1920
HOOK = {
    "width": 1080,
    "height": H,
    "max_px": 150,
    "min_px": 90,
    "fallback_top_frac": 0.09,
    "top_limit_frac": 0.05,
}


def test_hook_without_face_starts_at_fallback_and_is_max_size():
    hook = layout_hook("Red flag", SIMPLE, "outline", face_top=None, **HOOK)
    assert (hook.lines, hook.px) == (["Red flag"], 150)  # 8 chars x 0.5 em x 150 = 600 px
    assert hook.top == round(0.09 * H)


def test_hook_splits_long_text_into_two_balanced_lines_that_fit_the_width():
    hook = layout_hook("I moved and noticed something", SIMPLE, "outline", face_top=None, **HOOK)
    assert hook.lines == ["I moved and", "noticed something"]
    usable = 1080 - 2 * 40
    assert 17 * 0.5 * hook.px + 2 * 0.10 * hook.px <= usable  # widest line + outline
    assert hook.px == int(usable / (17 * 0.5 + 0.2)) == 114


def test_hook_never_exceeds_two_lines():
    hook = layout_hook("word " * 30, SIMPLE, "outline", face_top=None, **HOOK)
    assert len(hook.lines) == 2


@pytest.mark.parametrize("style", ["outline", "box"])
def test_hook_sits_above_the_face(style):
    face_top = 900
    hook = layout_hook("Why do they need YOU", SIMPLE, style, face_top=face_top, **HOOK)
    assert hook.bottom == pytest.approx(face_top - HOOK_FACE_GAP * H, abs=1)
    assert hook.top >= 0.05 * H


def test_hook_shrinks_to_fit_above_a_high_face_but_not_below_min():
    roomy = layout_hook("I just want it more", SIMPLE, "box", face_top=1000, **HOOK)
    tight = layout_hook("I just want it more", SIMPLE, "box", face_top=420, **HOOK)
    assert tight.px < roomy.px
    assert tight.px >= 90
    assert tight.bottom <= 420 - HOOK_FACE_GAP * H + 1
    cramped = layout_hook("I just want it more", SIMPLE, "box", face_top=150, **HOOK)
    assert cramped.px == 90  # floor: rather overlap the head than shrink further
    assert cramped.top == round(0.05 * H)


def test_box_hook_reserves_its_padding():
    outline = layout_hook("Red flag", SIMPLE, "outline", face_top=None, **HOOK)
    box = layout_hook("Red flag", SIMPLE, "box", face_top=None, **HOOK)
    assert box.bottom - box.top > outline.bottom - outline.top


def test_caps_words_are_yellow_in_outline_style_only():
    assert [is_caps_word(w) for w in ["YOU", "NQ", "I", "$1M", "PnL", "MORE?"]] == [
        True, True, False, False, False, True,
    ]  # fmt: skip
    assert hook_markup(["Why do they", "need YOU?"], "outline") == (
        f"Why do they\\Nneed {{\\c{YELLOW}}}YOU?{{\\c&HFFFFFF&}}"
    )
    assert hook_markup(["Why do they", "need YOU?"], "box") == "Why do they\\Nneed YOU?"


# --- full ASS file --------------------------------------------------------------------


def test_build_ass_caption_style(montserrat):
    ass = build_ass(make_words("Hello there."), "", CaptionStyle(), montserrat)
    assert "PlayResX: 1080" in ass
    # Montserrat Black, 90 px em = 141; outline 10, shadow 3 at 50% black, spacing 1
    assert (
        "Style: Caption,Montserrat Black,141,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,1,0,1,10,3,5,40,40,0,1"
    ) in ass
    assert ass.count("Dialogue: 1,") == 2
    assert ass.count("{\\an5\\pos(540,1267)\\fscx") == 2  # default 66%, no face known
    assert "Dialogue: 0," not in ass  # no hook


def test_build_ass_outline_hook(montserrat):
    ass = build_ass([], "Why do they need YOU", CaptionStyle(hook_style="outline"), montserrat)
    style = next(line for line in ass.splitlines() if line.startswith("Style: Hook,"))
    assert ",&H00FFFFFF,&H00FFFFFF,&H00000000," in style  # white text, black outline
    assert ",-1,0,0,0,100,100,0,0,1," in style  # bold, BorderStyle 1
    dialogue = next(line for line in ass.splitlines() if line.startswith("Dialogue: 0,"))
    assert dialogue.startswith("Dialogue: 0,0:00:00.00,0:00:03.00,Hook,")  # shown 3 s
    assert "\\an5\\pos(540," in dialogue and "\\q2" in dialogue
    assert "\\fscx85\\fscy85\\t(0,120,\\fscx100\\fscy100)\\fad(0,200)" in dialogue
    assert f"{{\\c{YELLOW}}}YOU" in dialogue


def test_build_ass_box_hook(montserrat):
    ass = build_ass([], "Why do they need YOU", CaptionStyle(hook_style="box"), montserrat)
    style = next(line for line in ass.splitlines() if line.startswith("Style: Hook,"))
    assert ",&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF," in style  # black on white box
    assert ",0,0,3," in style  # BorderStyle 3: opaque box
    assert YELLOW not in ass


def test_build_ass_places_hook_above_the_face(montserrat):
    style = CaptionStyle(hook_style="outline")
    no_face = build_ass([], "Red flag", style, montserrat)
    face = build_ass([], "Red flag", style, montserrat, face_top=1000)

    def hook_y(ass: str) -> int:
        dialogue = next(line for line in ass.splitlines() if line.startswith("Dialogue: 0,"))
        return int(dialogue.split("\\pos(540,", 1)[1].split(")", 1)[0])

    assert hook_y(no_face) < hook_y(face) < 1000


def test_build_ass_moves_captions_below_the_face(montserrat):
    words = make_words("one two three four five six seven")  # 3 chunks over ~1.75 s
    ass = build_ass(words, "", CaptionStyle(), montserrat, face_bottoms=[1400.0] * 10)
    assert ass.count("\\an5\\pos(540,1477)") == 7  # 1400 + 4% of 1920, every word event


# --- title card -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def semibold() -> FontMetrics:
    metrics = FontMetrics.find(FONTS, "Montserrat SemiBold")
    assert metrics is not None, "bundled font missing from assets/fonts"
    return metrics


CARD = {"width": 1080, "height": 1920, "max_px": 64, "top_frac": 0.08}


def test_titlecard_sits_at_8_percent_and_is_64px(semibold):
    card = layout_titlecard("TJR's 5:30 AM routine as a $10M trader", semibold, **CARD)
    assert card.px == 64 and card.top == round(0.08 * 1920)
    assert 1 <= len(card.lines) <= 2
    assert card.left >= 60 and card.right <= 1080 - 60  # inside the margins
    assert card.left + card.right == 1080  # centred
    assert all(card.top < y < card.bottom for y in card.line_ys)
    if len(card.lines) == 2:
        assert card.line_ys[1] - card.line_ys[0] == round(1.25 * 64)


def test_titlecard_prefers_one_line_and_never_exceeds_two(semibold):
    short = layout_titlecard("Why TJR left college", semibold, **CARD)
    assert short.lines == ["Why TJR left college"] and short.px == 64
    long = layout_titlecard("word " * 30, semibold, **CARD)
    assert len(long.lines) == 2 and long.px < 64  # shrinks only to fit the width
    widest = max(semibold.width(line, long.px) for line in long.lines)
    assert widest + 2 * 0.55 * long.px <= 1080 - 2 * 60 + 1
    assert layout_titlecard("", semibold, **CARD) is None


def test_rounded_rect_path_closes_at_its_start():
    path = rounded_rect(400, 100, 20)
    assert path.startswith("m 20 0 l 380 0 b ")
    assert path.endswith(" 20 0")
    assert rounded_rect(40, 10, 99).startswith("m 5 0")  # radius limited to half the height


def test_build_ass_titlecard(montserrat, semibold):
    ass = build_ass(
        make_words("Hello there."),
        "tJR's 5:30 AM routine as a $10M trader",
        CaptionStyle(),  # titlecard is the default
        montserrat,
        hook_metrics=semibold,
        duration=27.5,
    )
    styles = {line.split(",")[0]: line for line in ass.splitlines() if line.startswith("Style:")}
    hook = styles["Style: Hook"]
    assert hook.startswith("Style: Hook,Montserrat SemiBold,100,&H00FFFFFF,")  # 64 px em
    assert ",0,0,0,0,100,100,0,0,1,0,0,5," in hook  # not bold, no outline, no shadow
    assert styles["Style: Bar"].startswith("Style: Bar,Montserrat SemiBold,20,&H4D000000,")
    bar = next(line for line in ass.splitlines() if ",Bar,," in line)
    assert bar.startswith("Dialogue: 2,0:00:00.00,0:00:27.50,Bar,")  # the whole clip
    assert "\\an7\\pos(" in bar and "\\fad(150,0)\\p1}m " in bar
    texts = [line for line in ass.splitlines() if ",Hook,," in line]
    assert 1 <= len(texts) <= 2
    assert all("\\fad(150,0)" in t and "\\fscx" not in t and YELLOW not in t for t in texts)
    assert texts[0].split("}", 1)[1].startswith("TJR's")  # sentence case
