"""Word-highlight captions + hook text as an ASS subtitle file (pure functions)."""

from __future__ import annotations

from clipper.timing import SENTENCE_END, Word

HIGHLIGHT = "&H00FFFF&"  # yellow; ASS colours are BGR
WHITE = "&HFFFFFF&"

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, \
Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, \
Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,\
-1,0,0,0,100,100,0,0,1,6,2,5,60,60,0,1
Style: Hook,{font},{hook_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,\
-1,0,0,0,100,100,0,0,1,{hook_outline},4,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def display(word: str) -> str:
    """Caption form of a word: upper case, no soft punctuation, no ASS control chars."""
    text = word.upper().strip('"“”').rstrip(",.;:")
    return text.replace("{", "").replace("}", "").replace("\\", "")


def chunk_words(
    words: list[Word], max_words: int = 4, max_chars: int = 18, max_gap: float = 0.6
) -> list[list[Word]]:
    """Group words into caption chunks that fit one line. A chunk closes when it is full
    (words or characters), at a pause longer than `max_gap`, or at a sentence end."""
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if not display(w.w):
            continue
        if cur:
            prev = cur[-1]
            length = sum(len(display(x.w)) for x in cur) + len(cur) + len(display(w.w))
            if (
                len(cur) >= max_words
                or length > max_chars
                or w.s - prev.e > max_gap
                or prev.w.endswith(SENTENCE_END)
            ):
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def caption_events(chunks: list[list[Word]], linger: float = 0.2) -> list[tuple[float, float, str]]:
    """One (start, end, text) event per word: the whole chunk with that word highlighted.
    A chunk stays up `linger` s after its last word, but never into the next chunk."""
    events: list[tuple[float, float, str]] = []
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0].s if ci + 1 < len(chunks) else float("inf")
        chunk_end = max(chunk[-1].s, min(chunk[-1].e + linger, next_start))
        for wi, w in enumerate(chunk):
            end = chunk[wi + 1].s if wi + 1 < len(chunk) else chunk_end
            if end - w.s < 0.01:
                continue
            text = " ".join(
                f"{{\\c{HIGHLIGHT}}}{display(x.w)}{{\\c{WHITE}}}" if k == wi else display(x.w)
                for k, x in enumerate(chunk)
            )
            events.append((w.s, end, text))
    return events


def ass_time(sec: float) -> str:
    cs = max(0, round(sec * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


HOOK_CHAR_EM = 0.6  # average glyph width of a bold sans font in ems (errs on the wide side)
HOOK_MIN_SIZE = 56
HOOK_SIDE_MARGIN = 60


def fit_hook(text: str, max_width: int, max_size: int) -> tuple[list[str], int]:
    """Lay the hook out on at most 2 lines, as big as fits `max_width` (up to `max_size`).
    One line if that already reaches `max_size`, else the most balanced 2-line split."""
    words = text.split()
    if not words:
        return [], max_size

    def size_for(lines: list[str]) -> int:
        return int(max_width / (HOOK_CHAR_EM * max(len(line) for line in lines)))

    best = [" ".join(words)]
    if size_for(best) < max_size:
        for k in range(1, len(words)):
            cand = [" ".join(words[:k]), " ".join(words[k:])]
            if size_for(cand) > size_for(best):
                best = cand
    return best, max(HOOK_MIN_SIZE, min(max_size, size_for(best)))


def build_ass(
    words: list[Word],
    hook_text: str,
    *,
    width: int,
    height: int,
    font: str,
    font_size: int,
    y_frac: float,
    hook_sec: float,
    hook_size: int,
    hook_y_frac: float,
) -> str:
    """Full ASS file. `words` must be relative to the clip start (t=0). `hook_size` is the
    largest hook font size; long hooks shrink to stay on 2 lines."""
    hook = hook_text.replace("{", "").replace("}", "").replace("\\", "")
    hook_lines, size = fit_hook(hook, width - 2 * HOOK_SIDE_MARGIN, hook_size)
    lines = [
        HEADER.format(
            width=width,
            height=height,
            font=font,
            size=font_size,
            hook_size=size,
            hook_outline=max(6, round(size * 0.08)),
        )
    ]
    if hook_lines and hook_sec > 0:
        # \q2: no automatic wrapping, so the hook can never spill onto a third line
        tags = f"{{\\q2\\pos({width // 2},{round(height * hook_y_frac)})\\fad(0,200)}}"
        text = "\\N".join(hook_lines)
        lines.append(f"Dialogue: 0,{ass_time(0)},{ass_time(hook_sec)},Hook,,0,0,0,,{tags}{text}\n")
    pos = f"{{\\pos({width // 2},{round(height * y_frac)})}}"
    for start, end, text in caption_events(chunk_words(words)):
        lines.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Caption,,0,0,0,,{pos}{text}\n")
    return "".join(lines)
