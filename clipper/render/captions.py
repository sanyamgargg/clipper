"""Word-highlight captions + hook text as an ASS subtitle file (pure functions).

Sizes are em heights in output pixels. libass sizes a font so that its whole line cell
(OS/2 winAscent + winDescent) equals the ASS Fontsize, which for wide-cell fonts such as
Montserrat draws glyphs far smaller than the number suggests; FontMetrics converts.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from clipper.timing import SENTENCE_END, Word

log = logging.getLogger(__name__)

YELLOW = "&H00FFFF&"  # ASS colours are BGR
WHITE = "&HFFFFFF&"

# captions (pixels at 1080 wide)
CAPTION_OUTLINE = 10
CAPTION_SHADOW = 3
CAPTION_SPACING = 1
CAPTION_MARGIN = 40  # left/right
ACTIVE_SCALE = 112  # % for the word being spoken
POP_FROM = 0.85  # pop-in starts at this scale
CAPTION_POP_MS = 80

# hook
HookStyle = Literal["box", "outline"]
HOOK_POP_MS = 120
HOOK_FADE_MS = 200
HOOK_OUTLINE_EM = 0.10  # outline style: outline thickness
HOOK_SHADOW_EM = 0.03
HOOK_BOX_PAD_EM = 0.25  # box style: padding around the text
HOOK_FACE_GAP = 0.015  # gap between the hook and the face top, fraction of height
HOOK_MARGIN = 40  # left/right

FILLERS = frozenset({"um", "umm", "uh", "uhh", "uhm"})


# --- font metrics ----------------------------------------------------------------


@dataclass(frozen=True)
class FontMetrics:
    """The few numbers needed to size and place text like libass does (font units)."""

    upm: int  # units per em
    cell: int  # winAscent + winDescent: libass makes this the ASS Fontsize
    ascent: int  # winAscent: cell top to baseline
    cap: int  # capital height
    descent: int  # typical descender depth (positive)
    advances: dict[str, int] = field(default_factory=dict)
    default_advance: int = 600

    def ass_size(self, px: float) -> int:
        """ASS Fontsize that draws an em of `px` pixels."""
        return round(px * self.cell / self.upm)

    def width(self, text: str, px: float, spacing: float = 0.0) -> float:
        units = sum(self.advances.get(ch, self.default_advance) for ch in text)
        return units * px / self.upm + spacing * len(text)

    @classmethod
    def load(cls, path: Path) -> FontMetrics:
        from fontTools.ttLib import TTFont

        font = TTFont(path, lazy=True)
        os2, hmtx = font["OS/2"], font["hmtx"]
        upm = font["head"].unitsPerEm
        cap = getattr(os2, "sCapHeight", 0) or round(0.7 * upm)
        return cls(
            upm=upm,
            cell=os2.usWinAscent + os2.usWinDescent,
            ascent=os2.usWinAscent,
            cap=cap,
            descent=abs(font["hhea"].descent),
            advances={chr(cp): hmtx[glyph][0] for cp, glyph in font.getBestCmap().items()},
            default_advance=round(0.6 * upm),
        )

    @classmethod
    def find(cls, fonts_dir: Path, family: str) -> FontMetrics | None:
        """Metrics of the font in `fonts_dir` named `family` (family, full or PostScript
        name, case-insensitive), or None."""
        from fontTools.ttLib import TTFont

        want = family.casefold()
        for path in sorted(fonts_dir.glob("*")) if fonts_dir.is_dir() else []:
            if path.suffix.lower() not in (".ttf", ".otf"):
                continue
            names = {
                n.toUnicode().casefold() for n in TTFont(path, lazy=True)["name"].names
                if n.nameID in (1, 4, 6)
            }  # fmt: skip
            if want in names:
                return cls.load(path)
        return None


# Rough metrics for a system font we have no file for (Arial-like).
ESTIMATED = FontMetrics(upm=1000, cell=1150, ascent=920, cap=716, descent=210)


# --- caption words -----------------------------------------------------------------


def _norm(token: str) -> str:
    return re.sub(r"[^0-9a-z$%&']+", "", token.lower())


def apply_fixes(words: list[Word], fixes: dict[str, str]) -> list[Word]:
    """Replace misheard words or phrases, matched case- and punctuation-insensitively
    ("in santos" -> "Insentos"). A phrase becomes one word spanning its timings and
    keeps the trailing punctuation of its last word."""
    rules = sorted(
        ((tuple(_norm(t) for t in key.split()), value) for key, value in fixes.items()),
        key=lambda rule: -len(rule[0]),
    )
    rules = [(key, value) for key, value in rules if key and all(key)]
    if not rules:
        return list(words)
    normed = [_norm(w.w) for w in words]
    out: list[Word] = []
    i = 0
    while i < len(words):
        for key, value in rules:
            n = len(key)
            if tuple(normed[i : i + n]) == key:
                last = words[i + n - 1]
                trail = re.search(r"[.,!?;:]*$", last.w).group()  # type: ignore[union-attr]
                out.append(Word(value + trail, words[i].s, last.e))
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return out


def caption_words(words: list[Word], fixes: dict[str, str] | None = None) -> list[Word]:
    """Words as they should appear in captions: fillers dropped, campaign fixes applied.
    (The transcript itself is left untouched.)"""
    return apply_fixes([w for w in words if _norm(w.w) not in FILLERS], fixes or {})


def display(word: str) -> str:
    """Caption form of a word: upper case, no soft punctuation, no ASS control chars."""
    text = word.upper().strip('"“”').rstrip(",.;:")
    return text.replace("{", "").replace("}", "").replace("\\", "")


def chunk_words(
    words: list[Word], max_words: int = 3, max_chars: int = 14, max_gap: float = 0.6
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


def _scale(pct: int, pop_ms: int) -> str:
    """Scale tags; with pop_ms, grow from POP_FROM x pct to pct over that time."""
    if not pop_ms:
        return f"\\fscx{pct}\\fscy{pct}"
    start = round(pct * POP_FROM)
    return f"\\fscx{start}\\fscy{start}\\t(0,{pop_ms},\\fscx{pct}\\fscy{pct})"


def chunk_fit(chunk: list[Word], metrics: FontMetrics | None, px: float, max_width: float) -> float:
    """Scale factor (<= 1) that keeps the chunk on screen even with its widest word
    enlarged to ACTIVE_SCALE."""
    if metrics is None:
        return 1.0
    texts = [display(w.w) for w in chunk]
    base = metrics.width(" ".join(texts), px, CAPTION_SPACING)
    grow = max(metrics.width(t, px, CAPTION_SPACING) for t in texts) * (ACTIVE_SCALE / 100 - 1)
    return min(1.0, max_width / (base + grow))


def caption_events(
    chunks: list[list[Word]],
    metrics: FontMetrics | None = None,
    px: float = 90,
    max_width: float = 1080 - 2 * CAPTION_MARGIN,
    linger: float = 0.2,
) -> list[tuple[float, float, str]]:
    """One (start, end, text) event per word: the whole chunk, that word yellow and
    enlarged. Each chunk pops in on its first event only. A chunk stays up `linger` s
    after its last word, but never into the next chunk."""
    events: list[tuple[float, float, str]] = []
    for ci, chunk in enumerate(chunks):
        next_start = chunks[ci + 1][0].s if ci + 1 < len(chunks) else float("inf")
        chunk_end = max(chunk[-1].s, min(chunk[-1].e + linger, next_start))
        fit = chunk_fit(chunk, metrics, px, max_width)
        base, active = round(100 * fit), round(ACTIVE_SCALE * fit)
        first = True
        for wi, w in enumerate(chunk):
            end = chunk[wi + 1].s if wi + 1 < len(chunk) else chunk_end
            if end - w.s < 0.01:
                continue
            pop = CAPTION_POP_MS if first else 0
            first = False
            parts = [
                f"{{\\c{YELLOW}{_scale(active, pop)}}}{display(x.w)}"
                f"{{\\c{WHITE}{_scale(base, pop)}}}"
                if k == wi
                else display(x.w)
                for k, x in enumerate(chunk)
            ]
            events.append((w.s, end, f"{{{_scale(base, pop)}}}" + " ".join(parts)))
    return events


# --- hook -------------------------------------------------------------------------


@dataclass(frozen=True)
class HookLayout:
    lines: list[str]
    px: int  # em height
    center_y: int  # \an5 anchor: centre of the line cells
    top: int  # visible top edge (box or outline)
    bottom: int  # visible bottom edge


def _extent(metrics: FontMetrics, n_lines: int, style: HookStyle) -> tuple[float, float]:
    """Visible (top, bottom) of an n-line hook, in ems, relative to its first cell top."""
    cell = metrics.cell / metrics.upm
    if style == "box":
        return -HOOK_BOX_PAD_EM, n_lines * cell + HOOK_BOX_PAD_EM
    top = (metrics.ascent - metrics.cap) / metrics.upm - HOOK_OUTLINE_EM
    bottom = (n_lines - 1) * cell + (metrics.ascent + metrics.descent) / metrics.upm
    return top, bottom + HOOK_OUTLINE_EM


def layout_hook(
    text: str,
    metrics: FontMetrics,
    style: HookStyle,
    *,
    width: int,
    height: int,
    max_px: float,
    min_px: float,
    face_top: float | None,
    fallback_top_frac: float,
    top_limit_frac: float,
) -> HookLayout | None:
    """Fit the hook on 1 or 2 lines, as large as possible up to `max_px`.

    With a face, the hook sits just above the face top (`face_top`, output px) and
    shrinks to fit the space above it, but not below `min_px`; if it still does not fit
    it starts at `top_limit_frac` and may overlap the head. Without a face it starts at
    `fallback_top_frac`. It always fits the frame width."""
    words = text.split()
    if not words:
        return None
    side = 2 * (HOOK_BOX_PAD_EM if style == "box" else HOOK_OUTLINE_EM)
    usable = width - 2 * HOOK_MARGIN
    top_limit = top_limit_frac * height
    gap = HOOK_FACE_GAP * height

    best: tuple[tuple[float, int, float], list[str], float] | None = None
    candidates = [[" ".join(words)]] + [
        [" ".join(words[:k]), " ".join(words[k:])] for k in range(1, len(words))
    ]
    for lines in candidates:
        widest = max(metrics.width(line, 1.0) for line in lines)
        px_w = usable / (widest + side)
        if face_top is None:
            px_h = math.inf
        else:
            top, bottom = _extent(metrics, len(lines), style)
            px_h = max(0.0, face_top - gap - top_limit) / (bottom - top)
        px = min(max_px, px_w, max(min_px, px_h))
        key = (math.floor(px), -len(lines), -widest)  # biggest, then fewer, then balanced
        if best is None or key > best[0]:
            best = (key, lines, px)
    assert best is not None
    _, lines, px = best
    px = math.floor(px)

    top_em, bottom_em = _extent(metrics, len(lines), style)
    block = (bottom_em - top_em) * px
    if face_top is None:
        top = fallback_top_frac * height
    else:
        top = max(top_limit, face_top - gap - block)
    cell_top = top - top_em * px
    center = cell_top + len(lines) * metrics.cell / metrics.upm * px / 2
    return HookLayout(lines, px, round(center), round(top), round(top + block))


def is_caps_word(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def hook_markup(lines: list[str], style: HookStyle) -> str:
    """Hook text with line breaks; in outline style ALL-CAPS words are yellow."""
    if style == "box":
        return "\\N".join(lines)
    return "\\N".join(
        " ".join(
            f"{{\\c{YELLOW}}}{w}{{\\c{WHITE}}}" if is_caps_word(w) else w for w in line.split()
        )
        for line in lines
    )


# --- ASS file ---------------------------------------------------------------------


@dataclass(frozen=True)
class CaptionStyle:
    width: int = 1080
    height: int = 1920
    font: str = "Montserrat Black"
    caption_px: float = 90
    caption_y_frac: float = 0.66
    hook_style: HookStyle = "outline"
    hook_sec: float = 3.0
    hook_max_px: float = 150
    hook_min_px: float = 90
    hook_fallback_top_frac: float = 0.09
    hook_top_limit_frac: float = 0.05


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
Style: Caption,{font},{caption_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,\
-1,0,0,0,100,100,{spacing},0,1,{outline},{shadow},5,{margin},{margin},0,1
{hook_style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

HOOK_STYLES = {
    # white text, thick black outline, soft shadow
    "outline": "Style: Hook,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
    "-1,0,0,0,100,100,0,0,1,{outline},{shadow},5,0,0,0,1",
    # black text on an opaque white box; BorderStyle 3 draws the box, Outline pads it
    "box": "Style: Hook,{font},{size},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,"
    "-1,0,0,0,100,100,0,0,3,{pad},0,5,0,0,0,1",
}


def ass_time(sec: float) -> str:
    cs = max(0, round(sec * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def build_ass(
    words: list[Word],
    hook_text: str,
    style: CaptionStyle,
    metrics: FontMetrics = ESTIMATED,
    face_top: float | None = None,
) -> str:
    """Full ASS file. `words` are caption words relative to the clip start (t=0);
    `face_top` is the face's top edge in output pixels, if one was found."""
    s = style
    clean = " ".join(hook_text.replace("{", "").replace("}", "").replace("\\", "").split())
    hook = layout_hook(
        clean,
        metrics,
        s.hook_style,
        width=s.width,
        height=s.height,
        max_px=s.hook_max_px,
        min_px=s.hook_min_px,
        face_top=face_top,
        fallback_top_frac=s.hook_fallback_top_frac,
        top_limit_frac=s.hook_top_limit_frac,
    )
    hook_px = hook.px if hook else s.hook_max_px
    hook_style = HOOK_STYLES[s.hook_style].format(
        font=s.font,
        size=metrics.ass_size(hook_px),
        outline=max(4, round(HOOK_OUTLINE_EM * hook_px)),
        shadow=max(2, round(HOOK_SHADOW_EM * hook_px)),
        pad=round(HOOK_BOX_PAD_EM * hook_px),
    )
    lines = [
        HEADER.format(
            width=s.width,
            height=s.height,
            font=s.font,
            caption_size=metrics.ass_size(s.caption_px),
            spacing=CAPTION_SPACING,
            outline=CAPTION_OUTLINE,
            shadow=CAPTION_SHADOW,
            margin=CAPTION_MARGIN,
            hook_style=hook_style,
        )
    ]
    if hook and s.hook_sec > 0:
        # \q2: no automatic wrapping; the layout already decided the lines
        tags = (
            f"{{\\an5\\pos({s.width // 2},{hook.center_y})\\q2"
            f"{_scale(100, HOOK_POP_MS)}\\fad(0,{HOOK_FADE_MS})}}"
        )
        lines.append(
            f"Dialogue: 0,{ass_time(0)},{ass_time(s.hook_sec)},Hook,,0,0,0,,"
            f"{tags}{hook_markup(hook.lines, s.hook_style)}\n"
        )
    pos = f"{{\\an5\\pos({s.width // 2},{round(s.height * s.caption_y_frac)})}}"
    max_width = s.width - 2 * CAPTION_MARGIN
    for start, end, text in caption_events(chunk_words(words), metrics, s.caption_px, max_width):
        lines.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Caption,,0,0,0,,{pos}{text}\n")
    return "".join(lines)
