"""Pure transcript-timing logic: words, sentences, compact transcript, clip snapping."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

SENTENCE_END = (".", "?", "!")
MAX_SENTENCE_WORDS = 40  # hard cap when whisper gives no punctuation
START_PAD = 0.10  # breathing room before the first word / after the last, if silence allows
END_PAD = 0.30


@dataclass(frozen=True)
class Word:
    w: str
    s: float
    e: float


def load_words(path: Path) -> list[Word]:
    return [Word(**d) for d in json.loads(path.read_text())]


def save_words(words: list[Word], path: Path) -> None:
    path.write_text(json.dumps([asdict(w) for w in words]))


def parse_ts(ts: str | float) -> float:
    """'mm:ss', 'h:mm:ss' or plain seconds -> seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = ts.strip().strip("[]").split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"bad timestamp: {ts!r}")
    sec = 0.0
    for p in parts:
        sec = sec * 60 + float(p)
    return sec


def fmt_ts(sec: float) -> str:
    """Seconds -> 'mm:ss' (minutes keep counting past 59, floor)."""
    total = int(sec)
    return f"{total // 60:02d}:{total % 60:02d}"


def words_from_whisper(segments: list[dict]) -> list[Word]:
    """whisper.cpp JSON (`-ml 1 -sow`) -> words. Pieces without a leading space join the
    previous word ("don" + "'t"); bracketed non-speech like [MUSIC] is dropped."""
    words: list[Word] = []
    for seg in segments:
        raw: str = seg["text"]
        text = raw.strip()
        if not text or text[0] in "[(" or text[-1] in "])":
            continue
        s = seg["offsets"]["from"] / 1000
        e = seg["offsets"]["to"] / 1000
        if words and not raw.startswith(" "):
            prev = words[-1]
            words[-1] = Word(prev.w + text, prev.s, max(prev.e, e))
        else:
            words.append(Word(text, s, max(e, s)))
    return words


def sentences(words: list[Word]) -> list[tuple[int, int]]:
    """Sentence index ranges [i, j) over words."""
    out: list[tuple[int, int]] = []
    start = 0
    for i, w in enumerate(words):
        if w.w.endswith(SENTENCE_END):
            out.append((start, i + 1))
            start = i + 1
        elif i - start + 1 >= MAX_SENTENCE_WORDS:
            # run-on: break at the last comma in the second half, else right here
            half = start + MAX_SENTENCE_WORDS // 2
            cut = next((k for k in range(i, half, -1) if words[k].w.endswith(",")), i)
            out.append((start, cut + 1))
            start = cut + 1
    if start < len(words):
        out.append((start, len(words)))
    return out


def compact_transcript(words: list[Word]) -> str:
    """One line per sentence: '[mm:ss] text'. This is what Claude sees."""
    return "\n".join(
        f"[{fmt_ts(words[i].s)}] {' '.join(w.w for w in words[i:j])}" for i, j in sentences(words)
    )


def snap_clip(
    words: list[Word], start: float, end: float, min_sec: float, max_sec: float
) -> tuple[float, float] | None:
    """Map a rough (start, end) to sentence boundaries, so a clip never cuts mid-word or
    mid-sentence. `start` is when the first sentence begins; `end` is when the first
    sentence after the clip begins. Trims/extends by whole sentences to fit
    [min_sec, max_sec]; returns None if that is impossible."""
    sents = sentences(words)
    if not sents:
        return None
    starts = [words[i].s for i, _ in sents]
    first = min(range(len(sents)), key=lambda k: abs(starts[k] - start))
    # boundary k means "clip ends before sentence k"; len(sents) = end of transcript
    bounds = starts + [words[-1].e]
    stop = min(range(first + 1, len(bounds)), key=lambda k: abs(bounds[k] - end))

    def span(a: int, b: int) -> float:
        return words[sents[b - 1][1] - 1].e - words[sents[a][0]].s

    while stop - first > 1 and span(first, stop) > max_sec:
        stop -= 1
    while stop < len(sents) and span(first, stop) < min_sec and span(first, stop + 1) <= max_sec:
        stop += 1
    if not min_sec <= span(first, stop) <= max_sec:
        return None

    i, j = sents[first][0], sents[stop - 1][1] - 1  # first and last word index
    s, e = words[i].s, words[j].e
    if i > 0:
        s -= min(START_PAD, max(0.0, s - words[i - 1].e))
    else:
        s = max(0.0, s - START_PAD)
    if j + 1 < len(words):
        e += min(END_PAD, max(0.0, words[j + 1].s - e))
    else:
        e += END_PAD
    return round(s, 3), round(e, 3)


def words_between(words: list[Word], start: float, end: float) -> list[Word]:
    """Words fully inside [start, end], re-based so the clip starts at t=0."""
    return [
        Word(w.w, round(w.s - start, 3), round(w.e - start, 3))
        for w in words
        if w.s >= start - 0.01 and w.e <= end + 0.01
    ]
