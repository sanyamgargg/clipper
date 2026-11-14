"""transcribe: wav -> data/transcripts/<video_id>.words.json + .compact.txt

Runs whisper.cpp as a subprocess, so the model is unloaded when the stage returns.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from clipper import proc
from clipper.config import Config
from clipper.timing import Word, compact_transcript, words_from_whisper

log = logging.getLogger(__name__)


def whisper_prompt(vocabulary: list[str] | None) -> str:
    """Initial prompt that primes whisper with campaign names and jargon."""
    terms = [t.strip() for t in vocabulary or [] if t.strip()]
    return ", ".join(terms) + "." if terms else ""


def whisper_cmd(exe: str, model: Path, wav: Path, out_base: Path, prompt: str = "") -> list[str]:
    # -ml 1 -sow: one word per segment, which gives word-level timestamps
    cmd = [exe, "-m", str(model), "-f", str(wav), "-l", "en", "-ml", "1", "-sow",
           "-oj", "-of", str(out_base), "-np"]  # fmt: skip
    if prompt:
        # Carry the prompt into every 30 s window. Without it whisper conditions only the
        # first window on the prompt and, on a 26 min test video, stopped punctuating
        # after minute 7 (69 sentence ends instead of 392), which breaks clip snapping.
        cmd += ["--prompt", prompt, "--carry-initial-prompt"]
    return cmd


def _whisper_cpp(cfg: Config, wav: Path, out_base: Path, prompt: str = "") -> list[Word]:
    exe = shutil.which("whisper-cli")
    if not exe:
        raise RuntimeError("whisper-cli not found on PATH (run scripts/setup.sh)")
    model = cfg.data_dir / "models" / f"ggml-{cfg.machine.whisper_model}.bin"
    if not model.exists():
        raise RuntimeError(f"whisper model missing: {model} (run scripts/setup.sh)")
    if not wav.is_file():
        raise RuntimeError(f"audio missing: {wav}")
    result = proc.run(whisper_cmd(exe, model, wav, out_base, prompt))
    raw = out_base.with_name(out_base.name + ".json")
    if result.returncode != 0 or not raw.exists():
        raise RuntimeError(
            f"whisper-cli failed ({result.returncode}): {result.stderr.strip()[-800:]}"
        )
    words = words_from_whisper(json.loads(raw.read_text(errors="replace"))["transcription"])
    raw.unlink()
    return words


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def transcribe(
    cfg: Config, video_id: str, wav: Path, vocabulary: list[str] | None = None
) -> tuple[Path, Path]:
    """Returns (words.json, compact.txt). Skips whisper if both already exist.
    `vocabulary` (campaign names and jargon) is given to whisper as its initial prompt."""
    out_dir = cfg.dir("transcripts")
    words_path = out_dir / f"{video_id}.words.json"
    compact_path = out_dir / f"{video_id}.compact.txt"
    if words_path.exists() and compact_path.exists():
        log.info("transcript already present: %s", words_path.name)
        return words_path, compact_path

    backend = cfg.machine.whisper_backend
    if backend != "whisper_cpp":
        raise NotImplementedError(f"whisper backend not implemented yet: {backend}")
    log.info("transcribing %s with %s/%s", wav.name, backend, cfg.machine.whisper_model)
    words = _whisper_cpp(cfg, wav, out_dir / f"{video_id}.whisper", whisper_prompt(vocabulary))
    if not words:
        raise RuntimeError(f"no speech found in {wav}")

    compact = compact_transcript(words)
    _write_atomic(compact_path, compact + "\n")
    _write_atomic(words_path, json.dumps([{"w": w.w, "s": w.s, "e": w.e} for w in words]))
    log.info("transcribed: %d words, %d lines", len(words), compact.count("\n") + 1)
    return words_path, compact_path
