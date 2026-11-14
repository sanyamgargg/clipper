"""transcribe: wav -> data/transcripts/<video_id>.words.json + .compact.txt

Runs whisper.cpp as a subprocess, so the model is unloaded when the stage returns.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from clipper.config import Config
from clipper.timing import Word, compact_transcript, save_words, words_from_whisper

log = logging.getLogger(__name__)


def _whisper_cpp(cfg: Config, wav: Path, out_base: Path) -> list[Word]:
    exe = shutil.which("whisper-cli")
    if not exe:
        raise RuntimeError("whisper-cli not found on PATH (run scripts/setup.sh)")
    model = cfg.data_dir / "models" / f"ggml-{cfg.machine.whisper_model}.bin"
    if not model.exists():
        raise RuntimeError(f"whisper model missing: {model} (run scripts/setup.sh)")
    # -ml 1 -sow: one word per segment, which gives word-level timestamps
    cmd = [exe, "-m", str(model), "-f", str(wav), "-l", "en", "-ml", "1", "-sow",
           "-oj", "-of", str(out_base), "-np"]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    raw = out_base.with_name(out_base.name + ".json")
    if proc.returncode != 0 or not raw.exists():
        raise RuntimeError(f"whisper-cli failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
    words = words_from_whisper(json.loads(raw.read_text(errors="replace"))["transcription"])
    raw.unlink()
    return words


def transcribe(cfg: Config, video_id: str, wav: Path) -> tuple[Path, Path]:
    """Returns (words.json, compact.txt). Skips whisper if both already exist."""
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
    words = _whisper_cpp(cfg, wav, out_dir / f"{video_id}.whisper")
    if not words:
        raise RuntimeError(f"no speech found in {wav}")

    save_words(words, words_path)
    compact = compact_transcript(words)
    compact_path.write_text(compact + "\n")
    log.info("transcribed: %d words, %d lines", len(words), compact.count("\n") + 1)
    return words_path, compact_path
