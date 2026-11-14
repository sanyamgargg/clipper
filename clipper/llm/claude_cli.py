"""Wrap `claude -p` (subscription login). The only LLM entry point in the pipeline."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper.config import Config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a precise assistant. Follow the output format in the request exactly."


@dataclass
class LLMResult:
    text: str
    exit_code: int
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class ClaudeCLIError(RuntimeError):
    """`claude -p` exited non-zero or reported an error (e.g. credit exhausted)."""

    def __init__(self, message: str, result: LLMResult) -> None:
        super().__init__(message)
        self.result = result


def run_claude(cfg: Config, prompt: str, cwd: Path) -> LLMResult:
    """One `claude -p` call, prompt on stdin. `cwd` should hold no CLAUDE.md, so the call
    carries no project context. Raises ClaudeCLIError on failure."""
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not found on PATH")
    model = cfg.claude.model
    cmd = [
        exe, "-p",
        "--model", model,
        "--output-format", "json",
        "--max-turns", str(cfg.claude.max_turns),
        "--tools", "",  # pure text task: no tools, no MCP, no skills
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt", SYSTEM_PROMPT,
    ]  # fmt: skip
    # Never pass an API key down: with one set, claude bills the API, not the subscription.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    try:
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, cwd=cwd, env=env, check=False,
            timeout=cfg.claude.timeout_sec,
        )  # fmt: skip
    except subprocess.TimeoutExpired as exc:
        result = LLMResult(text="", exit_code=124, model=model)
        raise ClaudeCLIError(
            f"claude -p timed out after {cfg.claude.timeout_sec} s", result
        ) from exc

    result = LLMResult(text=proc.stdout, exit_code=proc.returncode, model=model)
    try:
        payload: dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {}
    usage = payload.get("usage") or {}
    if usage:
        result.input_tokens = sum(
            int(usage.get(k) or 0)
            for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        result.output_tokens = int(usage.get("output_tokens") or 0)
    result.cost_usd = payload.get("total_cost_usd")
    if isinstance(payload.get("result"), str):
        result.text = payload["result"]

    if proc.returncode != 0 or payload.get("is_error") or not payload:
        detail = (result.text or proc.stderr).strip()[-500:]
        raise ClaudeCLIError(f"claude -p failed (exit {proc.returncode}): {detail}", result)
    return result


def extract_json(text: str) -> Any:
    """Parse the JSON object in a model reply, tolerating code fences or stray prose."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start : end + 1])
