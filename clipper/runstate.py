"""Shared state between an on-demand pipeline run and the Telegram bot (PLAN.md §17).

- data/run.stop          exists = graceful stop requested (the run checks it between clips
                         and stages, then exits)
- data/run.status.json   written by the run: state, current video/stage, counts, why it
                         stopped. The bot turns it into the pinned control panel.

States: starting (bot launched it) -> running -> stopping (stop flag seen or set) ->
finished (with `reason`). "idle" is what the bot shows when no run holds the lock.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

STOP_FILE = "run.stop"
STATUS_FILE = "run.status.json"


def request_stop(data_dir: Path) -> None:
    (data_dir / STOP_FILE).write_text("stop\n")


def clear_stop(data_dir: Path) -> None:
    (data_dir / STOP_FILE).unlink(missing_ok=True)


def stop_requested(data_dir: Path) -> bool:
    return (data_dir / STOP_FILE).exists()


def read_status(data_dir: Path) -> dict[str, Any]:
    try:
        value = json.loads((data_dir / STATUS_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def write_status(data_dir: Path, status: dict[str, Any]) -> None:
    """Atomic write, so the bot never reads half a file."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / STATUS_FILE
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(status, indent=1))
    tmp.replace(path)


def update_status(data_dir: Path, **fields: Any) -> dict[str, Any]:
    status = read_status(data_dir) | fields
    write_status(data_dir, status)
    return status
