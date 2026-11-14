"""Free-disk-space guard. Call before any download or render."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


class InsufficientDiskSpace(RuntimeError):
    pass


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def ensure_free_space(path: Path, min_free_gb: float, needed_gb: float = 0.0) -> None:
    """Raise InsufficientDiskSpace unless `needed_gb` + `min_free_gb` is free on path's volume."""
    path.mkdir(parents=True, exist_ok=True)
    free = free_gb(path)
    required = min_free_gb + needed_gb
    if free < required:
        raise InsufficientDiskSpace(
            f"{free:.1f} GB free on {path}, need {required:.1f} GB "
            f"({min_free_gb:.1f} reserve + {needed_gb:.1f} for this job)"
        )
    log.info("disk ok: %.1f GB free (need %.1f)", free, required)
