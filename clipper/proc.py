"""Process plumbing: the single-run lock and a subprocess runner that never leaks children.

A heavy child (whisper, ffmpeg, yt-dlp, claude) must not outlive the run that started it:
on 8 GB a leftover whisper next to a new run is exactly the parallel load we forbid.
- Normal exit, Ctrl-C, SIGTERM: run() kills the child's process group on the way out.
- SIGKILL of the run: the child's pid is in <lock>.child; the next run kills it first.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

LOCK_NAME = "clipper.lock"
_child_file: Path | None = None  # set while a run lock is held


class AlreadyRunning(RuntimeError):
    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _kill_group(pid: int) -> None:
    """Kill a child started by run() and everything it spawned (it leads its own group)."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(pid, signal.SIGKILL)
        else:  # Windows
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def run(
    cmd: list[str],
    *,
    input: str | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """subprocess.run(capture_output=True, text=True) that kills the child (and its own
    children) whenever this process stops waiting for it, for any reason."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,  # own process group, so the whole tree can be killed
    )
    if _child_file is not None:
        _child_file.write_text(str(proc.pid))
    try:
        out, err = proc.communicate(input, timeout=timeout)
    except BaseException:  # incl. KeyboardInterrupt, SystemExit (SIGTERM), TimeoutExpired
        _kill_group(proc.pid)
        proc.wait()
        raise
    finally:
        if _child_file is not None:
            _child_file.unlink(missing_ok=True)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


@contextmanager
def run_lock(data_dir: Path) -> Iterator[None]:
    """Hold data/clipper.lock for the duration of a run. Raises AlreadyRunning if another
    live run holds it. A lock left by a dead run is taken over, after killing the child
    process that run may have left behind."""
    global _child_file
    data_dir.mkdir(parents=True, exist_ok=True)
    lock = data_dir / LOCK_NAME
    child = data_dir / (LOCK_NAME + ".child")
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            pid = _read_pid(lock)
            if pid is not None and _alive(pid):
                raise AlreadyRunning(f"another run is active (pid {pid})") from None
            log.warning("removing stale lock left by pid %s", pid)
            orphan = _read_pid(child)
            if orphan is not None and _alive(orphan):
                log.warning("killing child process %d left by that run", orphan)
                _kill_group(orphan)
            child.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)
    else:
        raise AlreadyRunning("could not take the run lock")
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    _child_file = child
    try:
        yield
    finally:
        _child_file = None
        lock.unlink(missing_ok=True)
