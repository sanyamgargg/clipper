"""Segment checks: a broken leftover is not reused, a broken download is retried once.
Uses tiny real files made with ffmpeg (lavfi); yt-dlp is faked."""

import shutil
import subprocess

import pytest

from clipper.config import Config, Machine
from clipper.stages import fetch
from clipper.stages.fetch import VideoMeta, segment_problem

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def make(path, seconds, audio=True):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:d={seconds}", "-shortest"]
    subprocess.run([*cmd, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_segment_problem(tmp_path):
    assert segment_problem(make(tmp_path / "ok.mp4", 2), 2) is None
    assert segment_problem(make(tmp_path / "mute.mp4", 2, audio=False), 2) == "no audio stream"
    assert segment_problem(make(tmp_path / "short.mp4", 1), 2).startswith("only 1.0 s of 2.0 s")
    (tmp_path / "junk.mp4").write_bytes(b"not a video")
    assert segment_problem(tmp_path / "junk.mp4", 2) == "no video stream"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = Config(machine=Machine(data_dir=tmp_path, min_free_gb=0))
    meta = VideoMeta("vid", "https://x/vid", "t", 600, "20261101", "tjr")
    downloads = []

    def fake_ytdlp(*args):
        out = args[args.index("-o") + 1]
        downloads.append(out)
        make(out, 2, audio=len(downloads) > 1)  # the first download comes out without audio

    monkeypatch.setattr(fetch, "ytdlp", fake_ytdlp)
    return cfg, meta, downloads


def test_broken_leftover_is_downloaded_again(setup):
    cfg, meta, downloads = setup
    make(cfg.dir("sources") / "vid_01.mp4", 2, audio=False)  # left by an earlier run
    out = fetch.fetch_segment(cfg, meta, 1, 10, 12)
    assert segment_problem(out, 2) is None and len(downloads) == 2  # retried once


def test_good_leftover_is_reused(setup):
    cfg, meta, downloads = setup
    make(cfg.dir("sources") / "vid_01.mp4", 2)
    fetch.fetch_segment(cfg, meta, 1, 10, 12)
    assert downloads == []
