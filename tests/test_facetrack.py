import itertools

import pytest

from clipper.render.facetrack import (
    Crop,
    Face,
    crop_path,
    sendcmd_script,
    smooth_path,
    track_primary,
)

A = Face(cx=0.3, size=0.10, score=0.9)
B = Face(cx=0.7, size=0.20, score=0.9)  # bigger face, other side of the frame


def test_track_follows_single_face_and_keeps_gaps():
    assert track_primary([[A], [], [A]]) == [0.3, None, 0.3]


def test_track_ignores_brief_bigger_face():
    frames = [[A]] * 5 + [[A, B]] * 5 + [[A]] * 5  # B is around for 1 s (< 1.5 s)
    assert track_primary(frames) == [0.3] * 15


def test_track_switches_after_min_time():
    out = track_primary([[A]] * 2 + [[A, B]] * 10)
    assert out[:9] == [0.3] * 9  # 7 frames at 5 fps = 1.4 s: not yet
    assert out[9:] == [0.7] * 3


def test_track_switches_at_once_when_current_face_is_gone():
    assert track_primary([[A], [B]]) == [0.3, 0.7]


def test_smooth_no_faces_is_center():
    assert smooth_path([None, None]) == [0.5, 0.5]


def test_smooth_holds_gaps():
    assert smooth_path([None, 0.4, None, None]) == [0.4] * 4


def test_smooth_ignores_jitter():
    assert smooth_path([0.5, 0.52, 0.49, 0.51] * 5) == [0.5] * 20


def test_smooth_cuts_on_big_jump():
    out = smooth_path([0.3] * 5 + [0.7] * 5)
    assert out == [0.3] * 5 + [0.7] * 5


def test_smooth_eases_slow_movement():
    out = smooth_path([0.3] * 3 + [0.45] * 30)
    assert all(b >= a for a, b in itertools.pairwise(out))
    assert out[3] < 0.35  # no jump
    assert out[-1] == pytest.approx(0.45, abs=0.05)


def test_crop_path_interpolates_and_clamps():
    crop = crop_path([0.0, 0.5, 1.0], 1920, 1080, 1080, 1920, n_frames=13, fps=30)
    assert (crop.w, crop.h, crop.y) == (606, 1076, 2)
    assert crop.xs[0] == 0  # clamped at the left edge
    assert crop.xs[6] == 960 - 303  # t = 0.2 s -> second sample, face centred
    assert crop.xs[12] == 1920 - 606  # clamped at the right edge
    assert crop.xs == sorted(crop.xs)


def test_crop_path_narrow_source_keeps_full_width():
    crop = crop_path([0.5], 1080, 1920, 1080, 1920, n_frames=3, fps=30)
    assert (crop.w, crop.h, crop.y, crop.xs) == (1080, 1920, 0, [0, 0, 0])


def test_sendcmd_only_on_change():
    script = sendcmd_script(Crop(606, 1076, 2, [10, 10, 20, 20, 30]), fps=30)
    assert script == "0.0667 crop x 20;\n0.1333 crop x 30;\n"
    assert sendcmd_script(Crop(606, 1076, 2, [10, 10]), fps=30).strip() == ""
