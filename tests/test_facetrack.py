import itertools

import pytest

from clipper.render.facetrack import (
    Crop,
    Face,
    crop_path,
    face_bottoms,
    face_top,
    sendcmd_script,
    shot_cuts,
    smooth_path,
    square_y_fracs,
    to_output_y,
    track_primary,
    tracked,
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


def test_tracked_returns_the_followed_detection_per_frame():
    me, other = Face(0.3, 0.1, 0.9), Face(0.8, 0.3, 0.9)
    assert tracked([[me, other], [], [other]], [0.3, None, 0.8]) == [me, None, other]


def test_largest_person_is_followed_when_there_is_no_face():
    small = Face(0.2, 0.10, 0.9, kind="person")
    large = Face(0.6, 0.30, 0.8, kind="person")
    assert track_primary([[small, large]]) == [0.6]


def test_face_top_is_the_highest_top_of_the_tracked_face_while_the_hook_shows():
    subjects = [
        Face(0.3, 0.1, 0.9, top=0.22),
        None,
        Face(0.3, 0.1, 0.9, top=0.18),
        Face(0.3, 0.4, 0.9, top=0.02, kind="person"),  # a person box is not a face
        Face(0.3, 0.1, 0.9, top=0.01),  # after the hook
    ]
    assert face_top(subjects, n=4) == 0.18
    assert face_top([None, Face(0.5, 0.3, 0.9, top=0.1, kind="person")], n=2) is None


def test_face_bottoms_are_the_estimated_chin_of_faces_only():
    subjects = [
        Face(0.3, 0.1, 0.9, top=0.35, bottom=0.65),  # box stops at the mouth
        None,
        Face(0.3, 0.4, 0.9, top=0.1, bottom=0.9, kind="person"),
    ]
    assert face_bottoms(subjects) == [pytest.approx(0.65 + 0.25 * 0.30), None, None]


def test_shot_cuts_on_kind_size_or_position_jump():
    wide = Face(0.46, 0.06, 0.7, kind="person")
    close = Face(0.56, 0.17, 0.8)
    closer = Face(0.56, 0.33, 0.8)
    assert shot_cuts([wide, close, close, None, close, closer]) == {1, 5}
    assert shot_cuts([close, Face(0.6, 0.2, 0.8), None]) == set()  # normal movement
    guest = Face(0.80, 0.17, 0.8)  # cut to the other speaker, same framing
    assert shot_cuts([close, guest, close]) == {1, 2}


def test_crop_snaps_at_a_shot_cut_instead_of_easing():
    xs = [0.46, 0.56, 0.56, 0.56]
    eased = smooth_path(xs)
    assert eased[1] < 0.5  # without a cut, 0.10 is eased over several samples
    assert smooth_path(xs, cuts={1}) == [0.46, 0.56, 0.56, 0.56]
    crop = crop_path([0.46, 0.56], 1920, 1080, 1080, 1920, n_frames=12, fps=30, cuts={1})
    first = round(0.46 * 1920 - 303)
    assert crop.xs[:6] == [first] * 6  # held until the cut, not interpolated
    assert crop.xs[6:] == [round(0.56 * 1920 - 303)] * 6


@pytest.mark.parametrize(
    ("cut_time", "old_frames"),
    [
        (0.29, 9),  # camera cut just after sample 1 (0.2 s): frames 0-8 (< 0.29 s) stay
        (0.15, 5),  # camera cut before sample 1: switch early, at 0.15 s
        (0.90, 6),  # no camera cut near sample 1: switch at the sample (0.2 s)
    ],
)
def test_crop_switches_at_the_real_camera_cut(cut_time, old_frames):
    old, new = round(0.46 * 1920 - 303), round(0.56 * 1920 - 303)
    crop = crop_path([0.46, 0.56, 0.56], 1920, 1080, 1080, 1920, n_frames=15, fps=30,
                     cuts={1}, cut_times=[cut_time])  # fmt: skip
    assert crop.xs == [old] * old_frames + [new] * (15 - old_frames)


def test_to_output_y_crop_and_fit():
    crop = Crop(w=606, h=1076, y=2, xs=[0])
    assert to_output_y(0.0, 1920, 1080, 1080, 1920, crop) == pytest.approx(-2 * 1920 / 1076)
    assert to_output_y(0.5, 1920, 1080, 1080, 1920, crop) == pytest.approx(538 * 1920 / 1076)
    # fit: 1920x1080 scaled to 1080x607.5, centred: band from 656.25 to 1263.75
    assert to_output_y(0.0, 1920, 1080, 1080, 1920, None) == pytest.approx(656.25)
    assert to_output_y(1.0, 1920, 1080, 1080, 1920, None) == pytest.approx(1263.75)


def test_sendcmd_only_on_change():
    script = sendcmd_script(Crop(606, 1076, 2, [10, 10, 20, 20, 30]), fps=30)
    assert script == "0.0667 crop x 20;\n0.1333 crop x 30;\n"
    assert sendcmd_script(Crop(606, 1076, 2, [10, 10]), fps=30).strip() == ""


def test_square_crop_is_capped_at_source_resolution():
    crop = crop_path([0.5], 1920, 1080, 1, 1, n_frames=2, fps=30, max_side=1080)
    assert (crop.w, crop.h, crop.y, crop.ys) == (1080, 1080, 0, None)  # full height, no room
    big = crop_path([0.5], 3840, 2160, 1, 1, n_frames=2, fps=30, max_side=1080,
                    y_fracs=[0.25, 0.25])  # fmt: skip
    assert (big.w, big.h) == (1080, 1080)  # 1080 source pixels, no downscale of a 4K frame
    assert big.ys == [round(0.25 * (2160 - 1080))] * 2 and big.y == 270


def test_square_y_puts_eyes_at_38_percent_but_never_cuts_the_head():
    face = Face(0.5, 0.1, 0.9, top=0.40, bottom=0.55, eye_y=0.46)
    assert square_y_fracs([face, None], src_h=1080, side=1080) == [0.0, 0.0]  # no room
    (frac,) = square_y_fracs([face], src_h=2160, side=1080)
    y = frac * (2160 - 1080)
    assert y == pytest.approx(0.46 * 2160 - 0.38 * 1080)  # eyes at 38% of the square
    high = Face(0.5, 0.1, 0.9, top=0.10, bottom=0.40, eye_y=0.30)
    (frac,) = square_y_fracs([high], src_h=2160, side=1080)
    assert frac * 1080 <= (0.10 - 0.35 * 0.30) * 2160 + 1e-6 or frac == 0.0  # head stays in
    assert square_y_fracs([Face(0.5, 0.3, 0.9, kind="person")], 2160, 1080) == [None]


def test_sendcmd_moves_x_and_y():
    script = sendcmd_script(Crop(1080, 1080, 0, [10, 10, 20], ys=[5, 6, 6]), fps=30)
    assert script == "0.0333 crop y 6;\n0.0667 crop x 20;\n"
