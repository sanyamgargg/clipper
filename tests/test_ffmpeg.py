from pathlib import Path

from clipper.config import Render
from clipper.render.facetrack import Crop
from clipper.render.ffmpeg import encoder_args, render_cmd, video_filter

R = Render()
CROP = Crop(w=606, h=1076, y=2, xs=[657, 660])


def test_crop_layout_filter():
    vf = video_filter(R, CROP, "crop.cmd", "captions.ass")
    assert vf == (
        "[0:v]fps=30,sendcmd=f=crop.cmd,crop=606:1076:657:2,scale=1080:1920:flags=lanczos,"
        "setsar=1,ass=captions.ass[v]"
    )
    assert "sendcmd" not in video_filter(R, CROP, None, "captions.ass")


def test_fonts_dir_is_passed_to_libass():
    vf = video_filter(R, CROP, None, "captions.ass", "fonts")
    assert vf.endswith(",setsar=1,ass=captions.ass:fontsdir=fonts[v]")
    cmd = render_cmd(Path("in.mp4"), Path("o.mp4"), CROP, R, "libx264", None, "c.ass", "fonts")
    assert cmd[cmd.index("-filter_complex") + 1].endswith("ass=c.ass:fontsdir=fonts[v]")


def test_fit_layout_filter_keeps_full_frame_over_blurred_fill():
    vf = video_filter(R, None, None, "captions.ass")
    bg, fg, out = vf.split(";")[1:]
    assert "crop=270:480" in bg and "boxblur" in bg and bg.endswith("[bg]")
    assert "scale=1080:1920" in bg  # background fills the whole output
    assert "force_original_aspect_ratio=decrease" in fg and "crop" not in fg  # nothing cut off
    assert out == "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,ass=captions.ass[v]"


def test_render_cmd_uses_configured_encoder_and_maps_filter_output():
    cmd = render_cmd(Path("in.mp4"), Path("out.mp4"), None, R, "libx264", None, "captions.ass")
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-map") + 1] == "[v]"
    assert "loudnorm=I=-14:TP=-1.5:LRA=11" in cmd
    assert cmd[-1] == "out.mp4"


def test_encoder_args_per_encoder():
    assert encoder_args("h264_videotoolbox", "9M") == [
        "-c:v", "h264_videotoolbox", "-b:v", "9M", "-pix_fmt", "yuv420p",
    ]  # fmt: skip
    assert "-preset" in encoder_args("libx264", "9M")
    assert encoder_args("h264_qsv", "9M")[-1] == "nv12"


def test_framed_filter_pads_the_square_onto_the_canvas():
    square = Crop(w=1080, h=1080, y=0, xs=[420])
    vf = video_filter(R, square, "crop.cmd", "c.ass", "fonts", framed=(484, 1080, "#0C0C0C"))
    assert vf == (
        "[0:v]fps=30,sendcmd=f=crop.cmd,crop=1080:1080:420:0,scale=1080:1080:flags=lanczos,"
        "pad=1080:1920:0:484:color=0x0C0C0C,setsar=1,ass=c.ass:fontsdir=fonts[v]"
    )
