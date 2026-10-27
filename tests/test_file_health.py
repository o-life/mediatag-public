"""file_health 体检核心单测——假 ffprobe/ffmpeg 脚本,不依赖真二进制。"""

import textwrap


from src.core.file_health import HealthChecker

HEALTHY_FFMPEG = """#!/usr/bin/env python3
import sys
print("frame=   30 fps=0.0 q=-1.0", file=sys.stderr)
sys.exit(0)
"""

TAIL_CORRUPT_FFMPEG = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
ss = float(args[args.index("-ss") + 1])
if ss >= 45:
    print("frame=    0 fps=0.0", file=sys.stderr)
    print("[h264] Error splitting the input into NAL units", file=sys.stderr)
else:
    print("frame=   30 fps=0.0", file=sys.stderr)
sys.exit(0)
"""

ALL_DEAD_FFMPEG = """#!/usr/bin/env python3
import sys
print("frame=    0 fps=0.0", file=sys.stderr)
print("[h264] corrupt input packet", file=sys.stderr)
sys.exit(1)
"""

GOOD_PROBE = """#!/usr/bin/env python3
print("100.0")
"""

BAD_PROBE = """#!/usr/bin/env python3
sys.exit(1)
"""


def make_checker(tmp_path, ffmpeg_src, probe_src=GOOD_PROBE):
    ff = tmp_path / "ffprobe_fake"
    ff.write_text(textwrap.dedent(probe_src))
    ff.chmod(0o755)
    fm = tmp_path / "ffmpeg_fake"
    fm.write_text(textwrap.dedent(ffmpeg_src))
    fm.chmod(0o755)
    return HealthChecker(ffprobe=str(ff), ffmpeg=str(fm)), tmp_path / "video.mp4"


def test_healthy_video_ok(tmp_path):
    hc, vid = make_checker(tmp_path, HEALTHY_FFMPEG)
    vid.write_bytes(b"x")
    r = hc.check(str(vid))
    assert r["verdict"] == "ok" and r["detail"] == ""


def test_tail_truncation_partial(tmp_path):
    hc, vid = make_checker(tmp_path, TAIL_CORRUPT_FFMPEG)
    vid.write_bytes(b"x")
    r = hc.check(str(vid))
    assert r["verdict"] == "部分损坏"
    assert "0帧可解" in r["detail"]


def test_all_positions_dead(tmp_path):
    hc, vid = make_checker(tmp_path, ALL_DEAD_FFMPEG)
    vid.write_bytes(b"x")
    r = hc.check(str(vid))
    assert r["verdict"] == "全坏"


def test_probe_fail_and_no_decode_container_broken(tmp_path):
    hc, vid = make_checker(tmp_path, ALL_DEAD_FFMPEG, probe_src=BAD_PROBE)
    vid.write_bytes(b"x")
    r = hc.check(str(vid))
    assert r["verdict"] == "容器损坏"


def test_missing_and_image_and_noerr(tmp_path):
    hc, vid = make_checker(tmp_path, HEALTHY_FFMPEG)
    r = hc.check(str(tmp_path / "nope.mp4"))
    assert r["verdict"] == "文件缺失"
    img = tmp_path / "pic.heic"
    img.write_bytes(b"x")
    assert hc.check(str(img))["verdict"] == "图片跳过"
