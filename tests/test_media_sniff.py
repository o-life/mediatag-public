"""media_types.sniff_media_ext 内容嗅探测试。

背景：照片库 originals/ 是 UUID 无扩展名文件，旧实现按扩展名过滤导致
.mov 等视频报扩展名错误无法导入（用户实测）。嗅探按文件头还原真实类型。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from src.core.media_types import sniff_media_ext  # noqa: E402


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_jpeg_magic(tmp_path):
    assert sniff_media_ext(_write(tmp_path, "a", b"\xff\xd8\xff\xe0" + b"\x00" * 12)) == ".jpg"


def test_png_magic(tmp_path):
    assert sniff_media_ext(_write(tmp_path, "a", b"\x89PNG\r\n\x1a\n" + b"\x00" * 12)) == ".png"


def test_heic_ftyp_brand(tmp_path):
    data = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 12
    assert sniff_media_ext(_write(tmp_path, "a", data)) == ".heic"


def test_mp4_ftyp_brand(tmp_path):
    data = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 12
    assert sniff_media_ext(_write(tmp_path, "a", data)) == ".mp4"


def test_mov_quicktime_brand(tmp_path):
    data = b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 12
    assert sniff_media_ext(_write(tmp_path, "a", data)) == ".mov"


def test_unknown_returns_none(tmp_path):
    assert sniff_media_ext(_write(tmp_path, "a", b"\x00" * 32)) is None
    assert sniff_media_ext(_write(tmp_path, "a", b"short")) is None
    assert sniff_media_ext(tmp_path / "nonexistent") is None
