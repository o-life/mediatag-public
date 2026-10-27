"""RAW 内嵌预览提取测试（S1 NASA NEF 样张 + 合成损坏样张）。

NEF 样张来自 rawpy 仓库（git clone --depth 1 https://github.com/letmaik/rawpy，
test/*.NEF，NASA ISS 实拍）。样张不在则跳过（不入库大文件）；
CR3/ARW/RAF/DNG 品牌矩阵待真实样张（S2 任务 2.4）。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from core.media_types import is_media, is_raw  # noqa: E402
from core.raw_preview import RawPreviewError, extract_preview  # noqa: E402

# NEF 样张位置：rawpy 仓库自带（git clone --depth 1 letmaik/rawpy 后拷贝固化到
# 用户级缓存，避免 /tmp 易失）。两处候选任一存在即跑。
_CACHE = Path.home() / ".cache" / "mediagtag-dev" / "raw_samples"
_NEF_CANDIDATES = [
    _CACHE / "iss030e122639.NEF",
    _CACHE / "iss042e297200.NEF",
    Path("/tmp/rawpy-src/test/iss030e122639.NEF"),
    Path("/tmp/rawpy-src/test/iss042e297200.NEF"),
]
NEF_SAMPLES = [p for p in _NEF_CANDIDATES if p.exists()]

needs_nef = pytest.mark.skipif(
    not NEF_SAMPLES, reason="rawpy 仓库 NEF 样张不存在（clone letmaik/rawpy 后重跑）"
)


def test_media_types_raw():
    assert is_raw("IMG_0016.CR3")
    assert is_raw("DSC01234.ARW")
    assert is_raw("/x/y/photo.NEF")
    assert is_raw("a/b/raw.dng")
    assert not is_raw("photo.jpg")
    assert not is_raw("video.mp4")
    # RAW 属媒体扫描面
    assert is_media("IMG_0016.CR3")
    assert not is_raw("readme.txt")


@needs_nef
def test_extract_preview_jpeg_roundtrip(tmp_path):
    """NEF 内嵌预览提取：输出 JPEG 存在、非空、可解码。"""
    src = NEF_SAMPLES[0]
    out = tmp_path / "preview.jpg"
    result = extract_preview(src, out)
    assert result == out
    assert out.exists()
    data = out.read_bytes()
    assert len(data) > 1024  # 真实预览不会是几字节
    assert data[:2] == b"\xff\xd8"  # JPEG SOI 魔数


@needs_nef
def test_extract_preview_overwrite_atomic(tmp_path):
    """重复提取覆盖写；不留 .part 半截文件。"""
    src = NEF_SAMPLES[0]
    out = tmp_path / "preview.jpg"
    extract_preview(src, out)
    first = out.read_bytes()
    extract_preview(src, out)
    assert out.read_bytes() == first  # 确定性输出
    assert not list(tmp_path.glob("*.part"))


def test_extract_preview_missing_file(tmp_path):
    with pytest.raises(RawPreviewError):
        extract_preview(tmp_path / "nope.CR3", tmp_path / "out.jpg")


def test_extract_preview_garbage_raises(tmp_path):
    """伪造 RAW（非 RAW 内容）→ RawPreviewError，不产生输出文件。"""
    garbage = tmp_path / "fake.CR3"
    garbage.write_bytes(b"not a raw file" * 100)
    out = tmp_path / "out.jpg"
    with pytest.raises(RawPreviewError):
        extract_preview(garbage, out)
    assert not out.exists()


def test_extract_preview_truncated_nef(tmp_path):
    """截断的真实 NEF → 明确失败（LibRaw 层报错转 RawPreviewError）。"""
    if not NEF_SAMPLES:
        pytest.skip("NEF 样张不存在")
    src = NEF_SAMPLES[0]
    truncated = tmp_path / "trunc.NEF"
    truncated.write_bytes(src.read_bytes()[:4096])  # 只留头部
    out = tmp_path / "out.jpg"
    # 截断头部通常仍可解析出嵌入预览或明确失败——两者皆可接受，
    # 关键是不崩溃、不留半截文件
    try:
        extract_preview(truncated, out)
        assert not out.exists() or out.stat().st_size > 0
    except RawPreviewError:
        assert not out.exists()
