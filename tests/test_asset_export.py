"""asset_export 单测（影楼 S4）：命名模板 / 冲突规避 / 复制与转码 / XMP sidecar。"""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, "src")

from src.core.asset_export import (  # noqa: E402
    export_asset, plan_dest, render_name, sanitize_name,
)


def _jpg(path: Path, size=(800, 600)) -> Path:
    img = Image.new("RGB", size, (120, 40, 200))
    img.save(path, "JPEG", quality=90)
    return path


# ── 命名模板 ─────────────────────────────────────────────────────────

def test_render_name_default_template():
    name = render_name("{项目名}_{序号:03d}_{原文件名}",
                       "客户A·婚礼", 7, "IMG_0016.JPG")
    assert name == "客户A·婚礼_007_IMG_0016"


def test_render_name_sanitizes_illegal_chars():
    assert sanitize_name('a<b>:c|d?e*f"g') == "a b  c d e f g"
    assert sanitize_name("  .x. ") == "x"
    # 模板引用不存在的占位符 → 原样输出（不强杀用户输入）
    assert render_name("{不存在的键}", "p", 1, "f.jpg") == "{不存在的键}"


def test_plan_dest_avoids_overwrite(tmp_path):
    p1 = plan_dest(tmp_path, "a", ".jpg")
    p1.write_bytes(b"x")
    p2 = plan_dest(tmp_path, "a", ".jpg")
    assert p1.name == "a.jpg" and p2.name == "a-1.jpg"


# ── 导出行为 ─────────────────────────────────────────────────────────

def test_export_video_copies_without_xmp(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"fake video")
    out = tmp_path / "out"
    r = export_asset(str(src), out, "clip01", is_video=True,
                     write_xmp=True, keywords=["标签"])
    dest = Path(r["dest"])
    assert dest.name == "clip01.mp4"
    assert dest.read_bytes() == b"fake video"
    assert r["xmp"] is None  # 视频 XMP 无消费方，不写
    assert list(out.glob("*.xmp")) == []


def test_export_photo_jpeg_resizes_and_writes_xmp(tmp_path):
    src = _jpg(tmp_path / "IMG_1.JPG", size=(4000, 3000))
    out = tmp_path / "out"
    r = export_asset(str(src), out, "客户A_001", is_video=False,
                     photo_mode="jpeg", longest_edge=2048,
                     jpeg_quality=85, keywords=["婚礼", "室内"])
    dest = Path(r["dest"])
    assert dest.suffix == ".jpg"
    with Image.open(dest) as img:
        assert max(img.size) <= 2048
    xmp = Path(r["xmp"])
    assert xmp.exists() and xmp.name == "客户A_001.xmp"
    text = xmp.read_text(encoding="utf-8")
    assert "<rdf:li>婚礼</rdf:li>" in text and "<rdf:li>室内</rdf:li>" in text
    assert "dc:subject" in text


def test_export_photo_original_mode_copies(tmp_path):
    src = _jpg(tmp_path / "RAW_PAIR.JPG")
    out = tmp_path / "out"
    r = export_asset(str(src), out, "keep", is_video=False,
                     photo_mode="original", keywords=["tag"])
    dest = Path(r["dest"])
    # 扩展名统一小写（跨平台安全，避免 .JPG/.jpg 双份）
    assert dest.suffix == ".jpg" and dest.read_bytes() == src.read_bytes()
    assert Path(r["xmp"]).exists()


def test_export_undecodable_falls_back_to_copy(tmp_path):
    """无法解码的"照片"（假 JPG）→ 回退原样复制 + 警告，不丢文件。"""
    src = tmp_path / "broken.jpg"
    src.write_bytes(b"not an image at all")
    out = tmp_path / "out"
    r = export_asset(str(src), out, "brk", is_video=False,
                     photo_mode="jpeg", keywords=["t"])
    dest = Path(r["dest"])
    assert dest.read_bytes() == b"not an image at all"
    assert r["warnings"] and "已按原图复制" in r["warnings"][0]


def test_export_missing_source_raises(tmp_path):
    try:
        export_asset(str(tmp_path / "nope.mp4"), tmp_path / "out", "x",
                     is_video=True)
        raise AssertionError("should have raised")
    except FileNotFoundError:
        pass


def test_xmp_drops_keywords_with_xml_special_chars(tmp_path):
    src = _jpg(tmp_path / "s.jpg")
    r = export_asset(str(src), tmp_path / "out", "s", is_video=False,
                     keywords=["好标签", "<bad>&"])
    text = Path(r["xmp"]).read_text(encoding="utf-8")
    assert "好标签" in text and "<bad>" not in text
