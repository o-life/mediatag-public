"""按项目批量导出（影楼基础包 S4）—— 纯逻辑层，可单测。

职责：命名模板渲染、目标冲突规避、按规格复制/转码、XMP sidecar 写出。
不做 UI（对话框见 ui/components/project_export.py）、不做线程管理。

语义（2026-09-30 定稿）：
- 导出是**复制**，源文件与库内登记永不改动（项目=自持清单，见 project_assets）；
- 照片：默认按规格转码 JPEG（最长边+质量）；选"原图"则原样复制；
  无法解码的格式（如无插件的 HEIC）自动回退原样复制并记警告——保底不丢文件；
- 视频：原样复制（无转码语义）；
- XMP sidecar：随照片写出 `<目标名>.xmp`（dc:subject=AI 标签关键词，
  LR/C1 直接可读）。星级/择用标记的文件级数据源尚未建（现状评分挂人脸维度），
  xmp:Rating 暂不写——writer 预留 rating 参数，数据线落地后一行接入。
"""

import shutil
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

__all__ = [
    "ExportWarning",
    "render_name",
    "plan_dest",
    "export_asset",
    "DEFAULT_NAME_TEMPLATE",
]

DEFAULT_NAME_TEMPLATE = "{项目名}_{序号:03d}_{原文件名}"

# 文件名/目录名非法字符（macOS/Windows/网络盘交集；冒号另处理）
_ILLEGAL = '<>:"|?*'


class ExportWarning(Exception):
    """非致命导出警告（已回退保底路径），聚合后向用户展示。"""


def sanitize_name(name: str) -> str:
    """清洗模板渲染结果中的非法字符与首尾空白/点（Windows 目录约束为准）。"""
    out = name
    for ch in _ILLEGAL:
        out = out.replace(ch, " ")
    out = out.strip().strip(".")
    return out or "未命名"


def render_name(template: str, project_name: str, index: int,
                original_name: str) -> str:
    """渲染命名模板。可用占位符：{项目名} {序号} {原文件名} {日期}。

    - 序号支持格式化规格（默认模板 {序号:03d}）；
    - 渲染结果统一过 sanitize_name（模板自带非法字符也拦得住）；
    - 模板渲染为空串时回退"未命名"。"""
    from datetime import date
    ctx = {
        "项目名": project_name or "未归集",
        "序号": index,
        "原文件名": Path(original_name).stem,
        "日期": date.today().isoformat(),
    }
    try:
        rendered = template.format(**ctx)
    except (KeyError, ValueError, IndexError):
        rendered = template  # 模板写错时原样输出，不强杀用户输入
    return sanitize_name(rendered)


def plan_dest(dest_dir: Path, base_name: str, ext: str) -> Path:
    """目标路径规划：同名冲突自动追加序号（-1/-2/…），绝不覆盖既有文件。"""
    ext = ext if ext.startswith(".") else f".{ext}"
    candidate = dest_dir / f"{base_name}{ext}"
    n = 1
    while candidate.exists():
        candidate = dest_dir / f"{base_name}-{n}{ext}"
        n += 1
    return candidate


def _write_xmp_sidecar(xmp_path: Path, keywords: list) -> None:
    """写最小 XMP packet（dc:subject 关键词袋），LR/C1/ExifTool 直接可读。"""
    li = "".join(
        f"<rdf:li>{kw}</rdf:li>" for kw in keywords
        if kw and ("<" not in kw and "&" not in kw))  # 关键词含 XML 特殊字符时丢弃该词
    packet = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        f'   <dc:subject><rdf:Bag>{li}</rdf:Bag></dc:subject>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>\n'
    )
    xmp_path.write_text(packet, encoding="utf-8")


def export_asset(src_path: str, dest_dir: Path, base_name: str, *,
                 is_video: bool, photo_mode: str = "jpeg",
                 longest_edge: int = 0, jpeg_quality: int = 90,
                 write_xmp: bool = True, keywords: Optional[list] = None,
                 project_name: str = "") -> dict:
    """导出单个资产，返回 {dest, xmp, warnings}。

    - is_video：原样复制（copy2 保留时间戳）；
    - 照片 photo_mode="original"：原样复制；"jpeg"：PIL 转码 JPEG
      （longest_edge>0 时等比压到最长边；EXIF 方向与 EXIF 元数据保留）；
      解码失败（HEIC 无插件等）→ 回退原样复制 + 警告；
    - write_xmp 且非视频：随导出文件写同名 .xmp（dc:subject=keywords）。
    异常（源不存在等）直接抛出，由调用方计数——单文件失败不静默。"""
    from datetime import date

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"源文件不存在: {src}")

    warnings: list = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    decoded_jpeg = False
    if not is_video and photo_mode == "jpeg":
        try:
            with Image.open(src) as img:
                img = ImageOps.exif_transpose(img)
                if longest_edge and longest_edge > 0:
                    img.thumbnail((longest_edge, longest_edge),
                                  Image.Resampling.LANCZOS)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                dest = plan_dest(dest_dir, base_name, ".jpg")
                exif = img.info.get("exif") or b""
                save_kwargs = {"quality": int(jpeg_quality)}
                if exif:
                    save_kwargs["exif"] = exif
                img.save(dest, "JPEG", **save_kwargs)
                decoded_jpeg = True
        except Exception as exc:  # 解码失败保底：原样复制
            warnings.append(f"{src.name}: 无法转码（{exc}），已按原图复制")
            decoded_jpeg = False

    if not decoded_jpeg:
        dest = plan_dest(dest_dir, base_name, src.suffix.lower())
        shutil.copy2(src, dest)

    xmp_path = None
    if write_xmp and not is_video:
        xmp_path = dest.with_suffix(".xmp")
        try:
            _write_xmp_sidecar(xmp_path, list(keywords or []))
        except OSError as exc:
            warnings.append(f"{xmp_path.name}: XMP 写入失败（{exc}）")
            xmp_path = None

    return {"dest": str(dest), "xmp": str(xmp_path) if xmp_path else None,
            "warnings": warnings,
            "project_name": project_name or "",
            "date": date.today().isoformat()}
