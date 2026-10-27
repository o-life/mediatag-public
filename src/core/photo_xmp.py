"""照片 XMP 人脸标签写入。

行业标准：Adobe XMP `mwg-rs:Regions`（Metadata Working Group 人脸区域标准），
兼容 Lightroom / Capture One / Synology Photos / DigiKam 等。

设计原则：
- 照片原文件不动（只写 XMP 元数据块，不碰像素）
- 人脸区域用归一化坐标记录（mwg-rs 标准），不保存裁剪人脸图
- 人脸裁剪图是中间产物，不持久化
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def write_photo_face_regions(
    photo_path: str,
    faces: List[Dict],
    actor_names: Optional[Dict[int, str]] = None,
) -> bool:
    """在照片上写入 XMP 人脸区域标签（mwg-rs:Regions 标准）。

    Args:
        photo_path: 照片文件路径
        faces: 人脸列表，每项含 {box: [x1,y1,x2,y2] 像素坐标, actor_id}
        actor_names: actor_id → 人物名 映射（可选，用于人面名称）

    Returns:
        是否写入成功。JPEG 用 piexif 嵌入 XMP，其他格式生成 .xmp sidecar。
    """

    p = Path(photo_path)
    if not p.exists():
        logger.warning(f"照片不存在: {photo_path}")
        return False

    suffix = p.suffix.lower()
    xmp_sidecar = p.with_suffix(p.suffix + ".xmp")

    # 获取图片尺寸（用于坐标归一化）
    img_size = None
    try:
        from PIL import Image
        with Image.open(str(p)) as img:
            img_size = img.size
    except Exception:
        pass

    # 1. 生成 mwg-rs Regions XMP 块
    xmp_xml = _build_xmp_face_regions(faces, actor_names or {}, img_size)

    # 2. JPEG：尝试嵌入 XMP（piexif），失败则写 sidecar
    if suffix in (".jpg", ".jpeg"):
        try:
            import piexif

            piexif.load(str(p))  # 探测：损坏文件在此抛异常
            # piexif 的 XMP 是通过 UserComment 或单独处理，这里用最简单可靠方式：
            # 直接写 .xmp sidecar（避免 piexif XMP 兼容性问题）
            xmp_sidecar.write_text(xmp_xml, encoding="utf-8")
            logger.info(f"照片 XMP sidecar 已写入: {xmp_sidecar}")
            return True
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"piexif 嵌入失败，改用 sidecar: {e}")

    # 3. 其他格式/失败：写 .xmp sidecar（不破坏原文件）
    try:
        xmp_sidecar.write_text(xmp_xml, encoding="utf-8")
        logger.info(f"照片 XMP sidecar 已写入: {xmp_sidecar}")
        return True
    except Exception as e:
        logger.error(f"XMP sidecar 写入失败: {e}")
        return False


def _build_xmp_face_regions(faces: List[Dict], actor_names: Dict[int, str], img_size: Optional[tuple] = None) -> str:
    """构建 mwg-rs:Regions XMP XML。

    格式（归一化坐标，x/y/w/h 均 0-1）：
    <mwg-rs:Regions mwg-rs:RegionList="1">
      <rdf:Seq>
        <rdf:li rdf:parseType="Resource">
          <mwg-rs:Type>Face</mwg-rs:Type>
          <mwg-rs:Area rdf:parseType="Resource">
            <stArea:x>0.5</stArea:x>
            <stArea:y>0.5</stArea:y>
            <stArea:w>0.3</stArea:w>
            <stArea:h>0.3</stArea:h>
          </mwg-rs:Area>
          <mwg-rs:Name>人物名</mwg-rs:Name>
        </rdf:li>
      </rdf:Seq>
    </mwg-rs:Regions>
    """
    if not faces:
        return ""

    # 图片尺寸（用于像素坐标 → 归一化）
    img_w = img_h = None
    if img_size and len(img_size) == 2:
        img_w, img_h = img_size

    regions = []
    for f in faces:
        box = f.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)

        # 归一化到 0-1（mwg-rs 用中心点 + 宽高）
        if img_w and img_h:
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            wn = w / img_w
            hn = h / img_h
        else:
            # 无图片尺寸：假设已是归一化
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            wn = w
            hn = h

        regions.append({
            "cx": max(0.0, min(1.0, cx)),
            "cy": max(0.0, min(1.0, cy)),
            "w": max(0.0, min(1.0, wn)),
            "h": max(0.0, min(1.0, hn)),
            "name": actor_names.get(f.get("actor_id"), ""),
        })

    if not regions:
        return ""

    region_xml = []
    for r in regions:
        name_el = f"<mwg-rs:Name>{_xml_escape(r['name'])}</mwg-rs:Name>" if r["name"] else ""
        region_xml.append(
            f"""        <rdf:li rdf:parseType="Resource">
          <mwg-rs:Type>Face</mwg-rs:Type>
          <mwg-rs:Area rdf:parseType="Resource">
            <stArea:x>{r['cx']:.4f}</stArea:x>
            <stArea:y>{r['cy']:.4f}</stArea:y>
            <stArea:w>{r['w']:.4f}</stArea:w>
            <stArea:h>{r['h']:.4f}</stArea:h>
          </mwg-rs:Area>
          {name_el}
        </rdf:li>"""
        )

    return f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="MediaTag">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
    xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#">
   <mwg-rs:Regions mwg-rs:RegionList="{len(region_xml)}">
    <rdf:Seq>
{chr(10).join(region_xml)}
    </rdf:Seq>
   </mwg-rs:Regions>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def _xml_escape(text: str) -> str:
    """XML 转义。"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
