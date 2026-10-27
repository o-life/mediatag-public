"""photos_importer 纯逻辑测试（不触 PhotoKit 授权/网络）。

背景：原「从照片导入」直接钻照片库 bundle 的 originals/ 目录挑 UUID 文件，
扩展名必然错乱（用户实测 .mov/.jpg 保存报错）。v0 改为 PhotoKit 只读导出；
本文件覆盖其中的纯逻辑：扩展名映射、重名避让、资源挑选、文件名消毒。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.photos_importer import (  # noqa: E402
    _ext_for,
    _pick_resource,
    _sanitize_filename,
    _unique_path,
)


class FakeResource:
    """模拟 PHAssetResource 的最小接口。"""

    def __init__(self, filename, uti, rtype):
        self._fn = filename
        self._uti = uti
        self._t = rtype

    def filename(self):
        return self._fn

    def uniformTypeIdentifier(self):
        return self._uti

    def type(self):
        return self._t


def test_ext_prefers_filename_suffix(tmp_path):
    r = FakeResource("IMG_0001.HEIC", "public.heic", 1)
    assert _ext_for(r) == ".heic"


def test_ext_falls_back_to_uti(tmp_path):
    # iCloud/历史素材 filename 可能为空 → UTI 映射兜底
    assert _ext_for(FakeResource("", "com.apple.quicktime-movie", 2)) == ".mov"
    assert _ext_for(FakeResource("", "public.mpeg-4", 2)) == ".mp4"
    # 完全未知 UTI → .bin 兜底，不再产生无扩展名文件
    assert _ext_for(FakeResource("x", "unknown.uti", 1)) == ".bin"


def test_unique_path_avoids_collision(tmp_path):
    (tmp_path / "a.mov").write_bytes(b"x")
    p1 = _unique_path(tmp_path, "a.mov")
    assert p1.name == "a_1.mov"
    p1.write_bytes(b"y")
    assert _unique_path(tmp_path, "a.mov").name == "a_2.mov"


def test_pick_resource_prefers_current_version():
    # 决策点 2：.photo(1)/.video(2) 资源即当前（已编辑）版本
    original = FakeResource("orig.jpg", "public.jpeg", 5)  # alternatePhoto
    edited = FakeResource("edit.jpg", "public.jpeg", 1)  # photo
    video = FakeResource("v.mov", "com.apple.quicktime-movie", 2)
    assert _pick_resource([original, edited], (1,)) is edited
    assert _pick_resource([video], (2,)) is video
    assert _pick_resource([original], (2,)) is None


def test_sanitize_filename_strips_invalid():
    assert _sanitize_filename('a/b:c*.mov') == "a_b_c_.mov"
    assert _sanitize_filename("   ") == "asset"
    assert _sanitize_filename("正常名.jpg") == "正常名.jpg"
