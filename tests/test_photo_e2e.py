"""照片端到端测试（Phase 2/7）。

验证照片导入 → 分析 → XMP sidecar 生成。
依赖 API 在 localhost:8000 运行。
"""
import os
import shutil
import tempfile
import time

import pytest
import requests

API_BASE = "http://localhost:8000"


def _wait_for_api(timeout=10):
    for _ in range(timeout):
        try:
            r = requests.get(f"{API_BASE}/status", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# 依赖真实 API 服务的集成测试：服务未启动时跳过（避免普通 pytest 必失败）
pytestmark = pytest.mark.skipif(
    not _wait_for_api(timeout=3),
    reason="需要本地 API 服务运行在 localhost:8000（./start-api.sh）",
)


def _find_test_photo():
    """找一张测试照片。"""
    candidates = [
        "/tmp/xmp_test2.jpg",
        "/tmp/theme_test_v2.jpg",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # 从人脸目录找
    for root, dirs, files in os.walk("/Volumes/MediaData/actor_faces_cross_video_test"):
        for f in files:
            if f.lower().endswith(".jpg"):
                return os.path.join(root, f)
    return None


class TestPhotoEndToEnd:
    """照片全链路测试。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        assert _wait_for_api(), "API not available at localhost:8000"
        self.photo = _find_test_photo()
        assert self.photo, "无测试照片可用"

        # 复制到临时文件（避免重复导入）
        self.tmp_photo = tempfile.mktemp(suffix=".jpg")
        shutil.copy(self.photo, self.tmp_photo)
        yield
        try:
            os.remove(self.tmp_photo)
        except Exception:
            pass

    def test_import_photo(self):
        """导入照片，应创建 media_type=photo 的 entity。"""
        r = requests.post(f"{API_BASE}/task/import", json={"files": [self.tmp_photo]}, timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success", data

        # 用刚导入文件的 MD5 精确查 entity（不依赖"最后一条"）
        import hashlib
        import sqlite3
        with open(self.tmp_photo, "rb") as f:
            file_md5 = hashlib.md5(f.read()).hexdigest()
        db = "/Users/demo/Library/Application Support/MediaTag/db/audit_cache.db"
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(
            "SELECT media_type FROM file_entities WHERE file_md5 = ?",
            (file_md5,),
        )
        row = cur.fetchone()
        conn.close()
        assert row and row[0] == "photo", f"media_type 应为 photo, got {row}"

    def test_photo_in_videos_list(self):
        """照片应在 /videos 列表中出现且带 media_type。"""
        r = requests.get(f"{API_BASE}/videos", timeout=10)
        assert r.status_code == 200
        photos = [v for v in r.json() if v.get("media_type") == "photo"]
        assert len(photos) > 0, "作品列表应有照片"
