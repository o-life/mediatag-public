"""自适应清晰度下限（face-quality-filter）回归测试。

背景（2026-09-17 实测 /Volumes/MediaData/iphone/11月5日分享-一只牛蛙（下）.mov）:
该片 1280x590、425s，画面质量评分 0.80、预检通过，AuraFace 在 60 帧里检出 61 个
人脸框，但固定下限 min_sharpness_absolute=50 把其中 85% 丢弃（只剩 8 张），
余下太少聚不成簇 -> 0 人物 -> 整片被记为 face_scan_status='none'（"无人脸"）。

固定阈值对内容/尺寸敏感：Laplacian 方差随人脸裁剪尺寸与纹理变化，实测入库
32,991 张人脸 sharpness 跨度 9.3~8539（p5=25.5 / p50=115.4）。同一个 50 在总体
只淘汰 ~11%，在本片淘汰 85%。故下限改为以"本视频自身人脸分布的低分位"为基准，
仅放宽、不收紧，并以 min_sharpness_absolute_low 兜底。
"""


from src.core.face_recognition import FaceRecognizer

# 该片 60 帧检出、并通过尺寸过滤(w,h>=50)的 52 张人脸的真实 sharpness 分布。
REAL_SOFT_VIDEO_SHARPNESS = [
    4.9, 5.4, 5.8, 6.6, 7.2, 7.5, 8.1, 9.9, 10.2, 10.8, 10.8, 12.1, 12.5,
    12.9, 13.4, 13.5, 15.0, 16.4, 16.6, 16.7, 20.1, 20.5, 23.5, 23.6, 24.2,
    26.2, 26.3, 27.6, 28.7, 28.8, 28.9, 29.0, 31.7, 31.9, 32.4, 38.6, 39.1,
    39.4, 40.6, 41.9, 42.6, 45.9, 48.3, 49.9, 53.6, 61.0, 80.2, 89.4, 93.0,
    93.3, 120.1, 166.3,
]

FQC = {
    "enabled": True,
    "min_sharpness_absolute": 50,
    "min_sharpness_absolute_low": 8,
    "adaptive_enabled": True,
    "adaptive_percentile": 30,
    "adaptive_min_faces": 5,
}


def _faces(values):
    return [{"sharpness": float(v)} for v in values]


def _kept(values, floor):
    return sum(1 for v in values if v >= floor)


class TestAdaptiveSharpnessFloor:
    def test_soft_video_is_not_declared_faceless(self):
        """回归本片：自适应下限必须明显低于 50，保留足够多的人脸进入聚类。"""
        floor = FaceRecognizer._adaptive_sharpness_floor(
            FQC, _faces(REAL_SOFT_VIDEO_SHARPNESS)
        )
        assert 8 < floor < 50
        # 至少保留一半人脸（固定 50 只留下 8/52）
        kept = _kept(REAL_SOFT_VIDEO_SHARPNESS, floor)
        assert kept >= len(REAL_SOFT_VIDEO_SHARPNESS) / 2
        assert kept >= 3 * _kept(REAL_SOFT_VIDEO_SHARPNESS, 50)

    def test_good_video_keeps_absolute_threshold(self):
        """清晰片源不得被改动行为：下限仍为 min_sharpness_absolute。"""
        sharp = [120.0, 180.0, 200.0, 260.0, 300.0, 420.0, 500.0, 640.0]
        assert FaceRecognizer._adaptive_sharpness_floor(FQC, _faces(sharp)) == 50.0

    def test_only_relaxes_never_tightens(self):
        """下限只能放宽到绝对上限以内，绝不能高于它。"""
        for values in ([1000.0] * 20, [0.5] * 20):
            floor = FaceRecognizer._adaptive_sharpness_floor(FQC, _faces(values))
            assert floor <= FQC["min_sharpness_absolute"]

    def test_never_below_junk_floor(self):
        """全部是垃圾帧时不得把下限压到 min_sharpness_absolute_low 以下。"""
        floor = FaceRecognizer._adaptive_sharpness_floor(FQC, _faces([1.0] * 30))
        assert floor == FQC["min_sharpness_absolute_low"]

    def test_too_few_faces_falls_back_to_absolute(self):
        """人脸样本太少时不做自适应（小样本分位数不可信）。"""
        assert FaceRecognizer._adaptive_sharpness_floor(FQC, _faces([4.0] * 4)) == 50.0

    def test_disabled_falls_back_to_absolute(self):
        fqc = dict(FQC, adaptive_enabled=False)
        assert FaceRecognizer._adaptive_sharpness_floor(
            fqc, _faces(REAL_SOFT_VIDEO_SHARPNESS)
        ) == 50.0

    def test_empty_or_missing_sharpness_is_safe(self):
        assert FaceRecognizer._adaptive_sharpness_floor(FQC, []) == 50.0
        assert FaceRecognizer._adaptive_sharpness_floor(FQC, [{}] * 50) == 50.0
