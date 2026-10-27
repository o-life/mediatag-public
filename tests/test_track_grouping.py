"""轨迹分组（`group_faces_into_tracks`）测试。

背景（见 docs/架构/人脸识别-验证样本登记.md 的验收表）：
采样加密后脸数从 15 涨到 54，固定阈值下能形成的簇也变多 —— 实测**从正确的 2 个人物碎成 6 个**。
轨迹分组的目的是：先把"时间连续 + 位置重合"的脸归并为轨迹、每轨只留少量代表脸，
再按代表脸聚类（点数变少 → 压住"碎"；代表脸取自同一段 → embedding 更稳）。
"""

import numpy as np

from src.core.face_clustering import FaceClusterer, group_faces_into_tracks


def face(frame_id, x=100, y=100, w=80, h=80, quality=1.0, emb=None, with_bbox=True):
    f = {
        "frame_id": frame_id,
        "embedding": np.asarray(emb if emb is not None else [1.0, 0.0, 0.0], dtype=float),
        "quality_score": quality,
        "det_score": quality,
    }
    if with_bbox:
        f["bbox"] = [x, y, x + w, y + h]
    return f


class TestBasicGrouping:
    def test_contiguous_same_person_is_one_track(self):
        faces = [face(i * 50) for i in range(10)]
        kept = group_faces_into_tracks(faces, max_gap_frames=100, keep_per_track=2)
        assert len(kept) == 2                       # 10 张 → 1 条轨迹 → 2 张代表
        assert {f["track_id"] for f in kept} == {0}
        assert all(f["track_size"] == 10 for f in kept)

    def test_gap_beyond_window_splits_tracks(self):
        faces = [face(0), face(50), face(50 + 5000), face(100 + 5000)]
        kept = group_faces_into_tracks(faces, max_gap_frames=100, keep_per_track=5)
        assert {f["track_id"] for f in kept} == {0, 1}
        assert len(kept) == 4

    def test_different_embedding_splits_tracks(self):
        """同一帧里两个人（特征距离 1.0 > 0.55）→ 两条轨迹。判据是特征，不是位置。"""
        other = [0.0, 1.0, 0.0]
        faces = [face(0), face(0, emb=other), face(50), face(50, emb=other)]
        kept = group_faces_into_tracks(faces, max_gap_frames=100, keep_per_track=5)
        assert len({f["track_id"] for f in kept}) == 2
        t0 = sorted(f["frame_id"] for f in kept if f["track_id"] == 0)
        t1 = sorted(f["frame_id"] for f in kept if f["track_id"] == 1)
        assert t0 == [0, 50] and t1 == [0, 50]

    def test_position_does_not_matter(self):
        """位置完全不同（远距离）但特征相同 → 仍是同一条轨迹（判据只看时间+特征）。"""
        faces = [face(0, x=0), face(50, x=2000)]
        assert len({f["track_id"] for f in group_faces_into_tracks(faces, max_gap_frames=100,
                                                                  keep_per_track=5)}) == 1

    def test_empty_input(self):
        assert group_faces_into_tracks([]) == []


class TestAdaptiveWindow:
    def test_window_derived_from_sampling_gap(self):
        """未显式给窗口时，按输入自身的采样间隔中位数 × 系数推导（随采样密度自适应）。"""
        dense = [face(i * 50) for i in range(6)]      # 间隔 50 帧 → 窗口 75
        kept = group_faces_into_tracks(dense, keep_per_track=2)
        assert len(kept) == 2                          # 全部落在同一条轨迹

        sparse = [face(0), face(50), face(100), face(576)]
        # 间隔为 [50, 50, 476] → 中位 50 → 窗口 75；最后一段 476 远超窗口 → 断开
        tracks = {}
        for f in group_faces_into_tracks(sparse, keep_per_track=5):
            tracks.setdefault(f["track_id"], []).append(f["frame_id"])
        assert sorted(tracks.values()) == [[0, 50, 100], [576]]

    def test_explicit_window_overrides(self):
        faces = [face(0), face(50), face(1000)]
        assert len({f["track_id"] for f in group_faces_into_tracks(faces, max_gap_frames=10000,
                                                                   keep_per_track=5)}) == 1
        assert len({f["track_id"] for f in group_faces_into_tracks(faces, max_gap_frames=10,
                                                                   keep_per_track=5)}) == 3


class TestRepresentatives:
    def test_picks_highest_quality(self):
        faces = [face(0, quality=0.1), face(50, quality=0.9), face(100, quality=0.5)]
        kept = group_faces_into_tracks(faces, max_gap_frames=200, keep_per_track=1)
        assert len(kept) == 1 and kept[0]["quality_score"] == 0.9

    def test_result_sorted_by_time(self):
        faces = [face(100, quality=0.9), face(0, quality=0.1), face(50, quality=0.5)]
        kept = group_faces_into_tracks(faces, max_gap_frames=200, keep_per_track=3)
        assert [f["frame_id"] for f in kept] == [0, 50, 100]

    def test_keep_at_least_one(self):
        faces = [face(0), face(50)]
        assert len(group_faces_into_tracks(faces, max_gap_frames=200, keep_per_track=0)) == 1


class TestRobustness:
    def test_missing_frame_id_each_own_track(self):
        faces = [{"embedding": np.array([1.0, 0.0]), "bbox": [0, 0, 10, 10], "det_score": 1.0},
                 {"embedding": np.array([1.0, 0.0]), "bbox": [0, 0, 10, 10], "det_score": 1.0}]
        kept = group_faces_into_tracks(faces, keep_per_track=5)
        assert len({f["track_id"] for f in kept}) == 2

    def test_missing_embedding_each_own_track(self):
        faces = [face(0, emb=None), face(50, emb=None)]
        for f in faces:
            f.pop("embedding", None)
        kept = group_faces_into_tracks(faces, max_gap_frames=500, keep_per_track=5)
        assert len({f["track_id"] for f in kept}) == 2

    def test_does_not_mutate_input(self):
        faces = [face(0), face(50)]
        group_faces_into_tracks(faces, max_gap_frames=200)
        assert "track_id" not in faces[0]


class TestClustererIntegration:
    """接进 FaceClusterer 后必须真的减少参与聚类的点数。"""

    def _faces(self, n, jitter=0.01):
        rng = np.random.default_rng(0)
        base = np.array([1.0, 0.0, 0.0])
        out = []
        for i in range(n):
            emb = base + rng.normal(0, jitter, 3)
            out.append(face(i * 50, emb=emb, quality=1.0))
        return out

    def test_disabled_by_default_changes_nothing(self):
        clusterer = FaceClusterer(min_samples=2)
        clusters = clusterer.cluster_faces(self._faces(10))
        assert sum(len(c) for c in clusters) == 10        # 全部参与

    def test_enabled_reduces_points(self):
        clusterer = FaceClusterer(min_samples=2, track_config={"enabled": True, "keep_per_track": 2})
        clusters = clusterer.cluster_faces(self._faces(10))
        assert sum(len(c) for c in clusters) == 2         # 10 张 → 1 条轨迹 → 2 张代表
        assert len(clusters) == 1

    def test_two_people_stay_two_clusters(self):
        """两人各一条轨迹 → 仍是 2 簇（分组不能把不同的人并起来）。"""
        rng = np.random.default_rng(1)
        left = self._faces(6)
        right = []
        for i in range(6):
            emb = np.array([0.0, 1.0, 0.0]) + rng.normal(0, 0.01, 3)
            right.append(face(i * 50, x=800, emb=emb))
        clusterer = FaceClusterer(min_samples=2, track_config={"enabled": True, "keep_per_track": 3})
        clusters = clusterer.cluster_faces(left + right)
        assert len(clusters) == 2
