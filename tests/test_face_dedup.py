"""BUG-023 修复的回归测试。

覆盖两处修复：
1. 帧预筛三类互斥（frontal 优先 > side > empty）—— 此前"同一帧可能同时出现在
   frontal 和 side"，导致正脸/侧脸两个阶段各写一次同一张脸。
2. `_dedup_faces` —— create_actor / update_actor 写入前按 (帧名, bbox) 去重。
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.face_recognition import FaceRecognizer


# ───────────────────────────────────────────────────────────
# _dedup_faces
# ───────────────────────────────────────────────────────────

def _face(frame_path, bbox, quality=0.5):
    return {"frame_path": frame_path, "bbox": bbox, "quality_score": quality}


def test_dedup_empty():
    assert FaceRecognizer._dedup_faces([]) == []


def test_dedup_no_duplicate():
    faces = [
        _face("a/frame_1.jpg", [0, 0, 10, 10]),
        _face("a/frame_2.jpg", [0, 0, 10, 10]),   # 不同帧 -> 不去重
        _face("a/frame_1.jpg", [20, 20, 30, 30]),  # 同帧不同框 -> 不去重
    ]
    assert len(FaceRecognizer._dedup_faces(faces)) == 3


def test_dedup_same_frame_same_bbox():
    faces = [
        _face("a/frame_1.jpg", [0, 0, 10, 10], 0.5),
        _face("a/frame_1.jpg", [0, 0, 10, 10], 0.9),  # 完全重复
    ]
    assert len(FaceRecognizer._dedup_faces(faces)) == 1


def test_dedup_keeps_higher_quality():
    faces = [
        _face("a/frame_1.jpg", [0, 0, 10, 10], 0.3),
        _face("a/frame_1.jpg", [0, 0, 10, 10], 0.8),
    ]
    out = FaceRecognizer._dedup_faces(faces)
    assert out[0]["quality_score"] == 0.8


def test_dedup_uses_frame_name_only():
    """帧目录名不同但帧文件名相同 -> 视为同帧（帧目录因任务私有目录而异）。"""
    faces = [
        _face("/tmp/task1/frames/frame_5.jpg", [0, 0, 10, 10]),
        _face("/tmp/task2/frames/frame_5.jpg", [0, 0, 10, 10]),
    ]
    assert len(FaceRecognizer._dedup_faces(faces)) == 1


def test_dedup_falls_back_to_det_score():
    """quality_score 缺失时用 det_score 比较，保留更高者。"""
    faces = [
        {"frame_path": "a/frame_1.jpg", "bbox": [0, 0, 10, 10], "det_score": 0.4},
        {"frame_path": "a/frame_1.jpg", "bbox": [0, 0, 10, 10], "det_score": 0.9},
    ]
    out = FaceRecognizer._dedup_faces(faces)
    assert len(out) == 1
    assert out[0]["det_score"] == 0.9


# ───────────────────────────────────────────────────────────
# 帧预筛互斥
# ───────────────────────────────────────────────────────────

class _FakeFace:
    def __init__(self, kps, bbox):
        self.kps = np.asarray(kps, dtype=np.float32)
        self.bbox = np.asarray(bbox, dtype=np.float32)


class _FakeGate:
    """按图像均值就近匹配返回预设人脸（用均值区分不同帧）。"""

    def __init__(self, faces_by_mean):
        self._faces = faces_by_mean

    def get(self, img):
        m = float(img.mean())
        key = min(self._faces.keys(), key=lambda k: abs(k - m))
        return self._faces[key]


# 正脸：鼻尖在双眼中点正下方 -> yaw≈0；侧脸：鼻尖大幅偏离 -> yaw 很大
_FRONTAL_KPS = [[40, 50], [72, 50], [56, 73], [44, 80], [68, 80]]
_SIDE_KPS = [[40, 50], [72, 50], [120, 73], [44, 80], [68, 80]]
_BBOX = [20, 30, 92, 100]


def _run_prescreen(faces_by_mean, tmp_path):
    d = Path(tmp_path)
    for mean in (10, 60, 110, 200):
        img = np.full((120, 160, 3), mean, dtype=np.uint8)
        cv2.imwrite(str(d / f"frame_{mean}.jpg"), img)
    r = object.__new__(FaceRecognizer)      # 跳过 __init__，避免加载模型
    r._gate_app = _FakeGate(faces_by_mean)  # 懒加载直接命中
    return r._prescreen_frames(str(d))


def test_prescreen_mixed_frame_is_frontal_only(tmp_path):
    """一帧同时有正脸和侧脸时只归入 frontal（互斥），不再两边都进。"""
    faces_by_mean = {
        10: [_FakeFace(_FRONTAL_KPS, _BBOX)],                             # 正脸帧
        60: [_FakeFace(_SIDE_KPS, _BBOX)],                                # 侧脸帧
        110: [_FakeFace(_FRONTAL_KPS, _BBOX), _FakeFace(_SIDE_KPS, _BBOX)],  # 混帧
        200: [],                                                          # 无人脸
    }
    res = _run_prescreen(faces_by_mean, tmp_path)

    assert len(res["frontal"]) == 2, "正脸帧 + 混帧应归入 frontal"
    assert len(res["side"]) == 1, "仅侧脸帧归入 side"
    assert len(res["empty"]) == 1, "无人脸帧归入 empty"


def test_prescreen_classes_disjoint(tmp_path):
    """三类互斥：没有任何帧同时出现在两个列表里。"""
    faces_by_mean = {
        10: [_FakeFace(_FRONTAL_KPS, _BBOX), _FakeFace(_SIDE_KPS, _BBOX)],
        60: [_FakeFace(_SIDE_KPS, _BBOX)],
        110: [_FakeFace(_FRONTAL_KPS, _BBOX)],
        200: [],
    }
    res = _run_prescreen(faces_by_mean, tmp_path)

    assert not (set(res["frontal"]) & set(res["side"]))
    assert not (set(res["frontal"]) & set(res["empty"]))
    assert not (set(res["side"]) & set(res["empty"]))
    total = len(res["frontal"]) + len(res["side"]) + len(res["empty"])
    assert total == 4, "每帧恰好归入一类"