"""垃圾帧判定(标签质量闸 3):黑帧/糊帧/废帧在送 VLM 与入库前剔除。

背景(2026-09-30 审计):抽帧全链原本无黑帧过滤,预筛 0 检出回退时甚至
"取目录前 N 帧"直送 VLM,导致纯黑转场帧被打上内容标签(实证:"二次元"
标签的证据帧为纯黑帧)。

判定口径(宁松勿严:只拦"明显无信息"的帧,避免误杀正常暗场景):
- 黑帧: 灰度均值 < black_mean(默认 16, 0-255)
- 糊帧: 统一缩放到 640px 长边后 Laplacian 方差 < min_sharpness(默认 12)
- 废帧: 读不出图

阈值经 config `analysis.frame_quality_black_mean` / `_min_sharpness` 覆盖
(调用方 set_thresholds 注入);测试可直接改模块常量。
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import cv2

logger = logging.getLogger(__name__)

BLACK_MEAN = 16.0
MIN_SHARPNESS = 12.0
_LAPLACIAN_SIDE = 640  # 清晰度统一归一到该长边,阈值才有可比性


def set_thresholds(black_mean: float, min_sharpness: float) -> None:
    """管线启动时从 config 注入阈值。"""
    global BLACK_MEAN, MIN_SHARPNESS
    BLACK_MEAN = float(black_mean)
    MIN_SHARPNESS = float(min_sharpness)


def frame_issue(path) -> Optional[str]:
    """返回垃圾帧原因(None = 可用帧)。"""
    try:
        img = cv2.imread(str(path))
    except Exception:
        img = None
    if img is None:
        return "unreadable"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < BLACK_MEAN:
        return "black"
    h, w = gray.shape[:2]
    scale = min(1.0, _LAPLACIAN_SIDE / max(h, w, 1))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < MIN_SHARPNESS:
        return "blurry"
    return None


def filter_frames(paths: Sequence[str]) -> Tuple[list, list]:
    """过滤垃圾帧。返回 (可用帧列表, [(帧路径, 原因)])。"""
    good, dropped = [], []
    for p in paths:
        why = frame_issue(p)
        if why:
            dropped.append((str(p), why))
        else:
            good.append(p)
    if dropped:
        by_reason: dict[str, int] = {}
        for _, why in dropped:
            by_reason[why] = by_reason.get(why, 0) + 1
        logger.info(
            "垃圾帧过滤: %d/%d 帧被剔除 (%s)",
            len(dropped), len(dropped) + len(good),
            ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items())),
        )
    return good, dropped
