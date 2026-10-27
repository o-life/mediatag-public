"""golden.py — 黄金截图回归（Qt Widgets 通用，零业务依赖）。

业内对标：Playwright toHaveScreenshot / Percy / Chromatic 的标准流——
基线 PNG 入库 → 每次渲染与基线 diff（容差内放行）→ 漂移输出热图 →
有意改动经显式 update 重新审批。Qt 官方 TestCase 自带 grabImage/compare
同构机制（调研 2026-09-28）。

用法（测试侧）：
    from ui.control.golden import assert_golden

    def test_detail_grid_golden(qapp):
        page = ...  # 构造+固定数据
        assert_golden(page, "detail_grid", threshold_ratio=0.001)

基线更新（有意改版后）：
    UI_GOLDEN_UPDATE=1 VT_UI_TESTS=1 pytest tests/test_golden_screenshots.py
    → 新基线生成，肉眼过目后随改动一起提交（=审批）。

同机确定性：offscreen 软件渲染 + 固定窗口尺寸 + 固定假数据 → 像素级
稳定；threshold_ratio 只吸收极端抗锯齿抖动，不做语义容差。
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtWidgets import QWidget

# 基线与差异产物目录（随仓库走，跨项目可换名）
BASELINE_DIR = Path(__file__).resolve().parents[3] / "tests" / "golden_baselines"
DIFF_DIR = Path(__file__).resolve().parents[3] / "test-artifacts" / "golden_diff"

UPDATE_ENV = "UI_GOLDEN_UPDATE"


def _render(widget: QWidget) -> QImage:
    widget.grab()
    return widget.grab().toImage()


def _diff_stats(a: QImage, b: QImage) -> tuple[int, int, QImage | None]:
    """返回 (不同像素数, 总像素数, 差异热图)。尺寸不同视为全不同。"""
    if a.size() != b.size():
        return a.width() * a.height(), max(1, a.width() * a.height()), None
    total = a.width() * a.height()
    diff = QImage(a.size(), QImage.Format.Format_ARGB32)
    diff.fill(0)
    changed = 0
    for y in range(a.height()):
        for x in range(a.width()):
            pa, pb = a.pixelColor(x, y), b.pixelColor(x, y)
            if pa != pb:
                changed += 1
                # 热图：漂移点标红，其余压暗
                diff.setPixelColor(x, y, QColor(255, 48, 58))
    return changed, total, diff


def assert_golden(widget: QWidget, name: str, threshold_ratio: float = 0.001,
                  baselines_dir: Path | None = None) -> None:
    """widget 渲染与基线 PNG 对比；漂移超阈值即断言失败并落热图。

    UI_GOLDEN_UPDATE=1 时改为生成/覆盖基线（=审批动作，须随代码提交）。
    """
    directory = Path(baselines_dir) if baselines_dir else BASELINE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    baseline_path = directory / f"{name}.png"

    img = _render(widget)

    if os.environ.get(UPDATE_ENV) == "1" or not baseline_path.exists():
        img.save(str(baseline_path))
        print(f"[golden] {'更新' if baseline_path.exists() else '生成'}基线 "
              f"{baseline_path.name} ({img.width()}x{img.height()})", flush=True)
        return

    baseline = QImage(str(baseline_path))
    changed, total, heatmap = _diff_stats(baseline, img)
    ratio = changed / max(1, total)
    if ratio <= threshold_ratio:
        return

    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    if heatmap is not None:
        heatmap_path = DIFF_DIR / f"{name}_diff.png"
        painter = QPainter(heatmap)
        painter.setOpacity(0.15)
        painter.drawImage(0, 0, img)
        painter.end()
        heatmap.save(str(heatmap_path))
        hint = f"（热图 {heatmap_path}）"
    else:
        hint = f"（尺寸 {baseline.width()}x{baseline.height()} → {img.width()}x{img.height()}）"
    raise AssertionError(
        f"[golden] {name} 漂移 {changed}/{total} 像素（{ratio:.4%} > "
        f"{threshold_ratio:.2%}）{hint}；确认改动有意后 "
        f"UI_GOLDEN_UPDATE=1 重跑并提交新基线")
