"""geometry.py — 位置契约断言 API（Qt Widgets 通用，零业务依赖）。

业内对标（2026-09-28 调研）：视觉回归工具链（Playwright snapshots、
Qt TestCase.grabImage/compare）的共识是——几何断言比裸黄金截图更稳，
黄金截图做补充。本模块提供"位置契约"：把设计稿里「A 与 B 的关系」
写成机器可验的断言，D7（缝隙）/错位类漂移在提交前必红。

用法（项目侧契约清单）：
    from ui.control.geometry import Geo

    bar = Geo(page.action_bar, "操作行")
    panel = Geo(page.media_panel, "素材面板")
    bar.bottom_adjacent(panel.top, gap=0, spec="18§4.1 操作行贴面板")

所有断言失败消息自带「实际值 vs 期望值 + 规格引用」，探针输出可直接
贴进缺陷台账。
"""

from __future__ import annotations

from PySide6.QtWidgets import QWidget


class Geo:
    """QWidget 的几何视图 + 关系断言。只读，不改控件。"""

    def __init__(self, widget: QWidget, name: str = ""):
        if widget is None:
            raise ValueError(f"Geo({name!r}): 控件为 None——契约目标不存在")
        self.w = widget
        self.name = name or type(widget).__name__

    # ── 自身几何 ─────────────────────────────────────────────
    @property
    def rect(self):
        return self.w.geometry()

    @property
    def top(self) -> int:
        return self.w.geometry().top()

    @property
    def bottom(self) -> int:
        return self.w.geometry().bottom()

    @property
    def left(self) -> int:
        return self.w.geometry().left()

    @property
    def right(self) -> int:
        return self.w.geometry().right()

    @property
    def width(self) -> int:
        return self.w.width()

    @property
    def height(self) -> int:
        return self.w.height()

    # ── 关系断言（同一坐标系内使用：先确保两者共父或用 map 接算）──

    def _expect(self, actual, expect, desc: str, spec: str) -> None:
        assert actual == expect, (
            f"[位置契约] {self.name}.{desc}: 实际 {actual} ≠ 期望 {expect}"
            + (f"（{spec}）" if spec else ""))

    def bottom_adjacent(self, other_top: int, gap: int = 0, spec: str = "") -> None:
        """本元素底边与某顶边相距 gap（贴边=0）——D7 类缝隙的主断言。"""
        actual = other_top - self.bottom - 1
        self._expect(actual, gap, "底边→下方元素顶边间距", spec)

    def right_adjacent(self, other_left: int, gap: int = 0, spec: str = "") -> None:
        """本元素右缘与某左缘水平间距 gap。"""
        actual = other_left - self.right - 1
        self._expect(actual, gap, "右缘→右侧元素左缘间距", spec)

    def top_aligns(self, other_top: int, tolerance: int = 0, spec: str = "") -> None:
        actual = self.top
        assert abs(actual - other_top) <= tolerance, (
            f"[位置契约] {self.name}.top: 实际 {actual} ≠ {other_top}"
            f"（±{tolerance}）" + (f"（{spec}）" if spec else ""))

    def left_aligns(self, other_left: int, tolerance: int = 0, spec: str = "") -> None:
        actual = self.left
        assert abs(actual - other_left) <= tolerance, (
            f"[位置契约] {self.name}.left: 实际 {actual} ≠ {other_left}"
            f"（±{tolerance}）" + (f"（{spec}）" if spec else ""))

    def inside(self, container: QWidget, margin: int | None = None,
               spec: str = "") -> None:
        """本元素完整落在 container 内；给 margin 则四边距恰好=margin
        （左右贴边满铺=margin 0，D4.8 直角满铺的主断言）。"""
        cl, ct = container.geometry().left(), container.geometry().top()
        cr, cb = container.geometry().right(), container.geometry().bottom()
        ok = (self.left >= cl and self.top >= ct
              and self.right <= cr and self.bottom <= cb)
        assert ok, (
            f"[位置契约] {self.name} 超出容器 {type(container).__name__}: "
            f"元素({self.left},{self.top},{self.right},{self.bottom}) "
            f"容器({cl},{ct},{cr},{cb})" + (f"（{spec}）" if spec else ""))
        if margin is not None:
            for edge, actual, expect in (
                ("left", self.left - cl, margin),
                ("top", self.top - ct, margin),
                ("right", cr - self.right, margin),
                ("bottom", cb - self.bottom, margin),
            ):
                self._expect(actual, expect, f"距容器{edge}", spec)

    def width_at_least(self, minimum: int, spec: str = "") -> None:
        self._expect(min(self.width, minimum), minimum, "最小宽度", spec)
