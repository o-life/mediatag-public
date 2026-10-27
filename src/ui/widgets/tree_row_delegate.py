"""TreeCountDelegate — 统一树行委托（18 号规格 §3.3-3.7 的单点实现）。

规格：行高 28 / 圆角 6 / 内边距 L10 R10 / 名称 12.5/400/txt / 计数 11/400
右对齐（与名称间距 8）/ hover=底 hover / 选中=accent 半透底 + 名称 600 +
计数 accent + 左缘指示线 2×(行高-8) 圆角 1。

数据约定：item 文字=名称；data(UserRole+1)=计数（int/None，None 不绘制）。
任务页来源树/项目树与详情页项目树共用，保证同一侧栏里两树观感一致。
"""

import re

from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

try:
    from ..components.theme import theme
    from ..tokens import T
except ImportError:  # 脚本态
    from ui.components.theme import theme
    from ui.tokens import T


def _css_color(spec: str, fallback: QColor) -> QColor:
    """把 'rgba(255,255,255,.06)' / '#RRGGBB' 解析成 QColor（失败回退）。"""
    spec = (spec or "").strip()
    m = re.match(
        r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)", spec)
    if m:
        c = QColor(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if m.group(4) is not None:
            c.setAlphaF(float(m.group(4)))
        return c
    c = QColor(spec)
    return c if c.isValid() else QColor(fallback)


def _is_dark_theme() -> bool:
    bg = theme().get("bg_primary", "#1C1C1E")
    c = QColor(bg)
    return c.isValid() and c.lightness() < 128


class TreeCountDelegate(QStyledItemDelegate):
    """树行绘制：名称左 / 计数右，选中半透底+指示线（规格见模块 docstring）。"""

    ROW_HEIGHT = T.ROW_H_TREE

    # 选中底透明度：深色 .18 / 浅色 .14（预览-16/17 定稿值）
    SEL_ALPHA_DARK = 46   # 0.18 × 255
    SEL_ALPHA_LIGHT = 36  # 0.14 × 255

    def sizeHint(self, option, index):  # noqa: N802 - Qt 命名
        width = option.rect.width() if option is not None else 0
        return QSize(width, self.ROW_HEIGHT)

    def paint(self, painter, option, index):
        t = theme()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(option.rect)
        # 行框左右各收 2px：圆角不贴栏缘（预览稿 ni 行与栏壁的呼吸）
        rect.adjust(T.TREE_ROW_INSET, 0, -T.TREE_ROW_INSET, 0)
        if rect.height() > self.ROW_HEIGHT:
            rect.setHeight(self.ROW_HEIGHT)

        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        if selected:
            bg = QColor(t.get("accent"))
            bg.setAlpha(self.SEL_ALPHA_DARK if _is_dark_theme()
                        else self.SEL_ALPHA_LIGHT)
        elif hovered:
            bg = _css_color(t.get("hover"), QColor(255, 255, 255, 15))
        else:
            bg = None
        if bg is not None:
            path = QPainterPath()
            path.addRoundedRect(rect, T.RADIUS_CTRL, T.RADIUS_CTRL)
            painter.fillPath(path, bg)

        # 左缘指示线（仅选中）：2 宽、行高-8、圆角 1、accent
        if selected:
            ind = QRectF(rect.left() + T.TREE_IND_X, rect.top() + T.TREE_IND_VPAD,
                         T.TREE_IND_W, rect.height() - 2 * T.TREE_IND_VPAD)
            ipath = QPainterPath()
            ipath.addRoundedRect(ind, T.TREE_IND_RADIUS, T.TREE_IND_RADIUS)
            painter.fillPath(ipath, QColor(t.get("accent")))

        name = index.data(Qt.ItemDataRole.DisplayRole) or ""
        count = index.data(Qt.ItemDataRole.UserRole + 1)

        # 名称 12.5px（选中 600；QFont 像素字号取整，QSS 侧保 12.5）
        f = QFont(option.font)
        f.setPixelSize(round(T.FONT_SMALL))
        f.setWeight(QFont.Weight.DemiBold if selected else QFont.Weight.Normal)
        painter.setFont(f)
        painter.setPen(QColor(t.get("text_primary")))
        _pad = T.TREE_PAD_X - T.TREE_ROW_INSET  # 行内文字左 8（框已收 2，合成 L10）
        text_rect = rect.adjusted(_pad, 0, -_pad, 0)
        fm = painter.fontMetrics()
        elided = fm.elidedText(name, Qt.TextElideMode.ElideRight,
                               int(text_rect.width()))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, elided)

        # 计数 11px：右对齐，右内边距 10；选中=accent，常态=txt2
        if count is not None:
            cf = QFont(option.font)
            cf.setPixelSize(T.FONT_HINT)
            cf.setWeight(QFont.Weight.DemiBold if selected
                         else QFont.Weight.Normal)
            painter.setFont(cf)
            painter.setPen(QColor(t.get("accent") if selected
                                  else t.get("text_secondary")))
            painter.drawText(
                rect.adjusted(0, 0, -_pad, 0),  # 计数右缘距行框 8（+框收 2 = 10）
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                str(count))

        painter.restore()
