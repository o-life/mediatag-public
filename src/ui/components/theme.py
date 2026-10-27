"""Theme system — macOS-native design tokens for AI Video Factory.

Provides a centralized theme manager with light/dark mode support,
design tokens (colors, spacing, radii, shadows), and helper methods
for applying consistent styling across all components.
"""

import sys
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtCore import QObject, QSettings, Signal


class ThemeTokens:
    """Design tokens for the current theme mode."""

    # ── Light mode tokens ──────────────────────────────────────────
    LIGHT = {
        # Backgrounds
        "bg_primary": "#FFFFFF",
        "bg_secondary": "#F5F5F7",
        "btn_glass_bg": "rgba(0, 0, 0, 0.05)",
        "btn_glass_hover": "rgba(0, 0, 0, 0.10)",
        "bg_sidebar": "#F5F5F7",
        "bg_card": "#FFFFFF",
        "bg_hover": "#E8E8ED",
        "bg_selected": "#007AFF",
        "bg_input": "#FFFFFF",

        # Text
        "text_primary": "#1D1D1F",
        "text_secondary": "#86868B",
        "text_tertiary": "#AEAEB2",
        "text_on_accent": "#FFFFFF",
        "text_link": "#007AFF",

        # Accent / brand
        "accent": "#007AFF",
        "accent_hover": "#0056CC",
        "success": "#34C759",
        "warning": "#FF9500",
        "danger": "#FF3B30",
        "danger_hover": "#E0362C",
        "info": "#5AC8FA",

        # Borders & dividers
        "border": "#D2D2D7",
        "border_light": "#E5E5EA",
        "divider": "#E5E5EA",

        # Shadows
        "shadow_color": "#000000",
        "shadow_opacity": 0.08,

        # Progress
        "progress_bg": "#E5E5EA",
        "progress_fill": "#007AFF",

        # Table
        "table_header_bg": "#F5F5F7",
        "table_alt_row": "#FAFAFA",
        "table_selected": "#E8F0FE",
    }

    # ── Dark mode tokens ──────────────────────────────────────────
    DARK = {
        "btn_glass_bg": "rgba(255, 255, 255, 0.07)",
        "btn_glass_hover": "rgba(255, 255, 255, 0.14)",
        # Backgrounds
        "bg_primary": "#1C1C1E",
        "bg_secondary": "#2C2C2E",
        "bg_sidebar": "#2C2C2E",
        "bg_card": "#38383A",
        "bg_hover": "#48484A",
        "bg_selected": "#0A84FF",
        "bg_input": "#38383A",

        # Text
        "text_primary": "#F5F5F7",
        "text_secondary": "#98989D",
        "text_tertiary": "#636366",
        "text_on_accent": "#FFFFFF",
        "text_link": "#0A84FF",

        # Accent / brand
        "accent": "#0A84FF",
        "accent_hover": "#409CFF",
        "success": "#30D158",
        "warning": "#FF9F0A",
        "danger": "#FF453A",
        "danger_hover": "#FF6259",
        "info": "#64D2FF",

        # Borders & dividers
        "border": "#48484A",
        "border_light": "#38383A",
        "divider": "#38383A",

        # Shadows
        "shadow_color": "#000000",
        "shadow_opacity": 0.3,

        # Progress
        "progress_bg": "#48484A",
        "progress_fill": "#0A84FF",

        # Table
        "table_header_bg": "#2C2C2E",
        "table_alt_row": "#2C2C2E",
        "table_selected": "#1C3A5C",
    }

    # ── Glass tokens（已退役 2026-09-22）─────────────────────────
    # 「玻璃」第三主题曾尝试 NSGlassEffectView 手工注入（v3.0-v3.2），经用户三轮
    # 目验全部否决；最终定案=打包主程序以 SDK 27 链接（build.sh vtool 步骤），
    # 系统全时自动提供官方新设计（玻璃窗口控件等），无需任何主题档区分。
    # 历史持久化值 "glass" 在 _detect_system_theme 归一化为深色。

    # ── Shared tokens (mode-independent) ───────────────────────────
    SHARED = {
        # Spacing
        "spacing_xs": 4,
        "spacing_sm": 8,
        "spacing_md": 12,
        "spacing_lg": 16,
        "spacing_xl": 24,

        # Radii
        "radius_sm": 4,
        "radius_md": 6,
        "radius_lg": 8,
        "radius_xl": 12,

        # Sidebar
        "sidebar_width": 200,
        "sidebar_item_height": 40,

        # Font sizes
        "font_size_xs": 11,
        "font_size_sm": 12,
        "font_size_md": 13,
        "font_size_lg": 15,
        "font_size_xl": 18,
        "font_size_title": 22,

        # ── 像素 Token（ui/tokens.py 镜像，⌘R 热重载覆盖）──
        "band_h": 40,
        "band_slot_h": 36,
        "ctrl_h": 28,
        "ctrl_sm_h": 26,
        "ctrl_min_w_2ch": 60,
        "row_h_table": 28,
        "row_h_tree": 28,
        "header_h": 32,
        "stat_card_h": 32,
        "pager_h": 30,
        "pager_btn_h": 26,
        "rail_top_pad": 8,
        "rail_gap": 6,
        "tree_indent_project": 26,
        "tree_indent_source": 0,
        "sp_4": 4,
        "sp_8": 8,
        "sp_12": 12,
        "sp_16": 16,
        "sp_24": 24,
        "hero_icon": 64,
        "hero_icon_radius": 16,
    }


class ThemeManager(QObject):
    """Singleton theme manager — emits signal when theme changes."""

    theme_changed = Signal(str)  # emits "light" or "dark"

    _instance = None
    _current_mode: str = "light"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        super().__init__()
        self._initialized = True
        self._detect_system_theme()

    def _detect_system_theme(self):
        """启动恢复上次外观（QSettings 持久化）；无记录/非法值时默认深色。

        历史值 "glass"（已退役的第三主题）归一化为深色——新设计由系统全时提供。
        """
        try:
            saved = QSettings("MediaTag", "MediaTag").value("appearance_mode", "dark")
        except Exception:
            saved = "dark"
        if saved == "glass":
            saved = "dark"
        self._current_mode = saved if saved in ("light", "dark") else "dark"

    def _persist_mode(self, value: str) -> None:
        """外观三态持久化：切换即写 QSettings，重启恢复。失败静默（不影响运行）。"""
        try:
            QSettings("MediaTag", "MediaTag").setValue("appearance_mode", value)
        except Exception:
            pass

    @property
    def mode(self) -> str:
        return self._current_mode

    #: 深色系外观（标题栏/调色板按深色处理）
    _DARK_LIKE = ("dark",)

    @mode.setter
    def mode(self, value: str):
        if value in ("light", "dark") and value != self._current_mode:
            self._current_mode = value
            self._persist_mode(value)
            self.theme_changed.emit(value)
            self.apply_palette()

    def toggle(self):
        """两态循环：深色 → 浅色 → 深色。"""
        order = ("dark", "light")
        idx = order.index(self._current_mode) if self._current_mode in order else 0
        self.mode = order[(idx + 1) % len(order)]

    def apply_palette(self):
        """同步 QApplication 调色板，让不依赖 QSS 的控件（标题栏/默认文字）也跟随主题。

        根因：widget 级 setStyleSheet 会阻断子树继承应用级 QSS 的文字颜色，
        未显式设色的 QLabel 等回退到系统调色板（macOS 强制深色 → 恒白）。
        设置应用调色板后，这些控件用调色板的 WindowText/Text 色。
        """
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is None:
                return
            t = self

            def cv(token: str) -> QColor:
                return QColor(t.get(token))

            pal = app.palette()
            bg = cv("bg_primary")
            bg2 = cv("bg_secondary")
            text = QColor(t.get("text_primary"))
            text_sec = QColor(t.get("text_secondary"))
            accent = QColor(t.get("accent"))

            pal.setColor(QPalette.ColorRole.Window, bg)
            pal.setColor(QPalette.ColorRole.WindowText, text)
            pal.setColor(QPalette.ColorRole.Base, bg2)
            pal.setColor(QPalette.ColorRole.AlternateBase, cv("table_alt_row"))
            pal.setColor(QPalette.ColorRole.Text, text)
            pal.setColor(QPalette.ColorRole.Button, bg2)
            pal.setColor(QPalette.ColorRole.ButtonText, text)
            pal.setColor(QPalette.ColorRole.Highlight, accent)
            pal.setColor(QPalette.ColorRole.HighlightedText, QColor(t.get("text_on_accent")))
            pal.setColor(QPalette.ColorRole.PlaceholderText, text_sec)
            pal.setColor(QPalette.ColorRole.ToolTipBase, cv("bg_secondary"))
            pal.setColor(QPalette.ColorRole.ToolTipText, text)
            pal.setColor(QPalette.ColorRole.Link, accent)
            # 禁用态
            pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, text_sec)
            pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, text_sec)
            pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, text_sec)
            # macOS：先让平台色方案(窗口标题栏)跟随主题，再设调色板——
            # setColorScheme 可能触发重刷，放前面避免覆盖刚设好的调色板。
            import sys as _sys
            if _sys.platform == "darwin":
                self._apply_macos_appearance()

            app.setPalette(pal)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"apply_palette 失败: {e}")

    def _apply_macos_appearance(self):
        """macOS：让系统窗口标题栏跟随应用主题（而非系统外观）。

        Qt 6.5+ 会按 QStyleHints::colorScheme（默认取系统外观）为**每个 NSWindow**
        设置窗口级 appearance，窗口级会覆盖应用级 NSApp.appearance —— 因此只设
        NSApp.appearance 不够：系统为浅色时标题栏仍为白色。必须让 Qt 自身的
        colorScheme 与主题一致，再由 Qt 把窗口外观置为深色。
        """
        dark = self._current_mode in self._DARK_LIKE
        # 1) Qt 色方案（决定窗口标题栏）—— Qt 6.5+ 提供
        try:
            from PySide6.QtGui import QGuiApplication
            from PySide6.QtCore import Qt
            hints = QGuiApplication.styleHints()
            if hasattr(hints, "setColorScheme"):
                hints.setColorScheme(
                    Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Qt setColorScheme 失败: {e}")
        # 2) 应用级 appearance 兜底（非窗口 chrome；旧版 Qt 无 setColorScheme 时也覆盖窗口）
        try:
            from AppKit import (
                NSApplication, NSAppearance,
                NSAppearanceNameDarkAqua, NSAppearanceNameAqua,
            )
            ns_app = NSApplication.sharedApplication()
            name = NSAppearanceNameDarkAqua if dark else NSAppearanceNameAqua
            ns_app.setAppearance_(NSAppearance.appearanceNamed_(name))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"macOS appearance 设置失败: {e}")

    def get(self, token_name: str, default: str | None = None) -> str:
        """Get a token value by name (always returned as str for CSS interpolation).

        兼容字典式 default 入参（licensing/status_panel 以 .get(key, fallback) 调用）；
        未命中时返回 default（None 则返回空串，保持旧签名行为）。"""
        if token_name in ThemeTokens.SHARED:
            return str(ThemeTokens.SHARED[token_name])
        tokens = self._mode_tokens()
        if token_name in tokens:
            return str(tokens[token_name])
        return str(default) if default is not None else ""

    def color(self, token_name: str) -> QColor:
        """Get a QColor for a color token."""
        return QColor(self.get(token_name))

    def tokens(self) -> dict:
        """Get all tokens for current mode (merged with shared)."""
        return {**ThemeTokens.SHARED, **self._mode_tokens()}

    def _mode_tokens(self) -> dict:
        if self._current_mode == "light":
            return ThemeTokens.LIGHT
        return ThemeTokens.DARK


# ── Convenience accessor ──────────────────────────────────────────────

def reload_pixel_tokens():
    """⌘R / 启动同步：把 ui.tokens 全部 UPPERCASE 值以小写名灌进
    ThemeTokens.SHARED（新增或覆盖）。tokens.py 是全应用布局数值的
    唯一来源，本函数只做镜像——theme().get('band_h') 即可查询。
    页面代码禁止字面量（scripts/lint_ui_literals.py 门禁）。"""
    import importlib
    try:
        import ui.tokens as _tokens_mod
    except ImportError:
        # 打包/脚本态：sys.path 可能只有 src/ui（无 ui 包父级）——
        # 回退顶层 tokens（spec 已把 src/ui 入包，Resources 两种布局均可命中的兜底）
        import tokens as _tokens_mod
    importlib.reload(_tokens_mod)
    count = 0
    for name in dir(_tokens_mod):
        if not name.isupper():
            continue
        val = getattr(_tokens_mod, name)
        if isinstance(val, (int, float)):
            ThemeTokens.SHARED[name.lower()] = val
            count += 1
    return count


# 导入即同步：任何页面 import theme 后即可查到全部几何 token，
# 不依赖 ⌘R 或显式调用（基建落地后无人接线的坑不再存在）。
try:
    reload_pixel_tokens()
except Exception:  # 脚本态/极早导入失败不阻塞主题加载
    pass


def theme() -> ThemeManager:
    """Get the global theme manager instance."""
    return ThemeManager()


# ── Style helpers ─────────────────────────────────────────────────────

def get_system_font(size: int = 13, weight: str = "normal") -> QFont:
    """Get system font matching macOS native style."""
    font = QFont()
    if sys.platform == "darwin":
        font.setFamily(".AppleSystemUIFont")
    else:
        font.setFamily("Segoe UI")  # Windows fallback
    font.setPointSize(size)
    if weight == "bold":
        font.setWeight(QFont.Weight.Bold)
    elif weight == "medium":
        font.setWeight(QFont.Weight.Medium)
    return font


def card_style() -> str:
    """Generate CSS for a card component with current theme."""
    t = theme()
    return f"""
        QFrame#card {{
            background: {t.get('bg_card')};
            border: 1px solid {t.get('border_light')};
            border-radius: {t.get('radius_lg')}px;
        }}
    """


def button_primary_style() -> str:
    """主钮（18§4.2）：accent 实底 + 白字 600 + hover 提亮——操作行的
    「主次」由它承担（导入/去导入任务等唯一主动作专用）。"""
    t = theme()
    hover = t.get('accent_hover') or "#2492FF"
    return f"""
        QPushButton {{
            background: {t.get('accent')};
            color: #FFFFFF;
            border: none;
            border-radius: {t.get('radius_md')}px;
            padding: 4px 14px;
            font-size: {t.get('font_size_md')}px;
            font-weight: 600;
        }}
        QPushButton:hover {{ background: {hover}; }}
        QPushButton:disabled {{ background: {t.get('border')}; color: {t.get('text_secondary')}; }}
        QPushButton:hover {{
            background: {t.get('btn_glass_hover')};
            border-color: {t.get('accent')};
            color: {t.get('text_primary')};
        }}
        QPushButton:pressed {{
            background: {t.get('btn_glass_hover')};
        }}
        QPushButton:disabled {{
            background: {t.get('bg_primary')};
            color: {t.get('text_tertiary')};
            border-color: {t.get('border_light')};
        }}
    """


def rgba_str(hex_color: str, alpha: float) -> str:
    """'#0A84FF' + 0.14 → 'rgba(10,132,255,0.14)'（半透 accent 派生，
    跟随主题 accent 变化；无效色回退 #0A84FF）。"""
    c = QColor(hex_color)
    if not c.isValid():
        c = QColor("#0A84FF")
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


def sm_button_style(checked: bool = False) -> str:
    """顶带 sm 档按钮（19§D1.2/D1.4）：padding 0 12、字 12；
    选中态=accent.14 底+accent 字+accent.4 边（视图切换互斥组）。
    高 26 由调用方以 theme().get('ctrl_sm_h') 钳制。"""
    t = theme()
    if checked:
        return (
            f"QPushButton {{ background: {rgba_str(t.get('accent'), 0.14)}; "
            f"color: {t.get('accent')}; "
            f"border: 1px solid {rgba_str(t.get('accent'), 0.4)}; "
            f"border-radius: {t.get('radius_ctrl')}px; "
            f"padding: 0 {t.get('sm_btn_pad_x')}px; "
            f"font-size: {t.get('font_secondary')}px; }}"
            f"QPushButton:hover {{ background: {rgba_str(t.get('accent'), 0.20)}; }}"
        )
    return (
        f"QPushButton {{ background: {t.get('btn_glass_bg')}; "
        f"color: {t.get('text_primary')}; "
        f"border: 1px solid {t.get('border')}; "
        f"border-radius: {t.get('radius_ctrl')}px; "
        f"padding: 0 {t.get('sm_btn_pad_x')}px; "
        f"font-size: {t.get('font_secondary')}px; }}"
        f"QPushButton:hover {{ background: {t.get('btn_glass_hover')}; }}"
        f"QPushButton:disabled {{ border-color: {t.get('border_light')}; "
        f"color: {t.get('text_tertiary')}; }}"
    )


def sm_combo_style() -> str:
    """顶带 sm 档下拉（19§D1.5）。"""
    t = theme()
    return (
        f"QComboBox {{ background: {t.get('btn_glass_bg')}; "
        f"color: {t.get('text_secondary')}; "
        f"border: 1px solid {t.get('border')}; "
        f"border-radius: {t.get('radius_ctrl')}px; "
        f"padding: 0 {int(t.get('sm_btn_pad_x') or 12) - 1}px; "
        f"font-size: {t.get('font_small')}px; }}"
        f"QComboBox:hover {{ background: {t.get('btn_glass_hover')}; }}"
        f"QComboBox::drop-down {{ border: none; width: 18px; }}"
    )


def search_box_style() -> str:
    """顶带搜索框（19§D1.3）：26 高玻璃底、字 12.5。"""
    t = theme()
    return (
        f"QLineEdit {{ background: {t.get('btn_glass_bg')}; "
        f"color: {t.get('text_primary')}; border: 1px solid {t.get('border')}; "
        f"border-radius: {t.get('radius_ctrl')}px; padding: 0 10px; "
        f"font-size: {t.get('font_small')}px; }}"
    )


def tag_chip_style() -> str:
    """标签 chip（19§D4.8a/b）：11 字/圆角 4/底 glass/字 txt2。"""
    t = theme()
    return (
        f"QLabel {{ font-size: {t.get('chip_fs')}px; "
        f"border-radius: {t.get('chip_radius')}px; "
        f"padding: 1px {t.get('chip_pad_x')}px; "
        f"background: {t.get('btn_glass_bg')}; "
        f"color: {t.get('text_secondary')}; }}"
    )


def sidebar_panel_style() -> str:
    """统一侧栏树面板实色垫底（D8 教训：透明面板透出 macOS 27
    窗背玻璃发黑；面板必须不透明）。"""
    t = theme()
    return f"background: {t.get('bg_secondary')}; color: {t.get('text_primary')};"


def button_secondary_style() -> str:
    """Generate CSS for secondary/outline button — 中性色纯文字按钮。"""
    t = theme()
    return f"""
        QPushButton {{
            background: {t.get('btn_glass_bg')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            border-radius: {t.get('radius_md')}px;
            padding: 4px 13px;
            font-size: {t.get('font_size_md')}px;
        }}
        QPushButton:hover {{
            background: {t.get('btn_glass_hover')};
            border-color: {t.get('accent')};
        }}
        QPushButton:disabled {{
            border-color: {t.get('border_light')};
            color: {t.get('text_tertiary')};
        }}
    """


def button_danger_style() -> str:
    """Generate CSS for destructive button — 中性色纯文字按钮（文字用危险色）。"""
    t = theme()
    return f"""
        QPushButton {{
            background: {t.get('btn_glass_bg')};
            color: {t.get('danger')};
            border: 1px solid {t.get('border')};
            border-radius: {t.get('radius_md')}px;
            padding: 4px 14px;
            font-size: {t.get('font_size_md')}px;
        }}
        QPushButton:hover {{
            background: {t.get('btn_glass_hover')};
            border-color: {t.get('danger')};
        }}
        QPushButton:disabled {{
            border-color: {t.get('border_light')};
            color: {t.get('text_tertiary')};
        }}
    """


def input_style() -> str:
    """Generate CSS for input fields."""
    t = theme()
    return f"""
        QLineEdit, QSpinBox, QComboBox {{
            background: {t.get('btn_glass_bg')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            border-radius: {t.get('radius_md')}px;
            padding: 4px 10px;
            font-size: {t.get('font_size_md')}px;
        }}
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
            border-color: {t.get('accent')};
        }}
    """


def group_box_style() -> str:
    """Generate CSS for group boxes (sections)."""
    t = theme()
    return f"""
        QGroupBox {{
            background: {t.get('bg_card')};
            border: 1px solid {t.get('border_light')};
            border-radius: {t.get('radius_lg')}px;
            margin-top: 12px;
            padding-top: 16px;
            font-size: {t.get('font_size_lg')}px;
            font-weight: 600;
            color: {t.get('text_primary')};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 4px 12px;
            color: {t.get('text_primary')};
        }}
    """


def table_style() -> str:
    """Generate CSS for table widgets."""
    t = theme()
    return f"""
        QTableWidget {{
            background: {t.get('bg_card')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border_light')};
            border-radius: {t.get('radius_lg')}px;
            gridline-color: {t.get('divider')};
            font-size: {t.get('font_size_md')}px;
            selection-background-color: {t.get('table_selected')};
            selection-color: {t.get('text_primary')};
            alternate-background-color: {t.get('table_alt_row')};
        }}
        QTableWidget::item {{
            padding: 4px 8px;
        }}
        QHeaderView::section {{
            background: {t.get('table_header_bg')};
            color: {t.get('text_secondary')};
            padding: 4px 8px;  /* 表头高一档制 28（17 号规范） */
            border: none;
            border-bottom: 1px solid {t.get('border')};
            font-size: {t.get('font_size_sm')}px;
            font-weight: 600;
        }}
    """


def progress_bar_style() -> str:
    """Generate CSS for progress bars."""
    t = theme()
    return f"""
        QProgressBar {{
            background: {t.get('progress_bg')};
            border: none;
            border-radius: {t.get('radius_sm')}px;
            text-align: center;
            color: {t.get('text_primary')};
            font-size: {t.get('font_size_sm')}px;
            height: 6px;
        }}
        QProgressBar::chunk {{
            background: {t.get('progress_fill')};
            border-radius: {t.get('radius_sm')}px;
        }}
    """


def scroll_bar_style() -> str:
    """Generate CSS for scroll bars."""
    t = theme()
    return f"""
        QScrollBar:vertical {{
            background: {t.get('btn_glass_bg')};
            width: 8px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {t.get('border')};
            border-radius: 4px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {t.get('text_tertiary')};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            background: {t.get('btn_glass_bg')};
            height: 8px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background: {t.get('border')};
            border-radius: 4px;
            min-width: 20px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {t.get('text_tertiary')};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
    """


def global_stylesheet() -> str:
    """Generate the complete application stylesheet."""
    t = theme()
    return f"""
        /* ── Global ─────────────────────────────────────────── */
        QMainWindow {{
            background: {t.get('bg_primary')};
        }}
        QWidget {{
            font-family: {get_system_font().family()};
            color: {t.get('text_primary')};
        }}

        /* ── Labels ─────────────────────────────────────────── */
        QLabel {{
            color: {t.get('text_primary')};
        }}

        /* ── Inputs ─────────────────────────────────────────── */
        {input_style()}

        /* ── Buttons ────────────────────────────────────────── */
        QPushButton {{
            background: {t.get('bg_secondary')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            border-radius: {t.get('radius_md')}px;
            padding: 4px 14px;
            font-size: {t.get('font_size_md')}px;
        }}
        QPushButton:hover {{
            background: {t.get('btn_glass_hover')};
            border-color: {t.get('accent')};
        }}
        QPushButton:pressed {{
            background: {t.get('btn_glass_hover')};
        }}
        QPushButton:disabled {{
            background: {t.get('bg_primary')};
            color: {t.get('text_tertiary')};
            border-color: {t.get('border_light')};
        }}

        /* ── Group boxes ────────────────────────────────────── */
        {group_box_style()}

        /* ── Tables ─────────────────────────────────────────── */
        {table_style()}

        /* ── Progress bars ──────────────────────────────────── */
        {progress_bar_style()}

        /* ── Scroll bars ────────────────────────────────────── */
        {scroll_bar_style()}

        /* ── Combo boxes ────────────────────────────────────── */
        QComboBox {{
            background: {t.get('bg_input')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            border-radius: {t.get('radius_md')}px;
            padding: 4px 10px;
            font-size: {t.get('font_size_md')}px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background: {t.get('bg_card')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            selection-background-color: {t.get('accent')};
            selection-color: {t.get('text_on_accent')};
        }}

        /* ── Tree widgets ───────────────────────────────────── */
        QTreeWidget {{
            background: {t.get('bg_card')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border_light')};
            border-radius: {t.get('radius_lg')}px;
            font-size: {t.get('font_size_md')}px;
            alternate-background-color: {t.get('table_alt_row')};
        }}
        QTreeWidget::item {{
            height: 28px;  /* 树行高一档制（根参数钉板 2026-09-28） */
        }}
        QTreeWidget::item:selected {{
            background: {t.get('table_selected')};
            color: {t.get('text_primary')};
        }}

        /* ── List widgets ───────────────────────────────────── */
        QListWidget {{
            background: {t.get('bg_card')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border_light')};
            border-radius: {t.get('radius_lg')}px;
            font-size: {t.get('font_size_md')}px;
        }}
        QListWidget::item:selected {{
            background: {t.get('table_selected')};
            color: {t.get('text_primary')};
        }}

        /* ── Text edit ──────────────────────────────────────── */
        QTextEdit {{
            background: {t.get('bg_input')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            border-radius: {t.get('radius_md')}px;
            padding: 6px;
            font-size: {t.get('font_size_md')}px;
        }}

        /* ── Splitter ───────────────────────────────────────── */
        QSplitter::handle {{
            background: {t.get('divider')};
        }}
        QSplitter::handle:horizontal {{
            width: 1px;
        }}
        QSplitter::handle:vertical {{
            height: 1px;
        }}

        /* ── Tooltip ────────────────────────────────────────── */
        QToolTip {{
            background: {t.get('bg_card')};
            color: {t.get('text_primary')};
            border: 1px solid {t.get('border')};
            border-radius: {t.get('radius_sm')}px;
            padding: 4px 8px;
        }}

        /* ── Checkbox ───────────────────────────────────────── */
        /* 不自定义 QCheckBox::indicator——自定义 checked 只填蓝底会丢失
           对勾符号(观感是「纯蓝方块」)。用 Qt/macOS 原生 indicator,
           自带方框+对勾, 深浅色自适应。 */
        QCheckBox {{
            spacing: 6px;
            color: {t.get('text_primary')};
        }}
    """
