"""主题像素 Token 注入 + ⌘R 热重载。

tokens.py 是全应用布局像素值的唯一来源。ThemeTokens.__init__ 末尾将
tokens 的值覆盖进 self._tokens——全局 QSS 与所有页面立即生效。

⌘R 热重载流程：
    importlib.reload(ui.tokens) → theme().reload_tokens() →
    app.setStyleSheet(global_stylesheet()) → 主窗口 apply_tokens() 重放几何
"""

import importlib
import logging


from .theme import ThemeTokens

log = logging.getLogger(__name__)


def inject_tokens(cls):
    """把 tokens.py 的值注入 ThemeTokens 实例（几何/字号/圆角覆盖）。"""
    try:
        import ui.tokens as _tokens_mod
        importlib.reload(_tokens_mod)
        for name in dir(_tokens_mod):
            if not name.isupper():
                continue
            value = getattr(_tokens_mod, name)
            if isinstance(value, int):
                cls._tokens[name] = value
        log.debug("tokens 注入完成 (%d 项)", sum(1 for n in dir(_tokens_mod) if n.isupper()))
    except Exception:
        log.warning("tokens 注入失败，使用内置默认值", exc_info=True)
    return cls


# ThemeTokens.__init__ 末尾调用（theme.py 里 monkey-patch 注入）
_orig_init = ThemeTokens.__init__


def _patched_init(self):
    _orig_init(self)
    inject_tokens(self)


ThemeTokens.__init__ = _patched_init


def apply_tokens_recursive(widget):
    """递归遍历 widget 树，调用各控件的 apply_tokens()（几何重放钩子）。

    页面/组件实现 apply_tokens() 来响应 token 变更（如 band 高、
    控件宽）。没有实现该方法的控件静默跳过。
    """
    applied = 0
    for child in widget.findChildren(object):
        if hasattr(child, "apply_tokens") and callable(child.apply_tokens):
            try:
                child.apply_tokens()
                applied += 1
            except Exception:
                log.debug("apply_tokens 失败: %r", child, exc_info=True)
    return applied


def bind_reload(shortcut_host, main_window):
    """绑定 ⌘R → 热重载（token 值变更→QSS 重生成→几何重放）。

    在 MainWindow.__init__ 末尾调用。
    """
    from PySide6.QtGui import QKeySequence, QShortcut

    def _reload():
        import ui.tokens
        importlib.reload(ui.tokens)
        from .theme import ThemeManager
        ThemeManager.instance().refresh_tokens()
        if main_window is not None:
            apply_tokens_recursive(main_window)
        log.info("⌘R token 热重载完成")

    QShortcut(QKeySequence("Ctrl+R"), shortcut_host, _reload)
    return _reload
