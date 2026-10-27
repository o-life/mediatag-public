"""macOS 原生毛玻璃（NSVisualEffectView / vibrancy）封装。

Qt Widgets 没有公开的 vibrancy API；本模块用仓库已有依赖 pyobjc-framework-cocoa
直接操作 NSWindow：把内容视图收编进一个 NSVisualEffectView（BehindWindow 模糊），
Qt 的 QSS 半透明背景（theme GLASS 档）叠在其上即呈现系统级毛玻璃。

设计要点：
- 纯 NSWindow 操作，不依赖 WA_TranslucentBackground（避免运行期重建原生窗口）；
- 任何一步失败都返回 False 并记日志，调用方据此回退纯色主题（不白屏）；
- 非 darwin 平台恒返回 False。
"""

import logging
import sys

logger = logging.getLogger(__name__)

#: NSVisualEffectMaterial.Sidebar 的原始值（Finder 侧栏同款磨砂质感，
#: 是用户认知中"官方玻璃"的代表材质；如需桌面透视更强可换
#: UnderWindowBackground=11）
_MATERIAL_SIDEBAR = 7


def _ns_window(widget):
    """Qt 顶层窗口 → NSWindow；失败返回 None。"""
    import objc

    ns_view = objc.objc_object(c_void_p=int(widget.winId()))
    ns_window = ns_view.window()
    if ns_window is None:
        # 窗口尚未显示（winId 存在但未挂到 NSWindow）
        return None
    return ns_window


def _is_effect_view(view) -> bool:
    """判断 view 是否为 NSVisualEffectView。

    注意：Qt 自建的 QNSView/QNSWindow 未被 pyobjc 桥接，class_() 等元方法
    会 AttributeError —— 用 isKindOfClass_ 探测并对异常一律按 False 处理
    （语义即"未挂载"）。
    """
    try:
        from AppKit import NSVisualEffectView

        return bool(view.isKindOfClass_(NSVisualEffectView))
    except Exception:
        return False


def apply_window_vibrancy(widget) -> bool:
    """给 widget 对应的原生窗口挂毛玻璃衬底。成功 True；失败/非 darwin False。

    可重入：已是 effect 视图时直接返回 True。
    """
    if sys.platform != "darwin":
        return False
    try:
        from AppKit import (
            NSColor,
            NSVisualEffectView,
            NSVisualEffectStateFollowsWindowActiveState,
            NSVisualEffectBlendingModeBehindWindow,
            NSViewHeightSizable,
            NSViewWidthSizable,
        )

        ns_window = _ns_window(widget)
        if ns_window is None:
            logger.warning("vibrancy: 窗口尚未就绪（未显示？）")
            return False

        content = ns_window.contentView()
        if content is None:
            return False
        if _is_effect_view(content):
            return True  # 已挂载

        effect = NSVisualEffectView.alloc().initWithFrame_(content.bounds())
        effect.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        effect.setMaterial_(_MATERIAL_SIDEBAR)
        effect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(NSVisualEffectStateFollowsWindowActiveState)
        effect.setWantsLayer_(True)

        # 把原 contentView 收编为 effect 的子视图，再以 effect 为 contentView：
        # 层级变为 NSWindow → effect(模糊) → 原 Qt 内容
        effect.addSubview_(content)
        ns_window.setContentView_(effect)

        # 窗口本体透明，让 BehindWindow 模糊透出来
        ns_window.setOpaque_(False)
        ns_window.setBackgroundColor_(NSColor.clearColor())
        # 标题栏延伸：原生红绿灯保留，材质向上延伸
        ns_window.setTitlebarAppearsTransparent_(True)
        return True
    except Exception as e:
        logger.warning(f"vibrancy: 挂载失败，将回退纯色主题: {e}")
        return False


def remove_window_vibrancy(widget) -> bool:
    """摘除毛玻璃衬底，恢复普通窗口。无 effect 视图时直接返回 True。"""
    if sys.platform != "darwin":
        return True
    try:
        from AppKit import NSColor

        ns_window = _ns_window(widget)
        if ns_window is None:
            return True
        content = ns_window.contentView()
        if content is None or not _is_effect_view(content):
            return True

        real = content.subviews()[0]
        ns_window.setContentView_(real)
        ns_window.setOpaque_(True)
        ns_window.setBackgroundColor_(NSColor.windowBackgroundColor())
        ns_window.setTitlebarAppearsTransparent_(False)
        return True
    except Exception as e:
        logger.warning(f"vibrancy: 摘除失败: {e}")
        return False
