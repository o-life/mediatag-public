"""``faces_dir``（人脸资源库目录）的**唯一解析入口**。

为什么单独收口：这个解析历史上散在 5 处、用了 3 种口径——

- 回退**项目根**（`FaceRecognizer.__init__`、`_api_faces_dir`）
- 按**进程 CWD**（`routes/actors._faces_dir`、`list_actors`/`get_actor` 两处内联）
- 是否 `expanduser` 也不一致

于是同一个相对路径可能被解析到**不同目录**。本模块给出唯一实现，
所有调用方一律经此函数。

规则（取既有实现中最完整的那一套，即 `_api_faces_dir` 的口径）：

    expanduser → 非绝对路径则相对项目根 → resolve

配置值为绝对路径（历史生产配置曾用外部卷绝对路径）时，
本函数与历史各口径结果完全一致 —— 收口本身不改变行为。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

#: 配置缺失时的默认值（与历史各处一致）
DEFAULT_FACES_DIR = "data/actor_faces"


def project_root() -> Path:
    """仓库/打包根目录（``src/core/`` 上溯三层）。"""
    return Path(__file__).resolve().parent.parent.parent


def resolve_faces_dir(
    raw: Optional[str] = None,
    *,
    get_config: Optional[Callable[[str, Any], Any]] = None,
) -> Path:
    """把 ``face_recognition.faces_dir`` 解析为绝对路径。

    Args:
        raw: 配置值；为 ``None`` 时自行读取配置。
        get_config: 可选的配置读取函数 ``(key, default) -> value``
            （用于已持有 ConfigManager 的调用方，避免重复构造）。

    Returns:
        绝对路径（未做目录创建；调用方按需 mkdir）。
    """
    if raw is None:
        if get_config is not None:
            raw = get_config("face_recognition.faces_dir", DEFAULT_FACES_DIR)
        else:
            from src.core.config import ConfigManager

            raw = ConfigManager().get("face_recognition.faces_dir", DEFAULT_FACES_DIR)

    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        # 相对路径：优先相对**当前主题目录**（每主题一个工作区时，`faces/`
        # 就该落在该主题目录内）；无主题工作区时回退项目根（历史口径）。
        base = _theme_base_dir()
        path = (base / path) if base is not None else (project_root() / path)
    return path.resolve()


def _theme_base_dir() -> Optional[Path]:
    """当前主题目录；无主题工作区（或未启用）时返回 None。"""
    try:
        from src.core.theme_workspace import get_theme_workspace

        ws = get_theme_workspace()
        name = ws.current_name()
        if name and ws.config_path(name).exists():
            return ws.theme_dir(name)
    except Exception:
        pass
    return None
