"""
路径管理器 - 统一管理应用数据路径

根据运行环境（开发/打包）自动选择合适的数据存储路径：
- 开发环境：项目目录下的 data/
- 打包环境：~/Library/Application Support/MediaTag/

数据库路径从 config.yaml 的 database.path 读取
"""

import sys
import yaml
from pathlib import Path
from typing import Optional

# 产品更名（2026-09-21）：Video Tag → MediaTag
PRODUCT_NAME = "MediaTag"
LEGACY_PRODUCT_NAME = "Video Tag"


def app_support_base() -> Path:
    '''打包态数据根目录（macOS）：~/Library/Application Support/MediaTag

    含产品更名的一次性迁移：老目录存在且新目录不存在时整体 rename
    （同卷、瞬时、原子，数据库/人脸/配置全部跟随）；迁移失败时回退老目录，
    保证应用可继续运行。所有需要打包态数据根目录的代码统一走本函数，
    禁止各自拼接旧路径。
    '''
    home = Path.home() / "Library" / "Application Support"
    new_base = home / PRODUCT_NAME
    old_base = home / LEGACY_PRODUCT_NAME
    if not new_base.exists() and old_base.exists():
        try:
            old_base.rename(new_base)
            print(f"[PathManager] 产品更名数据迁移: {old_base} -> {new_base}")
            # 连带修补：老 config.yaml 的 database.path 仍指向旧名绝对路径，
            # 不修会导致升级后 SQLite 在旧路径新建空库、用户数据"消失"。
            cfg = new_base / "config.yaml"
            if cfg.exists():
                try:
                    txt = cfg.read_text(encoding="utf-8")
                    fixed = txt.replace(
                        "Application Support/Video Tag",
                        "Application Support/MediaTag",
                    ).replace(
                        "Application Support\\Video Tag",
                        "Application Support\\MediaTag",
                    )
                    if fixed != txt:
                        cfg.write_text(fixed, encoding="utf-8")
                        print("[PathManager] 已同步修正迁移后 config.yaml 的数据库路径")
                except Exception as ce:
                    print(f"[PathManager] config.yaml 路径修补失败（可手动修）: {ce}")

        except Exception as e:
            print(f"[PathManager] 数据目录迁移失败，沿用老目录: {e}")
            return old_base
    return new_base


def bundled_ffmpeg_bin_dir() -> Optional[Path]:
    """内置 LGPL ffmpeg 可执行目录；找不到返回 None。

    解析顺序：打包态 = App 内 Resources/bundled_bin/ffmpeg_lgpl/bin；
    开发态 = 仓库 bundled_bin/ffmpeg_lgpl/bin。仅 arm64（本项目只支持 M 系列芯片）。
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent
        candidates = [
            exe_dir.parent / "Resources" / "bundled_bin" / "ffmpeg_lgpl" / "bin",
            exe_dir / "bundled_bin" / "ffmpeg_lgpl" / "bin",
        ]
    else:
        candidates = [
            Path(__file__).parent.parent.parent / "bundled_bin" / "ffmpeg_lgpl" / "bin"
        ]
    for c in candidates:
        if (c / "ffmpeg").exists():
            return c
    return None


_bundled_bin_path_injected = False


def ensure_bundled_bin_on_path() -> Optional[Path]:
    """把内置 ffmpeg 目录注入 PATH 前端（幂等，导入本模块即生效一次）。

    分析管线中十余处 ffmpeg/ffprobe 子进程调用都是裸命令名走 PATH——
    注入后统一解析到随包分发的 LGPL 构建，用户无需自行安装任何东西；
    用户自装的 ffmpeg 仍可作为 PATH 后续项被显式指定使用。
    """
    global _bundled_bin_path_injected
    if _bundled_bin_path_injected:
        return None
    _bundled_bin_path_injected = True
    import os

    d = bundled_ffmpeg_bin_dir()
    if d is None:
        return None
    d_str = str(d)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if d_str in entries:
        return Path(d_str)
    os.environ["PATH"] = os.pathsep.join([d_str] + entries)
    return Path(d_str)


# 模块导入即注入（幂等）：统一覆盖 main / desktop_app / api 三类入口的散点子进程调用
ensure_bundled_bin_on_path()


class PathManager:
    """路径管理器单例类"""
    
    _instance: Optional['PathManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._is_frozen = getattr(sys, 'frozen', False)
        self._base_dir = self._determine_base_dir()
        
    def _determine_base_dir(self) -> Path:
        """确定基础目录"""
        if self._is_frozen:
            # PyInstaller 打包后：使用用户目录（含更名迁移）
            base = app_support_base()
        else:
            # 开发环境：使用项目目录
            base = Path(__file__).parent.parent.parent / "data"
        
        # 确保目录存在
        try:
            base.mkdir(parents=True, exist_ok=True)
            print(f"[PathManager] 基础目录已创建/确认: {base}")
        except Exception as e:
            print(f"[PathManager] 创建基础目录失败: {e}")
            raise
        return base
    
    @property
    def base_dir(self) -> Path:
        """获取基础数据目录"""
        return self._base_dir
    
    def _theme_config_file(self) -> Optional[Path]:
        """当前**主题**的配置文件；尚无主题工作区时返回 None（调用方回退旧布局）。"""
        try:
            from src.core.theme_workspace import get_theme_workspace

            return get_theme_workspace().current_config_path()
        except Exception:
            pass
        return None

    @property
    def config_file(self) -> Path:
        """配置文件路径：优先当前主题工作区，无主题时回退旧布局。

        主题选择只发生在进程启动时——本 property 不做创建，只做解析。
        """
        theme_cfg = self._theme_config_file()
        return theme_cfg if theme_cfg is not None else (self._base_dir / "config.yaml")

    @property
    def db_file(self) -> Path:
        """数据库文件路径，从**当前工作区**配置的 database.path 读取。"""
        # 候选顺序：当前主题配置 → （开发态）项目根 config.yaml → base_dir/config.yaml
        config_candidates = []
        theme_cfg = self._theme_config_file()
        if theme_cfg is not None:
            config_candidates.append(theme_cfg)
        if not self._is_frozen:
            # 开发环境：项目根目录
            project_root = Path(__file__).parent.parent.parent
            config_candidates.append(project_root / "config.yaml")
        # 打包环境或备用
        config_candidates.append(self._base_dir / "config.yaml")

        for config_path in config_candidates:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}
                    db_path_str = config.get('database', {}).get('path')
                    if db_path_str:
                        db_path = Path(db_path_str).expanduser()
                        # 如果是相对路径，转换为绝对路径（相对于项目根目录）
                        if not db_path.is_absolute():
                            if not self._is_frozen:
                                project_root = Path(__file__).parent.parent.parent
                                db_path = project_root / db_path
                            else:
                                db_path = self._base_dir / db_path
                        # 确保父目录存在
                        db_path.parent.mkdir(parents=True, exist_ok=True)
                        return db_path
                except Exception as e:
                    print(f"Warning: Failed to read database.path from {config_path}: {e}")

        # 默认路径：当前主题的 data.db（无主题时回退 base_dir/db/audit_cache.db）
        try:
            from src.core.theme_workspace import get_theme_workspace

            ws = get_theme_workspace()
            name = ws.current_name()
            if name:
                db_path = ws.db_path(name)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                return db_path
        except Exception:
            pass
        db_dir = self._base_dir / "db"
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir / "audit_cache.db"
    
    @property
    def frames_dir(self) -> Path:
        """获取帧文件存储目录"""
        frames = self._base_dir / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        return frames
    
    @property
    def logs_dir(self) -> Path:
        """获取日志目录"""
        logs = self._base_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        return logs
    
    @property
    def models_dir(self) -> Path:
        """获取模型目录"""
        models = self._base_dir / "models"
        models.mkdir(parents=True, exist_ok=True)
        return models
    
    @property
    def is_frozen(self) -> bool:
        """是否为打包环境"""
        return self._is_frozen
    
    def get_path(self, *parts: str) -> Path:
        """获取自定义路径"""
        path = self._base_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def ensure_dirs(self):
        """确保所有必要目录存在"""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        (self._base_dir / "db").mkdir(exist_ok=True)
        (self._base_dir / "frames").mkdir(exist_ok=True)
        (self._base_dir / "logs").mkdir(exist_ok=True)
        (self._base_dir / "models").mkdir(exist_ok=True)


# 全局实例
_path_manager: Optional[PathManager] = None


def get_path_manager() -> PathManager:
    """获取路径管理器实例"""
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager()
    return _path_manager


# 便捷函数
def get_base_dir() -> Path:
    """获取基础目录"""
    return get_path_manager().base_dir


def get_config_file() -> Path:
    """获取配置文件路径"""
    return get_path_manager().config_file


def get_db_file() -> Path:
    """获取数据库文件路径"""
    return get_path_manager().db_file


def get_frames_dir() -> Path:
    """获取帧文件目录"""
    return get_path_manager().frames_dir


def get_logs_dir() -> Path:
    """获取日志目录"""
    return get_path_manager().logs_dir


def get_models_dir() -> Path:
    """获取模型目录"""
    return get_path_manager().models_dir


def is_frozen() -> bool:
    """是否为打包环境"""
    return get_path_manager().is_frozen
