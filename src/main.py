"""
MediaTag 应用入口

启动流程：
1. 启动内置 Ollama 服务（如果存在）
2. 检查依赖是否安装
3. 如果缺失依赖，显示安装向导
4. 启动后端服务（FastAPI）
5. 启动前端界面（PySide6）
"""

import sys
import os
import signal
import logging
import threading
import time
import subprocess
from pathlib import Path

# 设置环境变量，确保 PySide6 正确加载
os.environ['QT_MAC_WANTS_LAYER'] = '1'
# 禁用 Qt 的字体缓存，避免在全新 macOS 环境中因 fontd 服务未就绪导致挂起
os.environ['QT_QPA_FONTCONFIG_FALLBACK'] = '1'

# 打包后设置 Qt 平台插件路径
if getattr(sys, 'frozen', False):
    # PyInstaller 打包后的路径
    base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    plugin_path = os.path.join(base_path, 'PySide6', 'Qt', 'plugins', 'platforms')
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

from PySide6.QtWidgets import QApplication

from src.core.path_manager import get_path_manager
from src.core.dependency_checker import DependencyChecker
from src.ui.components.dependency_wizard import DependencyWizard


def ensure_config_file():
    """确保用户目录有完整的配置文件"""
    if not getattr(sys, 'frozen', False):
        return  # 开发环境不需要

    # 打包态 CWD 统一切到可写数据根:历史遗留的裸相对路径
    # (data/dictionary.md、data/audit_cache.db 等,见 schema.py/dictionary)
    # 在 Finder 启动(CWD=/)下会撞只读文件系统(2026-10-07 实测 Errno 30)。
    # 彻底的相对路径治理在 add-public-sale-readiness 阶段 2.3,此处先兜底。
    try:
        os.chdir(get_path_manager().base_dir)
        os.environ.setdefault("MEDIA_DATA_CWD", os.getcwd())
    except Exception as _cwd_exc:
        print(f"[startup] chdir 到数据根失败(保持原 CWD): {_cwd_exc}", file=sys.stderr)

    pm = get_path_manager()
    user_config = pm.config_file
    
    # 如果用户目录已有配置，不覆盖
    if user_config.exists():
        return
    
    # 从 app bundle 复制完整配置
    try:
        base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
        bundled_config = Path(base_path) / "config.yaml"
        
        if bundled_config.exists():
            import shutil
            user_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled_config, user_config)
            print(f"✅ 已复制配置文件到: {user_config}")
        else:
            print(f"⚠ 未找到内置配置文件: {bundled_config}")
    except Exception as e:
        print(f"⚠ 复制配置文件失败: {e}")


def start_bundled_ollama():
    """启动内置的 Ollama 服务"""
    from src.core.dependency_checker import DependencyChecker
    checker = DependencyChecker()
    bundled_ollama = checker._get_bundled_ollama_path()
    
    if not bundled_ollama:
        print("未找到内置 Ollama，跳过自动启动")
        return None
    
    print(f"启动内置 Ollama 服务: {bundled_ollama}")
    
    try:
        # 设置 Ollama 环境变量
        os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'
        
        # 启动 Ollama 服务
        process = subprocess.Popen(
            [str(bundled_ollama), 'serve'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy()
        )
        
        # 等待服务启动
        time.sleep(3)
        
        # 检查服务是否启动成功
        if process.poll() is None:
            print("✓ 内置 Ollama 服务启动成功")
            return process
        else:
            print("✗ 内置 Ollama 服务启动失败")
            return None
    except Exception as e:
        print(f"✗ 启动内置 Ollama 服务失败: {e}")
        return None


def download_minicpm_model(progress_callback=None, completion_callback=None):
    """下载 MiniCPM-V 模型（支持进度回调）

    Args:
        progress_callback: 可选，进度回调函数，签名 callback(progress: DownloadProgress)
        completion_callback: 可选，完成回调函数，签名 callback(success: bool, message: str)

    Returns:
        bool: 下载是否成功（仅在未提供 completion_callback 时有效）
    """
    from src.core.dependency_checker import DependencyChecker
    from src.core.model_downloader import ModelDownloader, DownloadProgress

    checker = DependencyChecker()

    # 检查模型是否已下载
    deps = checker.check_all()
    minicpm_status = deps.get('minicpm_model')
    if minicpm_status and minicpm_status.installed:
        print("✓ MiniCPM-V 模型已存在")
        if completion_callback:
            completion_callback(True, "模型已存在")
        return True

    bundled_ollama = checker._get_bundled_ollama_path()
    if not bundled_ollama:
        print("未找到内置 Ollama，无法下载模型")
        if completion_callback:
            completion_callback(False, "未找到内置 Ollama")
        return False

    print("开始下载 MiniCPM-V 模型（约 5GB，请耐心等待）...")

    # 如果提供了回调，使用异步下载
    if progress_callback or completion_callback:
        downloader = ModelDownloader(
            ollama_path=str(bundled_ollama),
            max_retries=3
        )

        def on_progress(progress: DownloadProgress):
            if progress_callback:
                progress_callback(progress)
            else:
                # 默认控制台输出
                if progress.percent > 0:
                    print(f"  {progress.percent:.1f}% | {progress.downloaded}/{progress.total} | {progress.speed}")
                else:
                    print(f"  {progress.status}")

        def on_completed(success: bool, message: str):
            if completion_callback:
                completion_callback(success, message)
            else:
                if success:
                    print(f"✓ {message}")
                else:
                    print(f"✗ {message}")

        downloader.download_async(
            model_name='minicpm-v4.5',
            on_progress=on_progress,
            on_completed=on_completed,
            on_started=lambda name: print(f"开始下载模型: {name}")
        )

        # 返回 worker 对象供调用者管理
        return downloader._worker
    else:
        # 同步下载（向后兼容）
        try:
            result = subprocess.run(
                [str(bundled_ollama), 'pull', 'minicpm-v4.5'],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=1800  # 30 分钟超时
            )

            if result.returncode == 0:
                print("✓ MiniCPM-V 模型下载完成")
                return True
            else:
                print(f"✗ 模型下载失败: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("✗ 模型下载超时")
            return False
        except Exception as e:
            print(f"✗ 下载模型失败: {e}")
            return False


def check_and_install_dependencies() -> bool:
    """检查并安装依赖，返回是否所有依赖就绪"""
    checker = DependencyChecker()
    deps = checker.check_all()
    
    # 检查是否有缺失的依赖
    missing = [dep for dep in deps.values() if not dep.installed]
    
    if not missing:
        print("✓ 所有依赖已安装")
        return True
    
    # 显示依赖安装向导
    print(f"发现 {len(missing)} 个缺失依赖，启动安装向导...")
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    wizard = DependencyWizard()
    result = wizard.exec()
    
    if result == wizard.DialogCode.Accepted:
        # 重新检查依赖
        checker = DependencyChecker()
        deps = checker.check_all()
        missing = [dep for dep in deps.values() if not dep.installed]
        
        if not missing:
            print("✓ 依赖安装完成")
            return True
        else:
            print(f"⚠ 仍有 {len(missing)} 个依赖未安装")
            return False
    else:
        print("✗ 用户取消安装")
        return False


def start_backend_server():
    """启动后端服务"""
    import uvicorn
    from src.api.main import app
    from src.core.log_rotation import setup_logging

    path_manager = get_path_manager()
    logs_dir = path_manager.logs_dir

    # 读取日志配置
    from src.core.config import ConfigManager
    config = ConfigManager()
    retention_days = config.get("logging.retention_days", 7)

    print("启动后端服务...")
    print(f"日志目录: {logs_dir}")
    print(f"日志保留: {retention_days} 天")

    # 配置日志（自动按天轮转 + 清理过期日志）
    setup_logging(logs_dir, retention_days=retention_days)

    # 启动 uvicorn（使用默认日志配置，root logger 已由 setup_logging 配置）
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        log_config=None,  # 使用已配置的 logger
    )


def main():
    """主函数"""
    print("=" * 60)
    print("MediaTag - 视频内容分析工具")
    print("=" * 60)
    
    # 确保配置文件存在（打包环境）
    ensure_config_file()

    # 主题工作区初始化：为"当前主题"建档（主题目录 + config.yaml + faces 目录）。
    # 在解析 DB / 配置路径之前完成；此后 ConfigManager / PathManager 均按当前主题解析。
    try:
        from src.core.theme_workspace import bootstrap_current_theme
        bootstrap_current_theme()
    except Exception as _theme_exc:
        print(f"⚠ 主题工作区初始化失败(回退旧布局): {_theme_exc}")
    
    # ── 单实例锁 ─────────────────────────────────────────────
    # 只允许一个实例运行：多实例会争抢 8000 端口/SQLite/人脸模型，
    # 且 PyInstaller+multiprocessing 下多开可能级联放大资源占用。
    # 用 flock 文件锁：进程退出/崩溃后内核自动释放，不会残留死锁。
    _singleton_lock = None
    try:
        import fcntl
        try:
            from src.core.path_manager import app_support_base
        except ImportError:
            from core.path_manager import app_support_base
        _lock_path = app_support_base() / "singleton.lock"
        _lock_path.parent.mkdir(parents=True, exist_ok=True)
        _singleton_lock = open(_lock_path, "w")
        try:
            fcntl.flock(_singleton_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("⚠ 已有 MediaTag 实例在运行，本实例退出")
            # 打包环境弹窗提示（GUI 下更友好）
            try:
                import PySide6.QtWidgets as _qtw
                _qapp = _qtw.QApplication.instance() or _qtw.QApplication(sys.argv)
                _qtw.QMessageBox.information(
                    None, "MediaTag",
                    "MediaTag 已在运行中。\n请切换到已打开的窗口，不要重复启动。",
                )
            except Exception:
                pass
            sys.exit(0)
    except ImportError:
        _singleton_lock = None  # 非 POSIX 平台跳过（本项目仅 macOS）
    except Exception as _lock_exc:
        print(f"⚠ 单实例锁创建失败(继续启动): {_lock_exc}")
    
    # ── 授权模式探测改至主窗体呈现之后（docs/界面设计/07-授权管理.md §2）：
    # 免费额度模式下主界面 0s 直接呈现，模式只影响额度，不再有任何启动门。

    # 启动内置 Ollama 服务
    print("\n启动内置 Ollama 服务...")
    ollama_process = start_bundled_ollama()
    
    # 初始化路径管理器
    path_manager = get_path_manager()
    print(f"\n数据目录: {path_manager.base_dir}")
    
    # 打包完整版: 提前把内置人脸模型(AuraFace)复制到
    # ~/.insightface/models/——否则依赖检查会误判「模型缺失」而弹欢迎页
    # (复制实际发生在后端首次加载时, 晚于本处检查)。
    try:
        from src.core.face_utils import ensure_insightface_models_copied
        if getattr(sys, "frozen", False):
            ensure_insightface_models_copied("auraface")
    except Exception as _copy_exc:
        print(f"⚠ 预复制内置人脸模型失败(可忽略): {_copy_exc}")
    
    # 检查依赖（非交互式检查，不显示向导）
    print("\n检查依赖...")
    checker = DependencyChecker()
    deps = checker.check_all()
    missing = [dep for dep in deps.values() if not dep.installed]
    
    if missing:
        print(f"⚠ 发现 {len(missing)} 个缺失依赖:")
        for dep in missing:
            print(f"  - {dep.name}: {dep.message}")
        print("  应用将继续启动，但部分功能可能不可用")
    
    # 检查是否是首次运行（需要下载模型）
    insightface_status = deps.get('insightface_model')
    # 人脸模型是必选组件(人脸识别依赖); MiniCPM-V(VLM)是可选组件——
    # 可用设置页配置其他 VLM 供应商/外部 Ollama, 缺失不阻塞使用。
    face_model_missing = not (insightface_status and insightface_status.installed)
    
    # 启动后端服务（在后台线程）
    print("\n启动后端服务...")
    backend_thread = threading.Thread(target=start_backend_server, daemon=True)
    backend_thread.start()
    
    # 等待后端启动
    print("等待后端启动...")
    time.sleep(2)
    
    # 检查后端是否启动成功
    import requests
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get("http://127.0.0.1:8000/status", timeout=1)
            if response.status_code == 200:
                print("✓ 后端服务启动成功")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("⚠ 后端服务启动失败，继续尝试启动界面...")
    
    # 启动前端界面
    print("\n启动前端界面...")
    from src.ui.desktop_app import MainWindow

    # 门禁阶段可能已创建 QApplication（授权对话框所需），此处复用
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("MediaTag")
    app.setApplicationDisplayName("MediaTag")
    
    # 设置应用图标
    icon_path = Path(__file__).parent.parent / "assets" / "AppIcon.icns"
    if icon_path.exists():
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(str(icon_path)))
        # 开发态(裸 python 运行, 无 .app bundle): Dock 默认显示 python3.11
        # 图标+名称。用 PyObjC 设置 NSApplication 图标/名称, 使 Dock 显示
        # 与应用一致(打包态有 Info.plist/icns, 无需此处)。
        if not getattr(sys, "frozen", False):
            try:
                from AppKit import NSApplication, NSImage
                _nsapp = NSApplication.sharedApplication()
                _img = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
                if _img:
                    _nsapp.setApplicationIconImage_(_img)
                _nsapp.setActivationPolicy_(0)  # NSApplicationActivationPolicyRegular
            except Exception as _dock_exc:
                print(f"⚠ 设置 Dock 图标失败(可忽略): {_dock_exc}")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()

    # ── 授权模式探测（后台）：完整版/免费版只影响额度，不阻塞呈现 ──
    # 免费额度模式见 docs/界面设计/07-授权管理.md；无任何启动门/锁定窗。
    try:
        from src.licensing.startup import start_mode_detection
    except ImportError:
        from licensing.startup import start_mode_detection
    start_mode_detection(window.license_mode_ready)
    
    # 人脸模型缺失(必选组件不齐)时才弹首次运行向导:
    # 引导下载人脸模型; 向导内保留 MiniCPM-V(VLM)作为「可选」下载,
    # 绝不默认/强制下载——VLM 可用设置页的其他供应商替代。
    if face_model_missing:
        from PySide6.QtCore import QTimer

        def show_first_run_wizard():
            """显示首次运行向导(与设置页「环境初始化」同一勾选式下载框)"""
            window.show_download_wizard()

        # 延迟显示向导，等待 UI 完全加载
        QTimer.singleShot(800, show_first_run_wizard)
    
    # 处理退出信号
    def signal_handler(signum, frame):
        print("\n收到退出信号，正在关闭...")
        if ollama_process:
            ollama_process.terminate()
        app.quit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行应用
    print("\n应用已启动")
    exit_code = app.exec()

    # 停止调度器：不再派发新任务，并丢弃队列中尚未开始的任务。
    try:
        from src.core.scheduler import get_scheduler, SchedulerStatus
        _sched = get_scheduler()
        _sched.status = SchedulerStatus.STOPPED
        _sched.executor.shutdown(wait=False, cancel_futures=True)
    except Exception as _stop_exc:
        print(f"⚠ 停止调度器失败(可忽略): {_stop_exc}")

    # 清理 Ollama 进程
    if ollama_process:
        try:
            ollama_process.terminate()
        except Exception:
            pass

    # 释放单实例锁(进程退出也会自动释放, 此处显式关闭更干净)
    if _singleton_lock is not None:
        try:
            _singleton_lock.close()
        except Exception:
            pass

    # 立刻结束进程，**不做解释器收尾**(Py_FinalizeEx)。
    # 原因: 调度器/流水线用 concurrent.futures.ThreadPoolExecutor，其 worker 是
    # 非守护线程；sys.exit() 会进入解释器收尾并 join 这些线程 —— 只要还有分析
    # 任务在跑，退出就会长时间"无响应"(实测采样: Py_FinalizeEx →
    # wait_for_thread_shutdown 卡住)。此处已完成自有清理(停调度器 / 终止 Ollama /
    # 释放单实例锁 / flush 日志)，直接退出即可。
    # 取舍: 正在跑的任务被中断，下次启动由调度器按"孤儿恢复"重新入队(1.2.32 起
    # 已有该机制)；尚未开始的任务保持原状态。
    try:
        logging.shutdown()
    except Exception:
        pass
    print("应用已退出")
    os._exit(exit_code if isinstance(exit_code, int) else 0)


if __name__ == "__main__":
    # PyInstaller 打包 + multiprocessing 必须调用 freeze_support()：
    # 缺失时 spawn 的子进程(resource_tracker 等)会重新执行整个入口,
    # 若顶层又触发新子进程 → 无限 fork 进程链爆炸
    # (2026-09 实测: 精简版 VM 上 MediaTag 进程链式增殖拖垮系统)。
    import multiprocessing
    multiprocessing.freeze_support()
    main()
