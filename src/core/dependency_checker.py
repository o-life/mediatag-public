"""
依赖检测器 - 检查应用运行所需的依赖项

检测以下依赖：
1. Ollama - VLM 推理服务
2. FFmpeg - 视频处理
3. MiniCPM-V 模型 - 视觉语言模型
4. InsightFace 模型 - 人脸识别
"""

import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DependencyStatus:
    """依赖状态"""
    name: str
    installed: bool
    version: Optional[str] = None
    message: str = ""
    install_command: str = ""
    download_url: str = ""  # 官方下载链接


class DependencyChecker:
    """依赖检测器"""
    
    def __init__(self):
        self.dependencies: Dict[str, DependencyStatus] = {}
    
    def is_bundled_app(self) -> bool:
        """检测是否是打包后的应用"""
        return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
    
    def _get_bundled_bin_dir(self) -> Optional[Path]:
        """获取内置二进制目录"""
        if getattr(sys, 'frozen', False):
            # PyInstaller 打包后
            if hasattr(sys, '_MEIPASS'):
                # onefile 模式
                base = Path(sys._MEIPASS)
            else:
                # onedir 模式或 macOS app bundle
                exe_dir = Path(sys.executable).parent
                # macOS app bundle: exe 在 Contents/MacOS, bundled_bin 在 Contents/Resources
                macos_resources = exe_dir.parent / "Resources"
                if (macos_resources / "bundled_bin").exists():
                    base = macos_resources
                else:
                    # 其他平台或 onedir 模式
                    base = exe_dir
        else:
            # 开发环境
            base = Path(__file__).parent.parent.parent
        bundled_dir = base / "bundled_bin"
        return bundled_dir if bundled_dir.exists() else None
    
    def _get_bundled_ffmpeg_path(self) -> Optional[Path]:
        """获取内置 FFmpeg 路径"""
        bundled_dir = self._get_bundled_bin_dir()
        if not bundled_dir:
            return None
        # 查找 ffmpeg 二进制：LGPL 构建优先，兼容旧 GPL 布局
        for rel in ("ffmpeg_lgpl/bin", "8.1.2_1/bin"):
            p = bundled_dir / rel / "ffmpeg"
            if p.exists():
                return p
        return None
    
    def _get_bundled_ollama_path(self) -> Optional[Path]:
        """获取内置 Ollama 路径"""
        bundled_dir = self._get_bundled_bin_dir()
        if not bundled_dir:
            return None
        ollama_path = bundled_dir / "ollama"
        return ollama_path if ollama_path.exists() else None
    
    def check_all(self) -> Dict[str, DependencyStatus]:
        """检查所有依赖"""
        self.check_ollama()
        self.check_ffmpeg()
        self.check_minicpm_model()
        self.check_insightface_model()
        return self.dependencies
    
    def check_ollama(self) -> DependencyStatus:
        """检查 Ollama"""
        status = DependencyStatus(
            name="Ollama",
            installed=False,
            install_command="已内置（首次运行时自动启动）",
            download_url="https://ollama.com/download"
        )
        
        # 优先检查内置二进制
        bundled_ollama = self._get_bundled_ollama_path()
        if bundled_ollama and bundled_ollama.exists():
            status.installed = True
            status.message = "已内置"
            self.dependencies["ollama"] = status
            return status
        
        # 检查系统安装
        ollama_path = shutil.which("ollama")
        if not ollama_path:
            status.message = "未安装 Ollama"
            self.dependencies["ollama"] = status
            return status
        
        # 检查服务是否运行
        try:
            result = subprocess.run(
                [ollama_path, "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                status.installed = True
                status.message = "已安装并运行"
            else:
                status.message = "已安装但服务未运行"
        except subprocess.TimeoutExpired:
            status.message = "Ollama 服务响应超时"
        except Exception as e:
            status.message = f"检查失败: {str(e)}"
        
        self.dependencies["ollama"] = status
        return status
    
    def check_ffmpeg(self) -> DependencyStatus:
        """检查 FFmpeg"""
        status = DependencyStatus(
            name="FFmpeg",
            installed=False,
            install_command="已内置（LGPL 构建，随应用分发）",
            download_url="https://ffmpeg.org/download.html"
        )
        
        # 优先检查内置二进制
        bundled_ffmpeg = self._get_bundled_ffmpeg_path()
        if bundled_ffmpeg:
            status.installed = True
            status.message = "已内置"
            self.dependencies["ffmpeg"] = status
            return status
        
        # 检查系统安装
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            status.message = "未找到内置 FFmpeg（异常，请反馈）"
            self.dependencies["ffmpeg"] = status
            return status
        
        # 获取版本
        try:
            result = subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                status.installed = True
                # 提取版本号
                first_line = result.stdout.split('\n')[0]
                if 'version' in first_line:
                    status.version = first_line.split('version')[1].strip().split()[0]
                status.message = f"已安装 (版本 {status.version})"
            else:
                status.message = "安装异常"
        except Exception as e:
            status.message = f"检查失败: {str(e)}"
        
        self.dependencies["ffmpeg"] = status
        return status
    
    def check_minicpm_model(self) -> DependencyStatus:
        """检查 MiniCPM-V 模型"""
        status = DependencyStatus(
            name="MiniCPM-V 模型",
            installed=False,
            install_command="ollama run minicpm-v4.5",
            download_url="https://ollama.com/library/minicpm-v"
        )
        
        # 优先使用内置 Ollama
        ollama_path = self._get_bundled_ollama_path()
        if not ollama_path:
            # 回退到系统 Ollama
            ollama_path = shutil.which("ollama")
        
        if not ollama_path:
            status.message = "需要先安装 Ollama"
            self.dependencies["minicpm_model"] = status
            return status
        
        # 检查模型是否存在
        try:
            result = subprocess.run(
                [ollama_path, "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                models = result.stdout.lower()
                if "minicpm-v" in models:
                    status.installed = True
                    status.message = "已下载"
                else:
                    status.message = "未下载，首次运行时自动下载（约 5GB）"
            else:
                status.message = "无法查询模型列表"
        except Exception as e:
            status.message = f"检查失败: {str(e)}"
        
        self.dependencies["minicpm_model"] = status
        return status
    
    def check_insightface_model(self) -> DependencyStatus:
        """检查人脸模型

        全量 AuraFace（Apache-2.0 可商用）：检测 + 识别 + 性别年龄。
        """
        status = DependencyStatus(
            name="人脸模型",
            installed=False,
            install_command="",  # 自动下载，无需手动安装
            download_url=""
        )

        base = Path.home() / ".insightface" / "models"
        # AuraFace 关键文件
        aura_ok = all((base / "auraface" / f).exists() for f in
                      ("scrfd_10g_bnkps.onnx", "glintr100.onnx", "genderage.onnx"))

        if aura_ok:
            status.installed = True
            status.message = "已就绪 (auraface)"
        else:
            status.message = "未下载，可在欢迎页/设置→环境初始化下载（约 600MB）"
        
        self.dependencies["insightface_model"] = status
        return status
    
    def get_missing_dependencies(self) -> List[DependencyStatus]:
        """获取缺失的依赖列表"""
        if not self.dependencies:
            self.check_all()
        
        return [dep for dep in self.dependencies.values() if not dep.installed]
    
    def all_dependencies_installed(self) -> bool:
        """检查所有依赖是否已安装"""
        if not self.dependencies:
            self.check_all()
        
        return all(dep.installed for dep in self.dependencies.values())
    
    def get_install_summary(self) -> str:
        """获取安装摘要"""
        if not self.dependencies:
            self.check_all()
        
        total = len(self.dependencies)
        installed = sum(1 for dep in self.dependencies.values() if dep.installed)
        missing = total - installed
        
        summary = f"依赖检查完成：{installed}/{total} 已安装"
        if missing > 0:
            summary += f"，{missing} 项需要安装"
        
        return summary


# 便捷函数
def check_dependencies() -> Dict[str, DependencyStatus]:
    """检查所有依赖"""
    checker = DependencyChecker()
    return checker.check_all()


def get_missing_dependencies() -> List[DependencyStatus]:
    """获取缺失的依赖"""
    checker = DependencyChecker()
    return checker.get_missing_dependencies()


def all_dependencies_ready() -> bool:
    """检查所有依赖是否就绪"""
    checker = DependencyChecker()
    return checker.all_dependencies_installed()
