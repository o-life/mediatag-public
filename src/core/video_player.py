"""视频播放器服务 - 支持自定义播放器和进度跳转，单窗口复用"""

import subprocess
import re
import socket
import json
from pathlib import Path
from typing import Optional


class VideoPlayerService:
    """视频播放器服务 - 支持 IINA 单窗口复用"""
    
    # IINA IPC socket 路径
    IPC_SOCKET_PATH = "/tmp/iina-ipc-socket"
    
    def __init__(self, player_type: str = "系统默认", player_path: str = ""):
        """
        初始化播放器服务
        
        Args:
            player_type: 播放器类型 (iina/mpv/vlc/系统默认)
            player_path: 播放器路径（.app 路径）
        """
        self.player_type = player_type
        self.player_path = player_path
        # 最近一次降级说明（S13 ⑩：非空=本次播放发生了"回退系统默认"）
        self.fallback_notice = ""
    
    def play(self, video_path: str, timestamp: Optional[float] = None):
        """
        播放视频 - 优先复用现有窗口
        
        Args:
            video_path: 视频文件路径
            timestamp: 起始时间戳（秒），None 表示从头播放
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        self.fallback_notice = ""
        
        if self.player_type == "系统默认" or not self.player_path:
            # 使用系统默认播放器
            self._play_with_default(video_path)
        else:
            # 使用自定义播放器（优先复用窗口）
            self._play_with_custom(video_path, timestamp)
    
    def _play_with_default(self, video_path: str):
        """使用系统默认播放器"""
        subprocess.run(["open", video_path], check=True)
    
    def _is_iina_running(self) -> bool:
        """检查 IINA 是否正在运行"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "IINA.app"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _send_ipc_command(self, command: dict) -> bool:
        """通过 IPC socket 发送命令到 IINA"""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(self.IPC_SOCKET_PATH)
            
            # 发送 JSON 命令
            cmd_json = json.dumps(command) + "\n"
            sock.sendall(cmd_json.encode('utf-8'))
            
            # 读取响应
            sock.recv(4096).decode('utf-8')  # 等待并丢弃响应（同步握手）
            sock.close()
            
            return True
        except Exception:
            return False
    
    def _play_with_custom(self, video_path: str, timestamp: Optional[float]):
        # S13 ⑩：选中播放器未安装 → 不再静默失败，回退系统默认打开并留提示
        _std_apps = {
            "iina": "/Applications/IINA.app",
            "mpv": "/Applications/mpv.app",
            "vlc": "/Applications/VLC.app",
        }
        _candidate = self.player_path or _std_apps.get(self.player_type, "")
        if _candidate and not Path(_candidate).exists():
            self.fallback_notice = (
                f"未找到播放器 {self.player_type}，已用系统默认打开（无跳转功能）。"
                "安装到「应用程序」文件夹后将自动启用跳转。")
            import logging
            logging.getLogger(__name__).warning(self.fallback_notice)
            self._play_with_default(video_path)
            return
        """使用自定义播放器 - 优先复用现有窗口"""
        import logging
        logger = logging.getLogger(__name__)
        
        player_path = self.player_path
        
        # 如果路径为空，尝试自动检测标准安装路径
        if not player_path:
            auto_paths = {
                "iina": "/Applications/IINA.app",
                "mpv": "/Applications/mpv.app",
                "vlc": "/Applications/VLC.app"
            }
            if self.player_type in auto_paths:
                player_path = auto_paths[self.player_type]
        
        if not player_path:
            # 回退到系统默认
            self._play_with_default(video_path)
            return
        
        app_path = Path(player_path)
        
        # macOS .app 包 - 使用 iina-cli（IINA 专用命令行工具）
        if app_path.suffix == '.app' or (app_path / "Contents").exists():
            # 获取 iina-cli 路径
            iina_cli_path = app_path / "Contents" / "MacOS" / "iina-cli"
            
            if iina_cli_path.exists():
                # 检查 IINA 是否已运行且有 IPC socket
                if self.player_type == "iina" and self._is_iina_running():
                    # 尝试通过 IPC 发送命令
                    if self._send_ipc_command({"command": ["loadfile", video_path, "replace"]}):
                        logger.info(f"通过 IPC 复用窗口: {video_path}")
                        
                        # 设置起始时间
                        if timestamp is not None and timestamp > 0:
                            self._send_ipc_command({"command": ["set", "start", str(timestamp)]})
                            logger.info(f"设置起始时间: {timestamp}")
                        
                        return
                
                # 启动新的 IINA 实例，创建 IPC socket
                cmd = [str(iina_cli_path)]
                
                # 明确禁用 stdin
                cmd.append("--no-stdin")
                
                # 创建 IPC socket（仅 IINA）
                if self.player_type == "iina":
                    cmd.append(f"--mpv-input-ipc-server={self.IPC_SOCKET_PATH}")
                
                # 构建播放器参数
                if timestamp is not None and timestamp > 0:
                    if self.player_type == "iina":
                        cmd.append(f"--mpv-start={timestamp}")
                        cmd.append("--mpv-resume-playback=no")
                        cmd.append("--mpv-no-osc")  # 隐藏控制栏
                    elif self.player_type == "mpv":
                        cmd.append(f"--mpv-start={timestamp}")
                        cmd.append("--mpv-resume-playback=no")
                    elif self.player_type == "vlc":
                        cmd.append(f"--mpv-start-time={timestamp}")
                
                cmd.append(video_path)
                
                logger.info(f"启动新的 IINA 实例 (iina-cli): {' '.join(cmd)}")
                subprocess.Popen(cmd)
                return
            
            # 回退到直接调用 IINA 可执行文件
            macos_dir = app_path / "Contents" / "MacOS"
            if macos_dir.exists():
                # 查找可执行文件
                exec_path = None
                for item in macos_dir.iterdir():
                    if item.is_file() and not item.name.startswith('.') and item.name != 'iina-cli':
                        exec_path = str(item)
                        break
                
                if exec_path:
                    # 检查是否可以通过 IPC 复用
                    if self.player_type == "iina" and self._is_iina_running():
                        if self._send_ipc_command({"command": ["loadfile", video_path, "replace"]}):
                            logger.info(f"通过 IPC 复用窗口: {video_path}")
                            if timestamp is not None and timestamp > 0:
                                self._send_ipc_command({"command": ["set", "start", str(timestamp)]})
                            return
                    
                    # 启动新实例
                    cmd = [exec_path]
                    
                    # 创建 IPC socket
                    if self.player_type == "iina":
                        cmd.append(f"--input-ipc-server={self.IPC_SOCKET_PATH}")
                    
                    if timestamp is not None and timestamp > 0:
                        if self.player_type == "iina":
                            cmd.append(f"--mpv-start={timestamp}")
                        elif self.player_type == "mpv":
                            cmd.extend(["--start", str(timestamp)])
                        elif self.player_type == "vlc":
                            cmd.extend(["--start-time", str(timestamp)])
                    
                    cmd.append(video_path)
                    logger.info(f"启动新的 IINA 实例: {' '.join(cmd)}")
                    subprocess.Popen(cmd)
                    return
            
            # 回退到 open -a 方式
            cmd = ["open", "-a", str(app_path)]
            
            player_args = []
            if timestamp is not None and timestamp > 0:
                if self.player_type == "iina":
                    player_args.append(f"--mpv-start={timestamp}")
                elif self.player_type == "mpv":
                    player_args.extend(["--start", str(timestamp)])
                elif self.player_type == "vlc":
                    player_args.extend(["--start-time", str(timestamp)])
            
            if player_args:
                cmd.extend(["--args"] + player_args)
            
            cmd.append(video_path)
            logger.info(f"执行命令 (open -a): {' '.join(cmd)}")
            subprocess.Popen(cmd)
        else:
            # 直接的可执行文件
            player_exec = str(app_path)
            if not app_path.is_file():
                self._play_with_default(video_path)
                return
            
            # 检查是否可以通过 IPC 复用
            if self.player_type == "iina" and self._is_iina_running():
                if self._send_ipc_command({"command": ["loadfile", video_path, "replace"]}):
                    logger.info(f"通过 IPC 复用窗口: {video_path}")
                    if timestamp is not None and timestamp > 0:
                        self._send_ipc_command({"command": ["set", "start", str(timestamp)]})
                    return
            
            cmd = [player_exec]
            
            # 创建 IPC socket
            if self.player_type == "iina":
                cmd.append(f"--input-ipc-server={self.IPC_SOCKET_PATH}")
            
            if timestamp is not None and timestamp > 0:
                if self.player_type == "iina":
                    cmd.append(f"--mpv-start={timestamp}")
                elif self.player_type == "mpv":
                    cmd.extend(["--start", str(timestamp)])
                elif self.player_type == "vlc":
                    cmd.extend(["--start-time", str(timestamp)])
            
            cmd.append(video_path)
            logger.info(f"执行命令: {' '.join(cmd)}")
            subprocess.Popen(cmd)
    
    def _get_player_executable(self) -> Optional[str]:
        """获取播放器可执行文件路径"""
        # 如果路径为空，尝试自动检测标准安装路径
        if not self.player_path:
            auto_paths = {
                "iina": "/Applications/IINA.app",
                "mpv": "/Applications/mpv.app",
                "vlc": "/Applications/VLC.app"
            }
            if self.player_type in auto_paths:
                self.player_path = auto_paths[self.player_type]
        
        if not self.player_path:
            return None
        
        app_path = Path(self.player_path)
        
        # macOS .app 包结构
        macos_dir = app_path / "Contents" / "MacOS"
        if macos_dir.exists():
            # 查找可执行文件
            for item in macos_dir.iterdir():
                if item.is_file() and not item.name.startswith('.'):
                    return str(item)
        
        # 如果是直接的可执行文件
        if app_path.is_file():
            return str(app_path)
        
        return None
    
    @staticmethod
    def calculate_timestamp(frame_path: str, frames_dir: str, frames_per_scene: int = 25) -> Optional[float]:
        """
        根据帧路径计算时间戳
        
        Args:
            frame_path: 帧图片路径
            frames_dir: 帧目录
            frames_per_scene: 每个场景的帧数
            
        Returns:
            时间戳（秒），None 表示无法计算
        """
        try:
            # 解析帧路径: /path/to/video_frames/unit_X/frame_Y.jpg
            frame_file = Path(frame_path).name
            match = re.match(r'frame_(\d+)\.jpg', frame_file)
            if not match:
                return None
            
            frame_number = int(match.group(1))
            
            # 解析 unit 目录
            unit_dir = Path(frame_path).parent
            unit_match = re.match(r'unit_(\d+)', unit_dir.name)
            if not unit_match:
                return None
            
            unit_index = int(unit_match.group(1))
            
            # 计算全局帧索引
            global_frame_index = unit_index * frames_per_scene + frame_number
            
            # 假设采样间隔为 2 秒（需要根据实际配置调整）
            # TODO: 从配置或数据库中读取实际的采样间隔
            sample_interval = 2.0
            timestamp = global_frame_index * sample_interval
            
            return timestamp
        except Exception:
            return None
