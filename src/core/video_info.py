"""视频元数据获取工具（文件大小、时长）"""

import os
import json
import subprocess
from typing import Optional, Tuple


def get_video_duration(file_path: str) -> Optional[float]:
    """使用 ffprobe 获取视频时长（秒）

    Args:
        file_path: 视频文件路径

    Returns:
        视频时长（秒），失败返回 None
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            file_path
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration_str = data.get('format', {}).get('duration')
            if duration_str:
                return float(duration_str)
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, FileNotFoundError):
        return None


def get_video_info(file_path: str) -> Tuple[int, Optional[float]]:
    """获取文件大小和时长

    Args:
        file_path: 视频文件路径

    Returns:
        (file_size, duration) 元组
        - file_size: 文件大小（字节），文件不存在返回 0
        - duration: 视频时长（秒），获取失败返回 None
    """
    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        file_size = 0

    duration = get_video_duration(file_path)

    return file_size, duration
