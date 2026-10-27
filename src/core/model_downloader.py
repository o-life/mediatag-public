"""Model downloader with progress tracking and retry mechanism."""

import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Callable
from PySide6.QtCore import QThread, Signal


@dataclass(frozen=True)
class DownloadProgress:
    """Immutable download progress information."""
    percent: float  # 0.0 to 100.0
    downloaded: str  # e.g., "2.5GB"
    total: str  # e.g., "5.0GB"
    speed: str  # e.g., "10MB/s"
    status: str  # e.g., "pulling manifest", "downloading"


_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}


def _parse_size(text: str) -> int:
    """把 ollama 的大小文本(如 '881 MB', '5.0GB')解析为字节数; 失败返回 0。"""
    m = re.match(r'([\d.]+)\s*([KMGTP]?B)', text.strip())
    if not m:
        return 0
    try:
        val = float(m.group(1))
    except ValueError:
        return 0
    unit = m.group(2).upper()
    if unit == "B":
        return int(val)
    return int(val * _SIZE_UNITS.get(unit, 1024))


def _fmt_size(num_bytes: int) -> str:
    """字节数 → 可读文本('881 MB'/'5.0 GB')。"""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


class ModelDownloadWorker(QThread):
    """Worker thread for downloading models with progress tracking."""

    # Signals
    progress_updated = Signal(object)  # DownloadProgress
    download_completed = Signal(bool, str)  # success, message
    download_started = Signal(str)  # model name

    def __init__(
        self,
        model_name: str,
        ollama_path: str,
        max_retries: int = 3,
        parent=None
    ):
        super().__init__(parent)
        self.model_name = model_name
        self.ollama_path = ollama_path
        self.max_retries = max_retries
        self._is_cancelled = False

    def cancel(self):
        """Cancel the download operation."""
        self._is_cancelled = True

    def run(self):
        """Execute the download with retry logic."""
        self.download_started.emit(self.model_name)

        for attempt in range(1, self.max_retries + 1):
            if self._is_cancelled:
                self.download_completed.emit(False, "下载已取消")
                return

            try:
                success = self._execute_download(attempt)
                if success:
                    self.download_completed.emit(True, "模型下载完成")
                    return

                # If not cancelled and not last attempt, wait before retry
                if attempt < self.max_retries and not self._is_cancelled:
                    wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                    time.sleep(wait_time)

            except Exception as e:
                if self._is_cancelled:
                    self.download_completed.emit(False, "下载已取消")
                    return

                if attempt == self.max_retries:
                    self.download_completed.emit(False, f"下载失败（已重试{self.max_retries}次）: {str(e)}")
                    return

        self.download_completed.emit(False, "下载失败：超过最大重试次数")

    def _execute_download(self, attempt: int) -> bool:
        """Execute a single download attempt. Returns True if successful."""
        env = os.environ.copy()

        process = subprocess.Popen(
            [self.ollama_path, 'pull', self.model_name],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            # ollama pull 的进度用 \r 原地刷新(非 \n 换行): 按「行」迭代会
            # 收不到中间帧(全部积压在缓冲, 直到进程结束才有数据)。
            # → 用原始字节流读取, 按 \r/\n 切帧, 每帧即时解析。
            buf = b""
            while True:
                if self._is_cancelled:
                    process.terminate()
                    return False
                chunk = process.stderr.read(4096)
                if not chunk:
                    break
                buf += chunk
                # 按 \r 或 \n 切出完整帧
                while b"\r" in buf or b"\n" in buf:
                    # 找最早的分隔符
                    i_r = buf.find(b"\r")
                    i_n = buf.find(b"\n")
                    if i_n != -1 and (i_r == -1 or i_n < i_r):
                        frame, buf = buf[:i_n], buf[i_n + 1:]
                    else:
                        frame, buf = buf[:i_r], buf[i_r + 1:]
                    self._handle_pull_frame(frame)

            # 处理尾部残留(无 \r/\n 的最后一帧)
            if buf.strip():
                self._handle_pull_frame(buf)

            # Wait for process to complete
            process.wait(timeout=1800)  # 30 minutes timeout

            return process.returncode == 0

        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
            raise TimeoutError("下载超时（超过30分钟）")
        except Exception as e:
            process.terminate()
            process.wait(timeout=5)
            raise e

    def _handle_pull_frame(self, raw: bytes) -> None:
        """处理一帧 ollama pull 输出(原始字节)。

        进度条显示「当前 layer 的百分比」——ollama 每帧是当前在拉
        的 layer 进度, 不做跨 layer 聚合(多 layer 时显示最近帧的 layer,
        每个 layer 各自 0→100%, 简单可靠)。
        """
        try:
            line = raw.decode("utf-8", errors="replace")
        except Exception:
            return
        progress = self._parse_progress(line)
        if progress is not None:
            self.progress_updated.emit(progress)

    def _parse_progress(self, line: str) -> Optional[DownloadProgress]:
        """Parse progress information from ollama pull output."""
        # Remove ANSI escape sequences (terminal control characters)
        # Matches: \033[...m, \033[K, \033[?25h, etc.
        line = re.sub(r'\033\[[0-9;]*[a-zA-Z]', '', line)
        line = re.sub(r'\033\[\?[0-9;]*[a-zA-Z]', '', line)
        # Remove carriage returns and extra whitespace
        line = line.replace('\r', ' ').strip()

        if not line:
            return None

        # 注意顺序: 必须先匹配「带百分比/字节数」的进度行, 再回落纯状态行——
        # 历史 bug: ollama pull 的进度行以 "pulling" 开头, 先匹配状态行会把
        # 真实百分比吞掉, 进度条恒 0%。
        # 进度帧示例(ollama 实际输出, 无 | 分隔, 大小数字与单位间有空格):
        #   "pulling afbc1adb650e:  32% ████░░ 1.6 GB/5.0 GB"
        #   "pulling afbc1adb650e:  18% | 881 MB/5.0 GB | 1.5 MB/s | 46m49s"(旧版)
        # 方案: 独立提取 百分比 与 "已下/总" 大小对, 不要求两者相邻/固定分隔。

        # 1) 百分比: 任意位置 "NN%" (进度帧一定有)
        pct_m = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
        # 2) 大小对: "X GB/Y GB" 或 "XGB/YGB" (空格可选)
        size_m = re.search(
            r'([\d.]+\s*[KMGTP]?B)\s*/\s*([\d.]+\s*[KMGTP]?B)',
            line,
        )
        # 3) 速度: "NN MB/s"
        speed_m = re.search(r'([\d.]+\s*[KMGTP]?B/s)', line)

        if pct_m and size_m:
            percent = float(pct_m.group(1))
            downloaded = re.sub(r'\s+', '', size_m.group(1))  # "1.6 GB"→"1.6GB"
            total = re.sub(r'\s+', '', size_m.group(2))
            speed = speed_m.group(1).replace(" ", "") if speed_m else ""
            # 状态取 "pulling <digest>" 前缀(去掉字符画/尺寸等冗余)
            prefix = line[: pct_m.start()].strip()
            if len(prefix) > 60:
                prefix = prefix[:60] + "…"
            return DownloadProgress(
                percent=percent,
                downloaded=downloaded,
                total=total,
                speed=speed,
                status=prefix or "downloading",
            )

        # 纯状态行("pulling manifest...", "verifying...", "writing..." 等)
        if line.startswith('pulling ') or line.startswith('verifying ') or line.startswith('writing '):
            # 去掉 ollama spinner(Braille U+2800-U+28FF)残留
            status = re.sub(r'[\u2800-\u28FF]', '', line).strip().rstrip('.')
            return DownloadProgress(
                percent=-1.0,  # -1: 状态行, UI 不动进度条只更新文本
                downloaded="",
                total="",
                speed="",
                status=status,
            )

        return None


class ModelDownloader:
    """High-level model downloader with UI integration."""

    def __init__(self, ollama_path: str, max_retries: int = 3):
        self.ollama_path = ollama_path
        self.max_retries = max_retries
        self._worker: Optional[ModelDownloadWorker] = None

    def download_async(
        self,
        model_name: str,
        on_progress: Callable[[DownloadProgress], None],
        on_completed: Callable[[bool, str], None],
        on_started: Optional[Callable[[str], None]] = None
    ) -> ModelDownloadWorker:
        """Start an asynchronous download with callbacks."""
        self._worker = ModelDownloadWorker(
            model_name=model_name,
            ollama_path=self.ollama_path,
            max_retries=self.max_retries
        )

        self._worker.progress_updated.connect(on_progress)
        self._worker.download_completed.connect(on_completed)

        if on_started:
            self._worker.download_started.connect(on_started)

        self._worker.start()
        return self._worker

    def cancel(self):
        """Cancel the current download."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(timeout=5)

    def is_downloading(self) -> bool:
        """Check if a download is in progress."""
        return self._worker is not None and self._worker.isRunning()
