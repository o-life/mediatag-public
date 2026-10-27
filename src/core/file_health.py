"""文件解码健康体检（add-public-sale-readiness 任务 2.1）。

方法论源自 2026-10-06 存量体检（docs/数据质量/存量文件体检-20261006.csv，
2,464 个视频揪出 201 个坏文件的同一套判定）：

- ffprobe 读头**不可作为判据**——尾部截断型坏文件能读出时长、能播前段;
- 判定 = 10%/50%/90% 三点各实解最多 1 秒,该点健康 = 实解出帧 ≥1 且无
  错误模式（只看退出码会把"干净 EOF 0 帧"误判为健康）;
- 三点全败 = 全坏;含失败点 = 部分损坏;容器时长读不出且 0s 解码失败 = 容器损坏;
  非视频扩展名 / 文件不存在单独归类。

判定语义（与分析管线约定）：
- 容器损坏 / 全坏 / 文件缺失 → 不可分析（调用方跳过并标记"文件损坏"）;
- 部分损坏 → 可继续分析（完好段可出帧）,调用方应记旗供 UI 呈现。

二进制可注入（测试用假脚本,不依赖真 ffmpeg）;打包态优先用随包
ffmpeg_lgpl,开发态回退 PATH。
"""

import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".wmv", ".flv",
             ".ts", ".m4v", ".mpg", ".mpeg"}
_ERR_PAT = ("corrupt", "invalid", "error", "partial", "moov", "failed",
            "no such file", "not found")
_FRAME_RE = re.compile(r"frame=\s*(\d+)")


def _default_bin(name: str) -> str:
    """打包态优先随包 ffmpeg_lgpl,开发态回退 PATH。"""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            cand = Path(meipass) / "bundled_bin" / "ffmpeg_lgpl" / "bin" / name
            if cand.exists():
                return str(cand)
    found = shutil.which(name)
    return found or name


class HealthChecker:
    def __init__(self, ffprobe: str = None, ffmpeg: str = None,
                 positions=(0.10, 0.50, 0.90), timeout: int = 60):
        self.ffprobe = ffprobe or _default_bin("ffprobe")
        self.ffmpeg = ffmpeg or _default_bin("ffmpeg")
        self.positions = positions
        self.timeout = timeout

    def _sh(self, args, timeout=None):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  errors="replace", timeout=timeout or self.timeout)
        except subprocess.TimeoutExpired:
            return None

    def check(self, path: str) -> dict:
        """返回 {verdict, detail}。

        verdict ∈ ok / 部分损坏 / 全坏 / 容器损坏 / 文件缺失 / 图片跳过
        """
        p = Path(path)
        if p.suffix.lower() not in VIDEO_EXT:
            return {"verdict": "图片跳过", "detail": ""}
        if not p.exists():
            return {"verdict": "文件缺失", "detail": path}

        dur = self._probe_duration(path)
        if dur is None:
            ok, msg = self._decode_at(path, 0)
            if ok:
                return {"verdict": "ok", "detail": ""}
            return {"verdict": "容器损坏", "detail": msg}

        positions = [max(0.0, dur * f) for f in self.positions]
        bad = []
        for pos in positions:
            ok, msg = self._decode_at(path, pos)
            if not ok:
                bad.append(f"{pos:.0f}s:{msg}")
        if not bad:
            return {"verdict": "ok", "detail": ""}
        if len(bad) == len(positions):
            return {"verdict": "全坏", "detail": "; ".join(bad[:2])}
        return {"verdict": "部分损坏", "detail": "; ".join(bad)}

    def _probe_duration(self, path):
        r = self._sh([self.ffprobe, "-v", "error", "-show_entries",
                      "format=duration", "-of", "csv=p=0", path])
        if r and r.returncode == 0 and r.stdout.strip():
            try:
                return float(r.stdout.strip().split(",")[0])
            except ValueError:
                return None
        return None

    def _decode_at(self, path, ss):
        """在 ss 秒处实解最多 1 秒,返回 (是否健康, 错误摘要)。"""
        r = self._sh([self.ffmpeg, "-v", "info", "-ss", f"{ss:.2f}", "-i", path,
                      "-t", "1", "-f", "null", "-"], timeout=self.timeout + 30)
        if r is None:
            return False, "解码超时"
        err = r.stderr or ""
        frames = 0
        for m in _FRAME_RE.finditer(err):
            frames = int(m.group(1))
        low = err.lower()
        if frames >= 1 and not any(p in low for p in _ERR_PAT):
            return True, ""
        if frames == 0:
            first = next((ln for ln in err.splitlines() if ln.strip()), "")
            return False, f"0帧可解({first[:60]})" if first else "0帧可解(干净EOF)"
        first = next((ln for ln in err.splitlines()
                      if any(p in ln.lower() for p in _ERR_PAT)),
                     err.splitlines()[0] if err else "")
        return False, first[:90]
