#!/bin/bash
# Launch script for AI Video Factory API
# Fixes cv2 FFmpeg library conflicts on macOS

cd "$(dirname "$0")"

# 优先使用仓库内置 FFmpeg（与打包版行为一致，避免 dev 依赖系统 PATH）
_BUNDLED_FFMPEG_BIN="$(ls -d "$(pwd)"/bundled_bin/ffmpeg_lgpl/bin 2>/dev/null | head -1)"
if [ -n "$_BUNDLED_FFMPEG_BIN" ]; then
    export PATH="$_BUNDLED_FFMPEG_BIN:$PATH"
    echo "使用内置 FFmpeg: $_BUNDLED_FFMPEG_BIN/ffmpeg"
fi

# Set DYLD_LIBRARY_PATH to fix cv2 FFmpeg library conflicts on macOS
export DYLD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/cv2/.dylibs:${DYLD_LIBRARY_PATH:-}"

# Disable Python buffering for better logging
export PYTHONUNBUFFERED=1

# Set process priority (lower number = higher priority)
# nice -n 0 is normal priority
NICE_CMD=""
if command -v nice &> /dev/null; then
    NICE_CMD="nice -n 0"
fi

# Check if port 8000 is already in use
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "警告: 端口 8000 已被占用，尝试清理..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 1
fi

# Start the API server (without --reload for stability)
echo "启动 API 服务器..."
$NICE_CMD uv run --active uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
