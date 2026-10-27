#!/bin/bash
# Launch script for AI Video Factory Desktop App
# Fixes Qt platform plugin path issues on macOS

cd "$(dirname "$0")"

# 优先使用仓库内置 FFmpeg（与打包版行为一致，避免 dev 依赖系统 PATH）
_BUNDLED_FFMPEG_BIN="$(ls -d "$(pwd)"/bundled_bin/ffmpeg_lgpl/bin 2>/dev/null | head -1)"
if [ -n "$_BUNDLED_FFMPEG_BIN" ]; then
    export PATH="$_BUNDLED_FFMPEG_BIN:$PATH"
fi

# Set Qt environment variables
export QT_QPA_PLATFORM=cocoa
export QT_QPA_PLATFORM_PLUGIN_PATH="$(pwd)/.venv/lib/python3.11/site-packages/PySide6/Qt/plugins/platforms"

# Set DYLD_LIBRARY_PATH to fix cv2 FFmpeg library conflicts on macOS
export DYLD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.11/site-packages/cv2/.dylibs:${DYLD_LIBRARY_PATH:-}"

# Disable Python buffering for better logging
export PYTHONUNBUFFERED=1

# Set process priority
NICE_CMD=""
if command -v nice &> /dev/null; then
    NICE_CMD="nice -n 0"
fi

# Start the desktop app
echo "启动桌面应用..."

# Redirect stdin from /dev/null to avoid zle errors in non-interactive mode
if [ ! -t 0 ]; then
    exec 0</dev/null 2>/dev/null
fi

$NICE_CMD uv run --active python src/ui/desktop_app.py "$@"
