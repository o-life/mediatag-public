#!/bin/bash

# MediaTag 打包脚本
#
# 用法:
#   ./scripts/build.sh [full|lite|all]
#
#   full  (默认) 完整版 —— 内置人脸模型(AuraFace, DMG ~990M), 开箱即用
#   lite          轻量版 —— 不内置人脸模型(DMG ~280M), 首次运行通过
#                  「设置 → 环境初始化」从网络下载到 ~/.insightface(需联网)
#   all           两个版本都打包
#
# 两版都不内置 VLM: VLM 走本机 Ollama 服务, 由用户在设置页配置。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

VARIANT="${1:-full}"

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# 打包必须 `.venv 的 python -m PyInstaller` 直调：
# .venv/bin/pyinstaller 的 shebang 指向更名前路径已失效，而 `uv run pyinstaller`
# 会静默回落用户级 Python3.9 的 PyInstaller，hiddenimport 全丢、产物不可信
# （详见 memory: pyinstaller-venv-trap）。
if [ ! -x "$VENV_PYTHON" ]; then
    echo "错误: 未找到 .venv Python: $VENV_PYTHON"
    echo "请先运行: uv sync"
    exit 1
fi

echo "使用 venv Python: $($VENV_PYTHON --version)"
echo "使用 PyInstaller: $($VENV_PYTHON -m PyInstaller --version)"

# VLM_BUNDLE=1: 内置 Qwen3-VL 推理运行时与模型(models/qwen3-vl/, ≈5.4GB,
# 需先存在;配合 bundled_bin/llama_cpp——scripts/prepare_llama_bundle.sh 生成)。
# 公开售卖形态(add-public-sale-readiness 任务 1.1)默认开启;不置空则维持
# 旧两版(用户自配 Ollama)。
if [ "${VLM_BUNDLE:-1}" = "1" ]; then
    if [ ! -f "models/qwen3-vl/Qwen3VL-8B-Instruct-Q4_K_M.gguf" ]; then
        echo "错误: VLM_BUNDLE=1 但缺 models/qwen3-vl/Qwen3VL-8B-Instruct-Q4_K_M.gguf" >&2
        echo "  staging: cp -c ~/models/Qwen3VL-8B-Instruct-Q4_K_M.gguf models/qwen3-vl/" >&2
        exit 1
    fi
    if [ ! -x "bundled_bin/llama_cpp/bin/llama-server" ]; then
        echo "提示: 生成自包含 llama-server 运行时..." >&2
        bash "$SCRIPT_DIR/prepare_llama_bundle.sh"
    fi
    export VLM_BUNDLE=1
    echo "VLM 内置: 开启(模型 ≈5.4GB + llama-server 运行时)"
else
    export VLM_BUNDLE=0
    echo "VLM 内置: 关闭(旧形态,用户自配 Ollama)"
fi

build_one() {
    local spec="$1" distpath="$2" appname="$3" dmgname="${4:-$3}"
    echo ""
    echo "=========================================="
    echo "打包: $spec ($appname)"
    echo "=========================================="
    rm -rf build/ "$distpath"
    "$VENV_PYTHON" -m PyInstaller "$spec" --clean --noconfirm --distpath "$distpath"
    local app_path="$distpath/$appname.app"
    if [ -d "$app_path" ]; then
        local size
        size=$(du -sh "$app_path" | cut -f1)
        # macOS 26+ 官方新设计（Liquid Glass 窗口控件/标题栏等）由系统按主程序
        # 链接的 SDK 自动启用。PyInstaller bootloader 的 arm64 切片链接的是旧 SDK
        # (15.5)，系统因此按兼容模式渲染旧镶边。用 vtool 把 SDK 提到 27.0 触发
        # 官方新设计（2026-09-22 实验实证），并重签名。
        local exe_path="$app_path/Contents/MacOS/$appname"
        if [ -f "$exe_path" ] && xcrun vtool -set-build-version macos 11.0 27.0 -output "$exe_path.new" "$exe_path"; then
            mv "$exe_path.new" "$exe_path"
            codesign -f -s - --deep "$app_path"
            echo "✔ 已提升链接 SDK 至 27.0（系统新设计生效）并重签名"
        else
            echo "⚠ vtool SDK 提升失败，保留原二进制" >&2
        fi
        echo "✔ $appname.app 打包成功: $size"
        if [ -f "scripts/create_dmg.sh" ]; then
            bash scripts/create_dmg.sh "$distpath" "$appname" "$dmgname"
        fi
    else
        echo "✘ $appname.app 打包失败!" >&2
        exit 1
    fi
}

case "$VARIANT" in
    full)
        build_one video_tag.spec dist "MediaTag"
        ;;
    lite)
        # 两版 .app 同名 MediaTag; DMG 文件用不同名区分(完整版 vs Lite)
        build_one video_tag_lite.spec dist_lite "MediaTag" "MediaTag-Lite"
        ;;
    all)
        build_one video_tag.spec dist "MediaTag"
        build_one video_tag_lite.spec dist_lite "MediaTag" "MediaTag-Lite"
        ;;
    *)
        echo "未知变体: $VARIANT (可选 full|lite|all)" >&2
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "全部完成!"
echo "=========================================="
