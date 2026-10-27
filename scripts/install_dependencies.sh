#!/bin/bash
# =============================================================================
# MediaTag - 依赖安装(内嵌推理架构,2026-10)
# =============================================================================
# 新架构不需要 Ollama:推理由打包内置的 llama.cpp + Qwen3-VL GGUF 完成。
# 本脚本安装:
#   1. Homebrew(macOS 包管理)
#   2. uv(Python 环境)
#   3. FFmpeg(视频处理)
#   4. llama.cpp(llama-server,打包时由 prepare_llama_bundle.sh 自包含化)
#   5. Python 依赖(uv sync)
#
# 模型权重不入库,构建前需下载(见 README「Build」段):
#   - Qwen3-VL-8B GGUF Q4_K_M + mmproj(约 5.5GB)
#   - AuraFace 人脸模型包(约 487MB,应用内「设置 → 环境初始化」亦可下载)
#
# Usage: ./scripts/install_dependencies.sh [--skip-brew] [--skip-ffmpeg] [--quiet]
# =============================================================================

set -e

# ── Homebrew ──────────────────────────────────────────────────────────────────
if ! command -v brew >/dev/null 2>&1; then
    if [[ "${1:-}" != "--skip-brew" ]]; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        echo "[错误] 未安装 Homebrew(--skip-brew 跳过了安装)" >&2
        exit 1
    fi
fi
echo "[OK] Homebrew"

# ── uv + Python 依赖 ─────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    brew install uv
fi
echo "[OK] uv"
uv sync
echo "[OK] Python 依赖(uv sync)"

# ── FFmpeg ───────────────────────────────────────────────────────────────────
if [[ "${1:-}" != "--skip-ffmpeg" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
    brew install ffmpeg
fi
echo "[OK] FFmpeg"

# ── llama.cpp(llama-server)─────────────────────────────────────────────────
if ! command -v llama-server >/dev/null 2>&1; then
    brew install llama.cpp
fi
echo "[OK] llama.cpp(开发态;打包时 prepare_llama_bundle.sh 会自包含化)"

# ── huggingface-cli(模型权重下载用)────────────────────────────────────────
uv run --active python -m pip install --quiet "huggingface_hub[cli]" 2>/dev/null || \
    brew install huggingface-cli 2>/dev/null || true
echo "[OK] huggingface-cli"

echo ""
echo "依赖安装完成。下一步:"
echo "  1. 下载模型权重(见 README「Build」段:GGUF 约 5.5GB + AuraFace 约 487MB)"
echo "  2. bash scripts/build.sh"
