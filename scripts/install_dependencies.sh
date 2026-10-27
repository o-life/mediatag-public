#!/bin/bash
# =============================================================================
# AI Video Factory - Dependency Installation Script
# =============================================================================
# This script installs all required dependencies for AI Video Factory:
#   1. Homebrew (macOS package manager)
#   2. FFmpeg (video processing)
#   3. Ollama (local LLM runtime)
#   4. MiniCPM-V model (vision-language model)
#
# Usage:
#   chmod +x scripts/install_dependencies.sh
#   ./scripts/install_dependencies.sh
#
# Options:
#   --skip-brew      Skip Homebrew installation
#   --skip-ffmpeg    Skip FFmpeg installation
#   --skip-ollama    Skip Ollama installation
#   --skip-model     Skip model download
#   --skip-face      Skip face recognition dependencies
#   --model NAME     Specify model name (default: minicpm-v:latest)
#   --quiet          Suppress non-essential output
#   --help           Show this help message
# =============================================================================

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-minicpm-v:latest}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── Parse arguments ───────────────────────────────────────────────────────────
SKIP_BREW=false
SKIP_FFMPEG=false
SKIP_OLLAMA=false
SKIP_MODEL=false
SKIP_FACE=false
QUIET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-brew)
            SKIP_BREW=true
            shift
            ;;
        --skip-ffmpeg)
            SKIP_FFMPEG=true
            shift
            ;;
        --skip-ollama)
            SKIP_OLLAMA=true
            shift
            ;;
        --skip-model)
            SKIP_MODEL=true
            shift
            ;;
        --skip-face)
            SKIP_FACE=true
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ── Helper functions ──────────────────────────────────────────────────────────
log_info() {
    if [[ "$QUIET" == "false" ]]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}==> $1${NC}"
}

check_command() {
    command -v "$1" &> /dev/null
}

# ── Pre-flight checks ────────────────────────────────────────────────────────
preflight_check() {
    log_step "Running pre-flight checks..."

    # Check macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        log_error "This script is designed for macOS only."
        log_warn "For other platforms, please install dependencies manually:"
        echo "  - FFmpeg: https://ffmpeg.org/download.html"
        echo "  - Ollama: https://ollama.com/download"
        exit 1
    fi

    log_success "macOS detected: $(sw_vers -productVersion)"

    # Check disk space (need at least 5GB free)
    local free_gb
    free_gb=$(df -g / | tail -1 | awk '{print $4}')
    if [[ "$free_gb" -lt 5 ]]; then
        log_warn "Low disk space: ${free_gb}GB free (recommended: 5GB+)"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "Disk space: ${free_gb}GB free"
    fi
}

# ── Install Homebrew ──────────────────────────────────────────────────────────
install_homebrew() {
    if [[ "$SKIP_BREW" == "true" ]]; then
        log_info "Skipping Homebrew installation (--skip-brew)"
        return 0
    fi

    log_step "Checking Homebrew..."

    if check_command brew; then
        log_success "Homebrew already installed: $(brew --version | head -1)"
        log_info "Updating Homebrew..."
        brew update --quiet 2>/dev/null || log_warn "Homebrew update failed (continuing anyway)"
        return 0
    fi

    log_info "Installing Homebrew..."
    echo ""
    echo "Homebrew is the missing package manager for macOS."
    echo "It will be installed to /opt/homebrew (Apple Silicon) or /usr/local (Intel)."
    echo ""
    echo "Press Ctrl+C to cancel, or any key to continue..."
    read -rs -n 1

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add Homebrew to PATH for Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"

        # Add to shell profile if not already there
        local shell_profile=""
        if [[ -f "$HOME/.zprofile" ]]; then
            shell_profile="$HOME/.zprofile"
        elif [[ -f "$HOME/.zshrc" ]]; then
            shell_profile="$HOME/.zshrc"
        fi

        if [[ -n "$shell_profile" ]]; then
            if ! grep -q "brew shellenv" "$shell_profile" 2>/dev/null; then
                echo "" >> "$shell_profile"
                echo '# Homebrew' >> "$shell_profile"
                echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$shell_profile"
                log_info "Added Homebrew to $shell_profile"
            fi
        fi
    fi

    if check_command brew; then
        log_success "Homebrew installed successfully"
    else
        log_error "Homebrew installation failed"
        exit 1
    fi
}

# ── Install FFmpeg ────────────────────────────────────────────────────────────
install_ffmpeg() {
    if [[ "$SKIP_FFMPEG" == "true" ]]; then
        log_info "Skipping FFmpeg installation (--skip-ffmpeg)"
        return 0
    fi

    log_step "Checking FFmpeg..."

    if check_command ffmpeg; then
        local ffmpeg_version
        ffmpeg_version=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')
        log_success "FFmpeg already installed: $ffmpeg_version"
        return 0
    fi

    if ! check_command brew; then
        log_error "Homebrew is required to install FFmpeg"
        log_info "Run script without --skip-brew, or install Homebrew manually"
        return 1
    fi

    log_info "Installing FFmpeg via Homebrew..."
    brew install ffmpeg

    if check_command ffmpeg; then
        log_success "FFmpeg installed successfully"
    else
        log_error "FFmpeg installation failed"
        return 1
    fi
}

# ── Install Ollama ────────────────────────────────────────────────────────────
install_ollama() {
    if [[ "$SKIP_OLLAMA" == "true" ]]; then
        log_info "Skipping Ollama installation (--skip-ollama)"
        return 0
    fi

    log_step "Checking Ollama..."

    # Check if Ollama is installed
    if check_command ollama; then
        local ollama_version
        ollama_version=$(ollama --version 2>/dev/null | awk '{print $2}' || echo "unknown")
        log_success "Ollama already installed: $ollama_version"
    else
        log_info "Installing Ollama..."
        echo ""
        echo "Ollama is a local LLM runtime for running AI models."
        echo "Download size: ~100MB"
        echo ""

        curl -fsSL https://ollama.com/install.sh | sh

        if check_command ollama; then
            log_success "Ollama installed successfully"
        else
            log_error "Ollama installation failed"
            return 1
        fi
    fi

    # Ensure Ollama service is running
    log_info "Checking Ollama service..."
    if curl -s "$OLLAMA_BASE_URL/api/version" &>/dev/null; then
        log_success "Ollama service is running"
    else
        log_info "Starting Ollama service..."

        # Try to start Ollama
        if [[ -d "/Applications/Ollama.app" ]]; then
            open -a Ollama
        elif check_command ollama; then
            ollama serve &>/dev/null &
        fi

        # Wait for service to start
        local retries=0
        while [[ $retries -lt 30 ]]; do
            if curl -s "$OLLAMA_BASE_URL/api/version" &>/dev/null; then
                log_success "Ollama service started"
                break
            fi
            sleep 1
            retries=$((retries + 1))
        done

        if [[ $retries -ge 30 ]]; then
            log_warn "Ollama service may not have started properly"
            log_info "Try running 'ollama serve' manually"
        fi
    fi
}

# ── Download model ────────────────────────────────────────────────────────────
download_model() {
    if [[ "$SKIP_MODEL" == "true" ]]; then
        log_info "Skipping model download (--skip-model)"
        return 0
    fi

    log_step "Checking model: $MODEL_NAME"

    # Ensure Ollama is running
    if ! curl -s "$OLLAMA_BASE_URL/api/version" &>/dev/null; then
        log_warn "Ollama service is not running"
        log_info "Please start Ollama first, then run: ollama pull $MODEL_NAME"
        return 1
    fi

    # Check if model already exists
    if ollama list 2>/dev/null | grep -q "^${MODEL_NAME%%:*}"; then
        log_success "Model already downloaded: $MODEL_NAME"
        return 0
    fi

    log_info "Downloading model: $MODEL_NAME"
    echo ""
    echo "This model is required for video analysis."
    echo "Download size: ~4.7GB (MiniCPM-V)"
    echo "This may take several minutes depending on your connection."
    echo ""
    read -p "Continue? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_warn "Model download skipped"
        log_info "You can download later with: ollama pull $MODEL_NAME"
        return 0
    fi

    ollama pull "$MODEL_NAME"

    if ollama list 2>/dev/null | grep -q "^${MODEL_NAME%%:*}"; then
        log_success "Model downloaded successfully: $MODEL_NAME"
    else
        log_error "Model download failed"
        log_info "Try manually: ollama pull $MODEL_NAME"
        return 1
    fi
}

# ── Install Python dependencies ──────────────────────────────────────────────
install_python_deps() {
    log_step "Checking Python dependencies..."

    if ! check_command uv; then
        log_warn "'uv' package manager not found"
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add to PATH
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if check_command uv; then
        log_success "uv is available"

        # Check if we're in the project directory
        if [[ -f "$PROJECT_DIR/pyproject.toml" ]]; then
            log_info "Installing Python dependencies..."
            cd "$PROJECT_DIR"
            uv sync
            log_success "Python dependencies installed"
        else
            log_warn "Not in project directory, skipping Python dependency installation"
            log_info "Run 'uv sync' in the project directory manually"
        fi
    else
        log_warn "uv installation failed"
        log_info "Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
}

# ── Install Face Recognition ──────────────────────────────────────────────────
install_face_recognition() {
    if [[ "$SKIP_FACE" == "true" ]]; then
        log_info "Skipping face recognition installation (--skip-face)"
        return 0
    fi

    log_step "Installing face recognition dependencies..."

    if ! check_command uv; then
        log_error "uv package manager not found"
        log_info "Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi

    # Check if we're in the project directory
    if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
        log_warn "Not in project directory, skipping face recognition installation"
        log_info "Run 'uv pip install insightface onnxruntime' in the project directory manually"
        return 1
    fi

    cd "$PROJECT_DIR"

    # Check if already installed
    if .venv/bin/python -c "import insightface" 2>/dev/null; then
        log_success "Face recognition dependencies already installed"
        return 0
    fi

    log_info "Installing insightface and onnxruntime..."
    echo ""
    echo "These packages enable automatic actor face recognition and clustering."
    echo "Download size: ~200MB"
    echo ""

    if uv pip install insightface onnxruntime; then
        log_success "Face recognition dependencies installed successfully"
        
        # Verify installation
        if .venv/bin/python -c "import insightface; print(f'InsightFace version: {insightface.__version__}')" 2>/dev/null; then
            log_success "Face recognition module verified"
        else
            log_warn "Installation completed but verification failed"
        fi
    else
        log_error "Face recognition dependencies installation failed"
        log_info "Try manually: uv pip install insightface onnxruntime"
        return 1
    fi
}

# ── Verification ──────────────────────────────────────────────────────────────
verify_installation() {
    log_step "Verifying installation..."

    local all_ok=true

    # Check FFmpeg
    if check_command ffmpeg; then
        log_success "FFmpeg: $(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')"
    else
        log_error "FFmpeg: NOT INSTALLED"
        all_ok=false
    fi

    # Check Ollama
    if check_command ollama; then
        log_success "Ollama: $(ollama --version 2>/dev/null | awk '{print $2}' || echo 'installed')"
    else
        log_error "Ollama: NOT INSTALLED"
        all_ok=false
    fi

    # Check Ollama service
    if curl -s "$OLLAMA_BASE_URL/api/version" &>/dev/null; then
        log_success "Ollama service: running"
    else
        log_warn "Ollama service: not running (start with 'ollama serve' or open Ollama app)"
    fi

    # Check model
    if ollama list 2>/dev/null | grep -q "^${MODEL_NAME%%:*}"; then
        log_success "Model: $MODEL_NAME"
    else
        log_warn "Model: not downloaded (run 'ollama pull $MODEL_NAME')"
    fi

    # Check Python
    if check_command python3; then
        log_success "Python: $(python3 --version 2>&1 | awk '{print $2}')"
    else
        log_error "Python: NOT INSTALLED"
        all_ok=false
    fi

    # Check uv
    if check_command uv; then
        log_success "uv: $(uv --version 2>&1 | awk '{print $2}')"
    else
        log_warn "uv: not installed (optional, for Python dependencies)"
    fi

    # Check face recognition
    if [[ -f "$PROJECT_DIR/.venv/bin/python" ]]; then
        if "$PROJECT_DIR/.venv/bin/python" -c "import insightface" 2>/dev/null; then
            local face_version
            face_version=$("$PROJECT_DIR/.venv/bin/python" -c "import insightface; print(insightface.__version__)" 2>/dev/null || echo "installed")
            log_success "Face recognition: $face_version"
        else
            log_warn "Face recognition: not installed (optional, run with --skip-face to skip)"
        fi
    fi

    echo ""
    if [[ "$all_ok" == "true" ]]; then
        log_success "All required dependencies are installed!"
        echo ""
        echo "Next steps:"
        echo "  1. Start the API server: ./start-api.sh"
        echo "  2. Start the desktop UI: ./start-ui.sh"
        echo ""
    else
        log_warn "Some dependencies are missing. See errors above."
        echo ""
        echo "Manual installation guides:"
        echo "  - FFmpeg: brew install ffmpeg"
        echo "  - Ollama: curl -fsSL https://ollama.com/install.sh | sh"
        echo "  - Model: ollama pull $MODEL_NAME"
        echo ""
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "=============================================="
    echo "  AI Video Factory - Dependency Installer"
    echo "=============================================="
    echo ""

    preflight_check
    install_homebrew
    install_ffmpeg
    install_ollama
    download_model
    install_python_deps
    install_face_recognition
    verify_installation

    echo "Installation complete!"
}

main "$@"
