#!/bin/bash
# 准备 bundled_bin/llama_cpp——自包含的 llama-server 运行时（MIT 许可）。
#
# 做法：解析 brew llama-server 的 otool -L 依赖闭包（跳过系统库），全部拷入
# lib/ 并把加载命令重写为 @loader_path(/../lib)/<name>；id 同步重写；最后
# ad-hoc 重签（Apple Silicon 无签名不运行）。
# 用法: scripts/prepare_llama_bundle.sh [llama-server路径] [输出目录]
set -euo pipefail

SRC="${1:-/opt/homebrew/bin/llama-server}"
OUT="${2:-bundled_bin/llama_cpp}"
SRC="$(readlink -f "$SRC")"

rm -rf "$OUT"
mkdir -p "$OUT/bin" "$OUT/lib"
cp "$SRC" "$OUT/bin/llama-server"

# 主二进制的 LC_RPATH 列表;@loader_path 相对形式按二进制自身目录实体化
LOADER_DIR="$(dirname "$SRC")"
RPATHS=("$LOADER_DIR")
while IFS= read -r _rp; do
    [ -z "$_rp" ] && continue
    case "$_rp" in
        @loader_path*) RPATHS+=("$LOADER_DIR/${_rp#@loader_path/}") ;;
        *) RPATHS+=("$_rp") ;;
    esac
done < <(otool -l "$SRC" | awk '/LC_RPATH/{f=1} f&&/ path /{print $2; f=0}')

resolve() {  # $1=依赖声明 $2=引用它的 dylib 目录 → 输出绝对路径或空
    local dep="$1" refdir="$2" cand
    case "$dep" in
        /*) [ -f "$dep" ] && { echo "$dep"; return; } ;;
    esac
    cand="$refdir/$(basename "$dep")"
    [ -f "$cand" ] && { echo "$cand"; return; }
    local rp
    for rp in "${RPATHS[@]}"; do
        cand="${rp}/${dep#@rpath/}"
        [ -f "$cand" ] && { echo "$cand"; return; }
        cand="/opt/homebrew/${dep#@rpath/}"
        [ -f "$cand" ] && { echo "$cand"; return; }
    done
    return 1
}

collect() {  # $1=dylib 绝对路径 → 拷入 lib/ 并递归其依赖
    local d="$1" base dep resolved refdir
    base="$(basename "$d")"
    [ -f "$OUT/lib/$base" ] && return 0
    cp -L "$d" "$OUT/lib/$base"
    refdir="$(dirname "$(readlink -f "$d")")"
    while IFS= read -r dep; do
        case "$dep" in
            /usr/lib/*|/System/*|@loader_path/*) continue ;;
        esac
        resolved="$(resolve "$dep" "$refdir")" || { echo "无法解析依赖 $dep (来自 $d)"; exit 1; }
        collect "$resolved"
    done < <(otool -L "$d" | tail -n +2 | awk '{print $1}')
}

collect "$SRC"

# 主二进制:依赖重写到 @loader_path/../lib/
while IFS= read -r dep; do
    case "$dep" in
        /usr/lib/*|/System/*|@loader_path/*) continue ;;
    esac
    install_name_tool -change "$dep" "@loader_path/../lib/$(basename "$dep")" "$OUT/bin/llama-server"
done < <(otool -L "$OUT/bin/llama-server" | tail -n +2 | awk '{print $1}')
install_name_tool -rpath "$(dirname "$(readlink -f "$SRC")")" "@loader_path/../lib" "$OUT/bin/llama-server" 2>/dev/null || true

# 每个拷入的 dylib:id 重写 + 自身依赖重写(闭包内互相指向 @loader_path)
for dy in "$OUT"/lib/*.dylib; do
    base="$(basename "$dy")"
    install_name_tool -id "@loader_path/$base" "$dy"
    while IFS= read -r dep; do
        case "$dep" in
            /usr/lib/*|/System/*|@loader_path/*) continue ;;
        esac
        [ -f "$OUT/lib/$(basename "$dep")" ] && \
            install_name_tool -change "$dep" "@loader_path/$(basename "$dep")" "$dy"
    done < <(otool -L "$dy" | tail -n +2 | awk '{print $1}')
    codesign -f -s - "$dy" 2>/dev/null
done

codesign -f -s - "$OUT/bin/llama-server"

# 真实二进制改名 .real,原路径放 DYLD shim:强制优先本 bundle 的 lib/,
# 否则 PyInstaller 拍平进 Frameworks 的 sklearn 旧版 libomp/libggml 会
# 被dyld 先解析(2026-10-07 打包实测 Symbol not found ___kmpc_dispatch_deinit)
mv "$OUT/bin/llama-server" "$OUT/bin/llama-server.real"
cat > "$OUT/bin/llama-server" <<'SHIM'
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export DYLD_LIBRARY_PATH="$DIR/../lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
exec "$DIR/llama-server.real" "$@"
SHIM
chmod +x "$OUT/bin/llama-server" "$OUT/bin/llama-server.real"
codesign -f -s - "$OUT/bin/llama-server.real"

echo "就绪: $OUT"
echo "自包含验证(shim):"
"$OUT/bin/llama-server" --version
