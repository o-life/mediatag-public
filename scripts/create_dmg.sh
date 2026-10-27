#!/bin/bash

# MediaTag DMG 制作脚本
#
# 用法:
#   scripts/create_dmg.sh [distpath] [appname] [dmgname]
#     默认: dist "MediaTag"（DMG 名 = MediaTag.dmg）
#     轻量版示例: scripts/create_dmg.sh dist_lite "MediaTag" "MediaTag-Lite"
#
# 说明: appname 用于定位 {distpath}/{appname}.app; dmgname 用于命名
#   输出 {distpath}/{dmgname}.dmg(不含 .dmg 后缀)。两版 .app 同名
#   "MediaTag", DMG 用 dmgname 区分。

set -e

DISTPATH="${1:-dist}"
APPNAME="${2:-MediaTag}"
DMGNAME="${3:-$APPNAME}"
APP_PATH="$DISTPATH/$APPNAME.app"

if [ ! -d "$APP_PATH" ]; then
    echo "错误: $APP_PATH 不存在"
    echo "请先运行 build.sh 打包应用"
    exit 1
fi

DMG_PATH="$DISTPATH/$DMGNAME.dmg"
echo "清理旧的 DMG..."
rm -f "$DMG_PATH"

TEMP_DIR="$DISTPATH/.dmg_temp"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo "复制应用到临时目录..."
cp -R "$APP_PATH" "$TEMP_DIR/"

echo "创建 Applications 软链接..."
ln -s /Applications "$TEMP_DIR/Applications"

echo "创建 DMG (UDZO)..."
hdiutil create \
    -volname "$APPNAME" \
    -srcfolder "$TEMP_DIR" \
    -ov \
    -format UDZO \
    "$DMG_PATH"

rm -rf "$TEMP_DIR"
SIZE=$(du -sh "$DMG_PATH" | cut -f1)
echo "✔ DMG 创建完成: $DMG_PATH ($SIZE)"
