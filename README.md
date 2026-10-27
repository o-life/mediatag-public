# Media_Tag

Local-first AI media tagging for macOS — bundled vision model, face clustering, five quality gates, NFO/XMP export and project delivery workflows. Fully on-device, no cloud. Source-available.

---

**Local-first AI media tagging for macOS.** Analyze video and photo libraries
entirely on your own machine — no cloud, no upload — and turn them into
searchable, export-ready metadata.

简体中文说明见下方 / 中文说明见下。

---

## What it does

- **AI frame-level tagging** — a bundled local vision-language model
  (Qwen3-VL-8B, GGUF Q4 + llama.cpp, no external runtime required) tags every
  scene with a closed-vocabulary dictionary; every tag keeps its evidence frame.
- **Face pipeline** — detection → embedding → clustering with a curated,
  commercially-licensed model set (AuraFace, Apache-2.0).
- **Quality gates** — closed vocabulary enforcement, dictionary alias mapping,
  junk-frame filtering, tag frequency thresholds, mutual-exclusion rules: five
  gates between raw model output and your library.
- **Export** — NFO sidecars (Jellyfin/Infuse/Kodi compatible) and per-project
  batch export with XMP sidecars readable by Lightroom Classic / Capture One.
- **Project & delivery** — projects, rosters anchored by file content
  hash, automatic grouping, batch delivery packaging.
- **Performance tiers** — auto-selects concurrency/context/frame-density for
  the machine it runs on (16 GB Apple Silicon minimum, verified 4.7 min for a
  30-min video on the 16 GB tier).

Everything runs locally: media files never leave the machine.

## Status

`source-available` under the [Fair Source License v1.1](LICENSE) — you can
read, audit, and run it for personal/internal use; commercial use (bundling,
reselling, offering as a service) requires a commercial license. Official
signed builds with activation licensing are distributed via the project
website. Each version converts to MIT two years after publication.

## Requirements

- macOS 14+ (Apple Silicon; M1 16 GB minimum, M-series 32 GB+ recommended)
- ~6.5 GB disk for the bundled model + runtime

## Build

```bash
./scripts/install_dependencies.sh   # uv sync + system deps
bash scripts/build.sh               # PyInstaller + bundled llama.cpp + DMG
```

The build bundles the Qwen3-VL GGUF weights and a self-contained
`llama-server` runtime (`scripts/prepare_llama_bundle.sh` resolves the dylib
closure automatically).

## Repository layout

```
src/core/       analysis pipeline, VLM clients, embedded inference, licensing
src/api/        FastAPI local server
src/ui/         PySide6 desktop app
src/db/         raw SQLite (no ORM on the hot path)
scripts/        build / packaging / dev-run scripts
tests/          pytest suite
```

---

# MediaTag(中文)

**本地优先的 macOS AI 素材标签系统。** 在你自己的电脑上完成视频/照片库的
AI 分析——全程不联网上传,产出可检索、可交付的元数据。

- **逐帧 AI 打标**:内置本地视觉语言模型(Qwen3-VL-8B + llama.cpp,无需安装
  任何外部运行时),按闭式词表给每个场景打标,**每个标签都保留证据帧**;
- **人脸管线**:检测→向量→聚类,内置商业授权模型(AuraFace,Apache-2.0);
- **五道质量闸**:闭式词表、词典别名、垃圾帧过滤、频率门槛、互斥规则;
- **交付**:NFO sidecar(兼容 Jellyfin/Infuse/Kodi)+ 按项目批量导出
  (XMP 可被 Lightroom Classic / Capture One 直读);
- **项目与交付**:项目/清单(内容哈希锚)/自动归集/交付打包;
- **性能档位**:按机器自动选择并发/上下文/抽帧密度(16GB Apple Silicon 起步,
  16GB 档 30 分钟视频全流程实测 4.7 分钟)。

全部计算在本机完成,素材不出门。

## 许可

[Fair Source License v1.1](LICENSE):源码公开可读、可自用;**商业使用需
商业授权**——购买官方安装包+激活码即是最便捷的授权方式;每版发布满两年
自动转 MIT。

## 构建

```bash
./scripts/install_dependencies.sh
bash scripts/build.sh
```
