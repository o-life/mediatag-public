# Media_Tag

English | [简体中文](README.zh-CN.md)

Local-first AI media tagging for macOS — bundled vision model, face clustering, five quality gates, NFO/XMP export and project delivery workflows. Fully on-device, no cloud. Source-available.

**Local-first AI media tagging for macOS.** Analyze video and photo libraries
entirely on your own machine — no cloud, no upload — and turn them into
searchable, export-ready metadata.

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
