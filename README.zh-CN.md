# Media_Tag

[English](README.md) | 简体中文

本地优先的 macOS AI 素材标签系统——内置视觉模型、人脸聚类、五道质量闸、NFO/XMP 导出与项目交付工作流。全程本机运算,无需云端。Source-available。

**本地优先的 macOS AI 素材标签系统。** 在你自己的电脑上完成视频/照片库的
AI 分析——全程不联网上传,产出可检索、可交付的元数据。

## 功能

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
mkdir -p models/qwen3-vl
# 先下载模型权重(不入库),例:
#   https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF
huggingface-cli download unsloth/Qwen3-VL-8B-Instruct-GGUF \
  Qwen3-VL-8B-Instruct-Q4_K_M.gguf --local-dir models/qwen3-vl
bash scripts/build.sh
```

## 仓库结构

```
src/core/       分析管线 / VLM 客户端 / 内嵌推理 / 授权
src/api/        FastAPI 本地服务
src/ui/         PySide6 桌面应用
src/db/         原生 SQLite(热路径无 ORM)
scripts/        构建 / 打包 / 开发启动脚本
tests/          pytest 测试
```
