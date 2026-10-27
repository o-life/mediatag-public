# Third-Party Notices

MediaTag incorporates and depends on the following third-party assets.
This file summarizes licenses for convenience — the license text of each
project governs. Full texts live in `third_party_licenses/`.

| Component | License | Usage |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Bundled `llama-server` inference runtime (`third_party_licenses/llamacpp`) |
| [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) weights | Qwen / Apache-2.0 (see `third_party_licenses/qwen3vl`) | Bundled tagging model (GGUF Q4_K_M + mmproj Q8_0) |
| [PySide6 / Qt](https://www.qt.io/) | LGPL-3.0 | Desktop UI framework (dynamically linked) |
| [OpenCV](https://opencv.org/) | Apache-2.0 | Video/frame processing |
| [PyTorch](https://pytorch.org/) | BSD-3 | ML runtime |
| [onnxruntime](https://onnxruntime.ai/) | MIT | Face pipeline inference |
| AuraFace model set | Apache-2.0 | Face detection/recognition models (InsightFace-compatible, commercially licensed replacement) |
| [FFmpeg](https://ffmpeg.org/) | LGPL-2.1+ (self-built build, spawned as a separate process) | Video decode/probe |
| [FastAPI](https://fastapi.tiangolo.com/) / [uvicorn](https://www.uvicorn.org/) | MIT / BSD-3 | Local API server |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3 | Performance tier detection |

Bundled model weights (`*.gguf`, face ONNX files) are distributed under their
upstream licenses; see `third_party_licenses/` for the complete texts.

If you believe any attribution is missing or incorrect, please open an issue.
