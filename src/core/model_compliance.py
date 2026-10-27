"""VLM 模型商业授权清单。

版权合规策略：
- 软件只对**可商业化的模型**（Apache 2.0 / MIT 等宽松许可）提供下载入口。
- **禁止商业化/未收录的模型**：不提示、不指引下载，但允许用户手动接入
  （用户自行负责下载与合规），检测到使用时提示"需在 24 小时内删除"。

下载动作本身由用户主动触发（软件只提供合规模型的下载引导），
任何模型都不捆绑进软件发行包。
"""

from typing import Dict, Optional

# 可商用模型清单：模型名（Ollama tag 前缀匹配）→ {名称, 许可, 说明}
COMMERCIALLY_ALLOWED_MODELS: Dict[str, Dict[str, str]] = {
    "qwen2.5-vl": {
        "name": "Qwen2.5-VL-7B",
        "license": "Apache 2.0",
        "note": "通义千问视觉语言模型，可商用",
    },
    "minicpm-v": {
        "name": "MiniCPM-V",
        "license": "Apache 2.0",
        "note": "面壁智能多模态模型，可商用",
    },
    "minicpm-v4": {
        "name": "MiniCPM-V 4.x",
        "license": "Apache 2.0",
        "note": "面壁智能多模态模型，可商用",
    },
    "llava": {
        "name": "LLaVA",
        "license": "Apache 2.0",
        "note": "开源视觉指令模型，可商用",
    },
    "gemma3": {
        "name": "Gemma 3",
        "license": "Gemma Terms (可商用，需遵守附加条款)",
        "note": "Google 开放模型，可商用",
    },
    "qwen3": {
        "name": "Qwen3 系列",
        "license": "Apache 2.0",
        "note": "通义千问文本模型，可商用",
    },
}

# 明确禁止商用的模型（仅用于识别和警示，不提供下载指引）
NON_COMMERCIAL_MODELS: Dict[str, Dict[str, str]] = {
    "gpt-4": {"name": "GPT-4", "license": "OpenAI 商业许可", "note": "不可商用"},
    "gpt-4o": {"name": "GPT-4o", "license": "OpenAI 商业许可", "note": "不可商用"},
    "gpt-4v": {"name": "GPT-4V", "license": "OpenAI 商业许可", "note": "不可商用"},
    "claude": {"name": "Claude", "license": "Anthropic 商业许可", "note": "不可商用"},
    "gemini": {"name": "Gemini", "license": "Google 商业许可", "note": "不可商用"},
    "llama": {"name": "Llama", "license": "Llama Community License (需申请)", "note": "商用需 Meta 批准"},
    "mistral": {"name": "Mistral", "license": "Mistral 许可", "note": "商用需确认"},
}


def check_model_commercial_status(model_name: str) -> Dict[str, Optional[str]]:
    """判断模型的可商用状态。

    Args:
        model_name: Ollama 模型名，如 "minicpm-v4.6:latest"

    Returns:
        {
            "status": "commercial" | "non_commercial" | "unknown",
            "license": 许可信息（如已知）,
            "note": 说明,
        }
    """
    if not model_name:
        return {"status": "unknown", "license": None, "note": None}

    name_lower = model_name.lower()

    # 先查可商用清单（前缀匹配）
    for prefix, info in COMMERCIALLY_ALLOWED_MODELS.items():
        if name_lower.startswith(prefix):
            return {
                "status": "commercial",
                "license": info["license"],
                "note": info["note"],
            }

    # 再查禁商用清单
    for prefix, info in NON_COMMERCIAL_MODELS.items():
        if name_lower.startswith(prefix):
            return {
                "status": "non_commercial",
                "license": info["license"],
                "note": info["note"],
            }

    return {"status": "unknown", "license": None, "note": None}


def is_commercial(model_name: str) -> bool:
    """模型是否可商用（在可商用清单内）。"""
    return check_model_commercial_status(model_name)["status"] == "commercial"
