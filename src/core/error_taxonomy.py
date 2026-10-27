"""错误分类目录（add-public-sale-readiness 任务 2.2）。

规格要求（specs/error-taxonomy）：
- 用户可读错误 = 分类编码 + 一句话 + 可行动建议;堆栈/驱动名/进程级异常
  文本只进日志,不进用户可见文案;
- 分类目录：文件损坏/解码失败/模型异常/资源不足/网络与授权/未知,每类有
  触发规则、文案模板、建议(即上报聚合键);
- 自愈优先：DB 锁、单帧失败等瞬时错误由既有重试/串行化机制自愈,本模块
  只对"最终失败"做归类展示。

用法：
- 新错误：``format_error(raw)`` → "[CODE] 文案（建议：...）" 直写
  task.error_message;原始堆栈/细节由调用方 logger 记录;
- 历史重映射：``scripts/remap_error_messages.py``(带备份,dry-run 默认)。
"""

import re
from typing import Dict

# 分类目录：code → 文案要素。advice 即上报聚合键的展示形态。
CATEGORIES: Dict[str, Dict[str, str]] = {
    "FILE_CORRUPT": {"label": "文件损坏",
                     "advice": "重新获取或修复该文件后重新分析"},
    "DECODE_FAIL": {"label": "解码失败",
                    "advice": "重新分析一次;反复出现说明文件数据异常"},
    "MODEL": {"label": "模型异常",
              "advice": "重试一次;反复出现请通过设置页反馈日志"},
    "RESOURCE": {"label": "资源或环境异常",
                 "advice": "关闭内存清理类工具、确认磁盘与内存充足后重试"},
    "NETWORK_AUTH": {"label": "网络或授权异常",
                     "advice": "检查网络连接与授权状态后重试"},
    "UNKNOWN": {"label": "未知错误",
                "advice": "重试一次;反复出现请通过设置页反馈日志"},
}

# 规则表(顺序即优先级,首中即停): (raw 正则, category, 用户一句话模板)
_RULES = [
    (r"文件损坏-|INF_MISSING_ASSET|INF_ASSET_CORRUPT|moov atom|partial file|"
     r"Invalid NAL|Error splitting",
     "FILE_CORRUPT", "文件数据损坏或不完整,无法完整解码"),
    (r"未找到帧文件|0帧可解", "DECODE_FAIL", "未能从视频解出有效画面"),
    (r"返回空结果或结果过短|refus", "MODEL", "模型未能给出有效结果"),
    (r"terminated abruptly|database is locked|OperationalError|"
     r"No such file or directory: 'ffmpeg'|Read-only file system|"
     r"NFO文件生成失败|内存不足|Cannot allocate",
     "RESOURCE", "处理资源或运行环境异常"),
    (r"Connection|Max retries|授权|license|unauthorized", "NETWORK_AUTH",
     "网络或授权异常"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), cat, tpl) for p, cat, tpl in _RULES]


def classify(raw: str) -> Dict[str, str]:
    """原始错误文本 → {code, label, advice}。首中即停;无命中归 UNKNOWN。"""
    raw = raw or ""
    for pat, cat, _tpl in _COMPILED:
        if pat.search(raw):
            return {"code": cat, **CATEGORIES[cat]}
    return {"code": "UNKNOWN", **CATEGORIES["UNKNOWN"]}


def format_error(raw: str) -> str:
    """原始错误文本 → 用户可读文案(写 task.error_message 的标准形态)。

    用户文案取自目录模板,**不含**原始异常细节——细节由调用方记录日志。
    """
    c = classify(raw)
    return f"[{c['code']}] {c['label']}（建议：{c['advice']}）"


def user_view(raw: str) -> str:
    """历史遗留消息的展示层重映射:可归类则返回标准文案,否则原样返回。"""
    if not raw:
        return raw
    c = classify(raw)
    if c["code"] == "UNKNOWN":
        return raw
    return f"[{c['code']}] {c['label']}（建议：{c['advice']}）"
