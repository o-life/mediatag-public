"""错误分类目录单测——样本取自生产 ERROR 分布的真实原文。"""

import pytest

from src.core.error_taxonomy import classify, format_error, user_view

SAMPLES = [
    ("处理异常: A process in the process pool was terminated abruptly while the future was",
     "RESOURCE"),
    ("处理异常: (sqlite3.OperationalError) database is locked\nSQL: INSERT INTO tag_frame_",
     "RESOURCE"),
    ("分析失败: 返回空结果或结果过短 (长度: 0, NSFW帧数: 8/8)", "MODEL"),
    ("处理异常: [Errno 2] No such file or directory: 'ffmpeg'", "RESOURCE"),
    ("未找到帧文件", "DECODE_FAIL"),
    ("文件损坏-部分损坏(体检20261006)", "FILE_CORRUPT"),
    ("处理异常: [Errno 30] Read-only file system: 'data'", "RESOURCE"),
    ("INF_ASSET_CORRUPT 模型资源校验失败", "FILE_CORRUPT"),
    ("完全没见过的错误", "UNKNOWN"),
]


@pytest.mark.parametrize("raw,want", SAMPLES)
def test_classify_real_samples(raw, want):
    assert classify(raw)["code"] == want


def test_format_error_shape():
    out = format_error(SAMPLES[0][0])
    assert out.startswith("[RESOURCE]")
    assert "建议：" in out


def test_user_copy_never_leaks_raw_detail():
    """规格红线:堆栈/驱动名/进程级异常文本不得进用户文案。"""
    leaky = ("处理异常: A process in the process pool was terminated abruptly "
             "while the future was running or completed.\nTraceback: sqlite3.OperationalError")
    out = format_error(leaky)
    assert "process pool" not in out
    assert "sqlite3" not in out
    assert "Traceback" not in out


def test_user_view_passthrough_unknown():
    raw = "完全没见过的错误"
    assert user_view(raw) == raw


def test_user_view_remaps_known():
    out = user_view("未找到帧文件")
    assert out.startswith("[DECODE_FAIL]") and "建议" in out
