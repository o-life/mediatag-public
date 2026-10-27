"""标签质量门槛(批3)单元测试 —— 全部纯逻辑/内存库,不碰网络。

覆盖五道闸的核心函数:
- 闸2 enforce_vocabulary(解析边界词典强制)
- 闸3 frame_quality.frame_issue / filter_frames(黑帧/糊帧/废帧)
- 闸4 apply_frequency_gate(证据频率门槛 + 数量上限)
- 闸5 filter_contradictions(帧级/文件级互斥) + map_raw_text_to_tags 收紧
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.dictionary import (
    apply_frequency_gate,
    enforce_vocabulary,
    filter_contradictions,
    sanitize_alias,
    CONTRADICTION_PAIRS_FRAME,
    DictionaryManager,
)
from src.core.frame_quality import frame_issue, filter_frames
from src.db.models import DictionaryEntry


STD_TAGS = {"青年", "浴室", "健身房", "双人", "后入", "写实", "二次元", "白天", "夜晚"}


# ── 闸 2: 词典强制 ────────────────────────────────────────────────

def test_enforce_vocabulary_keeps_std_drops_offvocab():
    # 保留词典内标签、剔除词典外、strip 后去重(标签是集合语义)
    kept, dropped = enforce_vocabulary(
        ["青年", "亲吻", "双人", "丝袜", "", " 青年 "], STD_TAGS)
    assert kept == ["青年", "双人"]
    assert dropped == ["亲吻", "丝袜"]


def test_enforce_vocabulary_empty_inputs():
    assert enforce_vocabulary([], STD_TAGS) == ([], [])
    assert enforce_vocabulary([None, None], STD_TAGS) == ([], [])


# ── 闸 3: 垃圾帧 ──────────────────────────────────────────────────

def _write_image(path, arr):
    import cv2
    assert cv2.imwrite(str(path), arr)


def test_frame_issue_black_frame(temp_dir):
    p = temp_dir / "black.jpg"
    _write_image(p, np.zeros((180, 320, 3), np.uint8))
    assert frame_issue(p) == "black"


def test_frame_issue_sharp_frame_ok(temp_dir):
    p = temp_dir / "sharp.jpg"
    rng = np.random.default_rng(20260930)
    _write_image(p, rng.integers(0, 255, (180, 320, 3), dtype=np.uint8))
    assert frame_issue(p) is None


def test_frame_issue_blurry_frame(temp_dir):
    import cv2
    p = temp_dir / "blur.jpg"
    rng = np.random.default_rng(20260930)
    img = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    # 递归均值模糊多轮,直到方差低于阈值(确定性构造糊帧)
    while True:
        img = cv2.GaussianBlur(img, (31, 31), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 12:
            break
    _write_image(p, img)
    assert frame_issue(p) == "blurry"


def test_frame_issue_unreadable(temp_dir):
    assert frame_issue(temp_dir / "nope.jpg") == "unreadable"


def test_filter_frames_splits_good_and_dropped(temp_dir):
    good = temp_dir / "good.jpg"
    black = temp_dir / "b.jpg"
    _write_image(good, np.random.default_rng(1).integers(0, 255, (90, 160, 3), dtype=np.uint8))
    _write_image(black, np.zeros((90, 160, 3), np.uint8))
    ok, dropped = filter_frames([str(good), str(black), str(temp_dir / "gone.jpg")])
    assert ok == [str(good)]
    assert {why for _, why in dropped} == {"black", "unreadable"}


# ── 闸 4: 频率门槛 ────────────────────────────────────────────────

def test_frequency_gate_short_file_keeps_all():
    counts = {"青年": 1, "浴室": 1}
    assert apply_frequency_gate(counts, n_units=1) == {"青年", "浴室"}
    assert apply_frequency_gate(counts, n_units=4) == {"青年", "浴室"}  # ceil(4×.25)=1


def test_frequency_gate_multi_scene_requires_support():
    counts = {"双人": 3, "后入": 2, "写实": 1}
    kept = apply_frequency_gate(counts, n_units=10)  # 须≥min(2, ceil(2.5)=3)=2
    assert kept == {"双人", "后入"}


def test_frequency_gate_caps_by_support():
    counts = {f"tag{i:02d}": i + 1 for i in range(40)}  # 支持 1..40
    kept = apply_frequency_gate(counts, n_units=100, max_tags=30)
    assert len(kept) == 30
    # 门槛=2 先滤掉 tag00;再按支持数降序截断,保 40..11(tag39..tag10)
    assert "tag39" in kept and "tag10" in kept and "tag09" not in kept


def test_frequency_gate_empty():
    assert apply_frequency_gate({}, 5) == set()


# ── 闸 5: 互斥过滤 ────────────────────────────────────────────────

def test_contradiction_frame_level_drops_minority():
    counts = {"白天": 5, "夜晚": 1}
    out = filter_contradictions(["白天", "夜晚", "双人"], counts)
    assert out == ["白天", "双人"]


def test_contradiction_tie_drops_later():
    out = filter_contradictions(["写实", "二次元"])
    assert out == ["写实"]


def test_contradiction_file_level_keeps_scene_pairs():
    # 场景在多场景视频中可先后出现:文件级不删,帧级删
    assert filter_contradictions(["卧室", "健身房"], file_level=True) == ["卧室", "健身房"]
    assert filter_contradictions(["卧室", "健身房"]) == ["卧室"]
    # 跨维度硬冲突文件级仍删
    assert filter_contradictions(["写实", "二次元"], file_level=True) == ["写实"]


def test_contradiction_pairs_curation_guard():
    # 审计策划中剔除的伪互斥不得回流
    assert ("白天", "雪天") not in CONTRADICTION_PAIRS_FRAME
    assert ("晴天", "雪天") not in CONTRADICTION_PAIRS_FRAME
    # 已实证的互斥必须在
    assert ("写实", "二次元") in CONTRADICTION_PAIRS_FRAME
    assert ("单人", "双人") in CONTRADICTION_PAIRS_FRAME


# ── 闸 5: 词典匹配收紧(内存库) ───────────────────────────────────

@pytest.fixture
def dict_mgr(temp_dir):
    # 不用 conftest 的 db_engine(:memory: 与当前 SQLAlchemy 的 pool 参数不兼容,
    # 既有问题),按 test_dictionary_listing.py 同款自建文件库
    from src.db.models import Base, get_db_engine, get_session
    engine = get_db_engine(str(temp_dir / "dict.db"))
    Base.metadata.create_all(engine)
    session = get_session(engine)
    session.add(DictionaryEntry(standard_tag="青年", aliases="年轻的"))
    session.add(DictionaryEntry(standard_tag="浴室", aliases="卫生间,澡堂"))
    session.add(DictionaryEntry(standard_tag="健身房", aliases=""))
    session.add(DictionaryEntry(standard_tag="双人", aliases="两个人,两人"))
    session.commit()
    manager = DictionaryManager(str(temp_dir / "none.md"), session)
    yield manager
    session.close()
    engine.dispose()


def test_map_rejects_sui_style_overgeneral_alias(dict_mgr):
    # "岁"类过泛别名已废除:含"18岁"不得映射出 青年
    mapped, _ = dict_mgr.map_raw_text_to_tags("18岁,后入")
    assert "青年" not in mapped


def test_map_alias_exact_token_hit(dict_mgr):
    mapped, _ = dict_mgr.map_raw_text_to_tags("卫生间,后入")
    assert "浴室" in mapped


def test_map_three_char_std_substring_hit(dict_mgr):
    mapped, _ = dict_mgr.map_raw_text_to_tags("豪华健身房场景,双人")
    assert "健身房" in mapped


def test_map_two_char_std_needs_exact_token(dict_mgr):
    mapped, _ = dict_mgr.map_raw_text_to_tags("新青年客栈,双人")
    assert "青年" not in mapped  # 2 字标准标签不再子串命中
    assert "双人" in mapped


def test_sanitize_alias_guard():
    assert sanitize_alias("岁") is None
    assert sanitize_alias("  ") is None
    assert sanitize_alias("12") is None
    assert sanitize_alias(" 丝袜 ") == "丝袜"
