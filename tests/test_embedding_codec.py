"""人脸 embedding 的存储编解码（JSON 文本 → float32 BLOB）。

为什么需要双向兼容：迁移是分步的——旧库仍是 JSON、新库已是 BLOB。
`decode_embedding` 必须两种都能解，否则过渡期会崩。

规模背景（2026-09-18 实测）：JSON 文本 ≈ 10 KB/行（512 维），
float32 BLOB = 2 KB/行；39,531 行转换后库从 1.2 GB 降到 838 MB。
"""

import json

import numpy as np
import pytest

from src.core.embedding_codec import decode_embedding, encode_embedding


class TestEncode:
    def test_none_passthrough(self):
        assert encode_embedding(None) is None

    def test_returns_bytes_of_expected_length(self):
        blob = encode_embedding(np.zeros(512, dtype=np.float32))
        assert isinstance(blob, bytes)
        assert len(blob) == 512 * 4          # float32 = 4 字节

    def test_accepts_list_and_downcasts(self):
        blob = encode_embedding([0.5, -1.25])
        arr = np.frombuffer(blob, dtype=np.float32)
        assert arr.tolist() == pytest.approx([0.5, -1.25])

    def test_flattens(self):
        blob = encode_embedding(np.zeros((2, 3), dtype=np.float32))
        assert len(blob) == 6 * 4


class TestDecode:
    def test_blob_roundtrip(self):
        original = np.random.RandomState(0).randn(512).astype(np.float32)
        got = decode_embedding(encode_embedding(original))
        assert got is not None
        assert np.allclose(got, original, atol=0)

    def test_legacy_json_text_still_readable(self):
        """关键契约：老格式（JSON 文本）必须仍能解。"""
        values = [0.1, -0.2, 0.3]
        got = decode_embedding(json.dumps(values))
        assert got is not None
        assert got.tolist() == pytest.approx(values, abs=1e-7)

    def test_both_formats_agree(self):
        """同一向量的两种存储形态，解出来必须一致（float32 精度内）。"""
        values = [0.123456789, -1.987654321, 3.14159265]
        arr = np.asarray(values, dtype=np.float32)
        from_text = decode_embedding(json.dumps(values))
        from_blob = decode_embedding(encode_embedding(arr))
        assert np.array_equal(from_text, from_blob)

    def test_returns_writable_array(self):
        """调用方可能就地归一化，而 np.frombuffer 的结果是只读的。"""
        arr = decode_embedding(encode_embedding(np.ones(4, dtype=np.float32)))
        arr /= 2.0                            # 只读数组会在此抛错
        assert arr.tolist() == [0.5] * 4

    def test_bytes_like_inputs(self):
        blob = encode_embedding(np.ones(3, dtype=np.float32))
        assert decode_embedding(bytearray(blob)) is not None
        assert decode_embedding(memoryview(blob)) is not None

    @pytest.mark.parametrize("bad", [None, "", "   ", "not-json", "{}", b"", [], "[]"])
    def test_invalid_returns_none_instead_of_raising(self, bad):
        """原代码靠 try/except 容错；新实现必须自己吞掉坏值。"""
        assert decode_embedding(bad) is None

    def test_sequence_input_passthrough(self):
        got = decode_embedding([1.0, 2.0])
        assert got is not None and got.tolist() == [1.0, 2.0]


class TestSizeReduction:
    def test_blob_is_much_smaller_than_json(self):
        """512 维：JSON ≈ 10 KB，BLOB = 2 KB（约 5 倍）。"""
        vec = np.random.RandomState(1).randn(512).astype(np.float32)
        json_len = len(json.dumps(vec.tolist()))
        blob_len = len(encode_embedding(vec))
        assert blob_len == 2048
        assert json_len > blob_len * 4
