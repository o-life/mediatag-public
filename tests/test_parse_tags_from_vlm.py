"""Tests for _parse_tags_from_vlm function to ensure correct tag parsing.

This test suite verifies that the tag parsing logic correctly handles:
1. JSON array format: '["互撸", "后入"]'
2. Comma-separated format: '互撸,后入'
3. Malformed JSON fragments: '"互撸"', '"互撸"]', '"体位": ["互撸"'
4. Mixed formats and edge cases
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import _parse_tags_from_vlm


class TestParseTagsFromVLM:
    """Test the _parse_tags_from_vlm function."""

    def test_json_array_format(self):
        """Test parsing JSON array format."""
        result = _parse_tags_from_vlm('["互撸", "后入", "卧室"]')
        assert result == ["互撸", "后入", "卧室"]

    def test_json_array_single_item(self):
        """Test parsing JSON array with single item."""
        result = _parse_tags_from_vlm('["互撸"]')
        assert result == ["互撸"]

    def test_json_dict_format(self):
        """Test parsing JSON dict format like {"体位": ["互撸"]}."""
        result = _parse_tags_from_vlm('{"体位": ["互撸", "后入"]}')
        assert result == ["互撸", "后入"]

    def test_json_dict_multiple_keys(self):
        """Test parsing JSON dict with multiple keys."""
        result = _parse_tags_from_vlm('{"体位": ["互撸"], "场景": ["卧室"]}')
        assert set(result) == {"互撸", "卧室"}

    def test_comma_separated_format(self):
        """Test parsing comma-separated format."""
        result = _parse_tags_from_vlm("互撸,后入,卧室")
        assert result == ["互撸", "后入", "卧室"]

    def test_comma_separated_with_spaces(self):
        """Test parsing comma-separated format with spaces."""
        result = _parse_tags_from_vlm("互撸, 后入, 卧室")
        assert result == ["互撸", "后入", "卧室"]

    def test_malformed_json_with_quotes(self):
        """Test parsing malformed JSON: '"互撸"'."""
        result = _parse_tags_from_vlm('"互撸"')
        assert result == ["互撸"]

    def test_malformed_json_with_bracket(self):
        """Test parsing malformed JSON: '"互撸"]'."""
        result = _parse_tags_from_vlm('"互撸"]')
        assert result == ["互撸"]

    def test_malformed_json_key_value_fragment(self):
        """Test parsing malformed JSON: '"体位": ["互撸"'."""
        result = _parse_tags_from_vlm('"体位": ["互撸"')
        assert result == ["互撸"]

    def test_malformed_multiple_fragments(self):
        """Test parsing multiple malformed fragments."""
        result = _parse_tags_from_vlm('"互撸", "后入"]')
        assert result == ["互撸", "后入"]

    def test_empty_string(self):
        """Test parsing empty string."""
        result = _parse_tags_from_vlm("")
        assert result == []

    def test_none_input(self):
        """Test parsing None input."""
        result = _parse_tags_from_vlm(None)
        assert result == []

    def test_whitespace_only(self):
        """Test parsing whitespace-only string."""
        result = _parse_tags_from_vlm("   ")
        assert result == []

    def test_single_tag(self):
        """Test parsing single tag."""
        result = _parse_tags_from_vlm("互撸")
        assert result == ["互撸"]

    def test_tags_with_special_characters(self):
        """Test parsing tags with special characters."""
        result = _parse_tags_from_vlm("双人-后入, 卧室_场景")
        assert result == ["双人-后入", "卧室_场景"]

    def test_json_array_with_whitespace(self):
        """Test parsing JSON array with extra whitespace."""
        result = _parse_tags_from_vlm('  ["互撸", "后入"]  ')
        assert result == ["互撸", "后入"]

    def test_mixed_json_and_comma_separated(self):
        """Test that JSON format takes precedence when valid."""
        # Valid JSON should be parsed as JSON
        result = _parse_tags_from_vlm('["互撸", "后入"]')
        assert result == ["互撸", "后入"]

    def test_json_with_numbers(self):
        """Test that non-string items in JSON are ignored."""
        result = _parse_tags_from_vlm('["互撸", 123, "后入"]')
        assert result == ["互撸", "后入"]

    def test_json_dict_with_string_values(self):
        """Test parsing JSON dict with string values."""
        result = _parse_tags_from_vlm('{"体位": "互撸"}')
        assert result == ["互撸"]

    def test_comma_separated_with_quotes(self):
        """Test parsing comma-separated with quotes around tags."""
        result = _parse_tags_from_vlm('"互撸","后入"')
        assert result == ["互撸", "后入"]

    def test_complex_malformed_json(self):
        """Test parsing complex malformed JSON fragments."""
        # Multiple key-value fragments
        result = _parse_tags_from_vlm('"体位": ["互撸"], "场景": ["卧室"')
        assert result == ["互撸", "卧室"]

    def test_nested_brackets(self):
        """Test handling of nested brackets."""
        result = _parse_tags_from_vlm('[["互撸", "后入"]]')
        # Should extract string values from nested structure
        assert "互撸" in result
        assert "后入" in result

    def test_empty_json_array(self):
        """Test parsing empty JSON array."""
        result = _parse_tags_from_vlm('[]')
        assert result == []

    def test_empty_json_dict(self):
        """Test parsing empty JSON dict."""
        result = _parse_tags_from_vlm('{}')
        assert result == []

    def test_real_world_vlm_output_1(self):
        """Test real-world VLM output example 1."""
        vlm_output = '["双人", "后入", "卧室", "肌肉男"]'
        result = _parse_tags_from_vlm(vlm_output)
        assert result == ["双人", "后入", "卧室", "肌肉男"]

    def test_real_world_vlm_output_2(self):
        """Test real-world VLM output example 2."""
        vlm_output = '双人,后入,卧室'
        result = _parse_tags_from_vlm(vlm_output)
        assert result == ["双人", "后入", "卧室"]

    def test_real_world_malformed_output(self):
        """Test real-world malformed VLM output."""
        # This is the actual bug we're fixing
        vlm_output = '"互撸"'
        result = _parse_tags_from_vlm(vlm_output)
        assert result == ["互撸"]
        assert result[0] == "互撸"  # No quotes

    def test_preserves_chinese_characters(self):
        """Test that Chinese characters are preserved correctly."""
        result = _parse_tags_from_vlm('["互撸", "后入", "传教士式"]')
        assert result == ["互撸", "后入", "传教士式"]

    def test_handles_unicode_whitespace(self):
        """Test handling of various whitespace characters."""
        result = _parse_tags_from_vlm("互撸\t,\n后入\r, 卧室")
        assert result == ["互撸", "后入", "卧室"]

    def test_markdown_code_block_json_dict(self):
        """Test parsing JSON dict wrapped in markdown code block."""
        result = _parse_tags_from_vlm('```json\n{"体位": ["后入"]}\n```')
        assert result == ["后入"]

    def test_markdown_code_block_json_dict_multiple_keys(self):
        """Test parsing JSON dict with multiple keys wrapped in markdown code block."""
        result = _parse_tags_from_vlm('```json\n{"体位": ["后入"], "动作": ["抚摸"]}\n```')
        assert set(result) == {"后入", "抚摸"}

    def test_markdown_code_block_json_array(self):
        """Test parsing JSON array wrapped in markdown code block."""
        result = _parse_tags_from_vlm('```\n["后入", "抚摸"]\n```')
        assert result == ["后入", "抚摸"]

    def test_markdown_code_block_json_array_with_json_tag(self):
        """Test parsing JSON array wrapped in markdown code block with json tag."""
        result = _parse_tags_from_vlm('```json\n["后入", "抚摸"]\n```')
        assert result == ["后入", "抚摸"]

    def test_markdown_code_block_with_extra_whitespace(self):
        """Test parsing markdown code block with extra whitespace."""
        result = _parse_tags_from_vlm('  ```json\n{"体位": ["后入"]}\n```  ')
        assert result == ["后入"]


class TestParseTagsIntegration:
    """Integration tests to ensure parsed tags work with database operations."""

    def test_parsed_tags_are_clean_strings(self):
        """Ensure all parsed tags are clean strings without JSON artifacts."""
        test_cases = [
            '["互撸"]',
            '"互撸"',
            '"互撸"]',
            '"体位": ["互撸"',
            "互撸",
            "互撸,后入",
        ]
        
        for test_input in test_cases:
            result = _parse_tags_from_vlm(test_input)
            for tag in result:
                # Should not contain JSON structural characters
                assert "[" not in tag, f"Tag '{tag}' from input '{test_input}' contains '['"
                assert "]" not in tag, f"Tag '{tag}' from input '{test_input}' contains ']'"
                assert "{" not in tag, f"Tag '{tag}' from input '{test_input}' contains '{{'"
                assert "}" not in tag, f"Tag '{tag}' from input '{test_input}' contains '}}'"
                assert ":" not in tag, f"Tag '{tag}' from input '{test_input}' contains ':'"
                # Should not start or end with quotes
                assert not tag.startswith('"'), f"Tag '{tag}' from input '{test_input}' starts with '\"'"
                assert not tag.endswith('"'), f"Tag '{tag}' from input '{test_input}' ends with '\"'"
                assert not tag.startswith("'"), f"Tag '{tag}' from input '{test_input}' starts with '''"
                assert not tag.endswith("'"), f"Tag '{tag}' from input '{test_input}' ends with '''"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
