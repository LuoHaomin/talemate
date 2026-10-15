"""
Tests for the native MiniMax visual backend helpers.
"""

from talemate.agents.visual.backends.minimax_image import (
    aspect_ratio_for,
    clamp_prompt,
    resolution_for_format,
)
from talemate.agents.visual.schema import FORMAT_TYPE, Resolution


class TestClampPrompt:
    def test_short_prompt_untouched(self):
        prompt = "雪原上的篝火"
        assert clamp_prompt(prompt) == prompt

    def test_long_prompt_cut_at_sentence_boundary(self):
        prompt = ("一句话。" * 600) + "尾巴不该出现"
        result = clamp_prompt(prompt)
        assert len(result) <= 1400
        assert result.endswith("。")
        assert "尾巴" not in result

    def test_no_separator_falls_back_to_hard_cut(self):
        prompt = "字" * 2000
        result = clamp_prompt(prompt)
        assert len(result) == 1400

    def test_uses_chinese_and_ascii_separators(self):
        prompt = ("第几段！" * 400) + "多余的"
        result = clamp_prompt(prompt)
        assert result.endswith("！")


class TestAspectRatio:
    def test_square(self):
        assert aspect_ratio_for(Resolution(width=1024, height=1024)) == "1:1"

    def test_landscape(self):
        assert aspect_ratio_for(Resolution(width=1600, height=900)) == "16:9"

    def test_portrait(self):
        assert aspect_ratio_for(Resolution(width=900, height=1600)) == "9:16"

    def test_snap_to_closest(self):
        # 1792x1024 (7:8) is closest to 16:9 (1.777)
        assert aspect_ratio_for(Resolution(width=1792, height=1024)) == "16:9"

    def test_zero_dimensions_fallback(self):
        assert aspect_ratio_for(Resolution(width=0, height=0)) == "1:1"


class TestResolutionForFormat:
    def test_landscape(self):
        assert resolution_for_format(FORMAT_TYPE.LANDSCAPE) == Resolution(
            width=1600, height=900
        )

    def test_portrait(self):
        assert resolution_for_format(FORMAT_TYPE.PORTRAIT) == Resolution(
            width=900, height=1600
        )

    def test_other_is_square(self):
        assert resolution_for_format(FORMAT_TYPE.SQUARE) == Resolution(
            width=1024, height=1024
        )


class TestRegistration:
    def test_backend_registered(self):
        import talemate.agents.visual.backends as backends
        import talemate.agents.visual.backends.minimax_image  # noqa: F401

        assert "minimax" in backends.BACKENDS
        assert backends.BACKENDS["minimax"].image_create is True
