"""
Tests for ``talemate.agents.narrator.NarratorAgent.clean_result``.

Focuses on the screenplay speaker-line detection: narration lines that
merely start with a character's name must not be dropped, while genuine
uppercase screenplay headers still are.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

import talemate.agents.narrator as narrator_module
from talemate.agents.narrator import NarratorAgent


def make_narrator(character_names):
    narrator = NarratorAgent.__new__(NarratorAgent)
    narrator.scene = SimpleNamespace(
        get_characters=lambda: [
            SimpleNamespace(name=name) for name in character_names
        ]
    )
    return narrator


@pytest.fixture(autouse=True)
def no_editor_exposition():
    editor = SimpleNamespace(
        fix_exposition_enabled=False, fix_exposition_narrator=False
    )
    with mock.patch.object(narrator_module, "get_agent", return_value=editor):
        yield


def test_cjk_name_prefixed_narration_is_kept():
    # str.upper() is a no-op on CJK names, so the uppercase-surname branch
    # used to claim ANY line starting with the character's Chinese name.
    narrator = make_narrator(["由纪"])
    result = narrator.clean_result("*风声呼啸。*\n由纪把枪管拨了拨，添了根柴。")
    assert "由纪把枪管拨了拨" in result


def test_uppercase_screenplay_header_is_dropped():
    narrator = make_narrator(["Alice"])
    result = narrator.clean_result("ALICE: we should move now")
    assert result == ""


def test_lowercase_name_colon_dialogue_is_dropped():
    # quirk kept from upstream: the lowercase branch lowercases the line
    # but not the stored name, so it only matches names stored lowercase.
    narrator = make_narrator(["alice"])
    result = narrator.clean_result("alice: we should move now")
    assert result == ""

def test_mixed_case_name_colon_narration_is_kept():
    # consequence of the quirk above: a mixed-case name with colon is
    # treated as narration and kept (documented, not changed here).
    narrator = make_narrator(["Alice"])
    result = narrator.clean_result("Alice: we should move now")
    assert "Alice: we should move now" in result
