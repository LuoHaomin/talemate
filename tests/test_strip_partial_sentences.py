import pytest
from talemate.util import strip_partial_sentences


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            "This is a test{delim} This is a test{delim}",
            "This is a test{delim} This is a test{delim}",
        ),
        ("This is a test{delim} This is a test", "This is a test{delim}"),
        ("This is a test{delim}\nThis is a test", "This is a test{delim}"),
    ],
)
def test_strip_partial_sentences(input, expected):
    delimiters = [".", "!", "?", '"', "*"]

    for delim in delimiters:
        input = input.format(delim=delim)
        expected = expected.format(delim=delim)
        assert strip_partial_sentences(input) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        # Closing square brackets should be preserved
        ("I walked over. [I smile.]", "I walked over. [I smile.]"),
        ("I walked over. [I smile!]", "I walked over. [I smile!]"),
        ("I walked over. [I smile?]", "I walked over. [I smile?]"),
        # Closing parentheses should be preserved
        ("I walked over. (I smile.)", "I walked over. (I smile.)"),
        ("I walked over. (I smile!)", "I walked over. (I smile!)"),
        ("I walked over. (I smile?)", "I walked over. (I smile?)"),
        # Closing curly braces should be preserved
        ("I walked over. {I smile.}", "I walked over. {I smile.}"),
        # Partial sentence after complete bracketed expression should be stripped
        (
            "I walked over. [I smile.] And then",
            "I walked over. [I smile.]",
        ),
        (
            "I walked over. (I smile.) And then",
            "I walked over. (I smile.)",
        ),
        # Nested brackets
        (
            "I walked over. [I smile (happily.)]",
            "I walked over. [I smile (happily.)]",
        ),
        # Balanced closing bracket without sentence ending inside is still valid
        ("I walked over. [I smile]", "I walked over. [I smile]"),
        ("I walked over. (I nod)", "I walked over. (I nod)"),
        ("I walked over. {I smile}", "I walked over. {I smile}"),
        # Partial sentence after balanced bracket expression should be stripped
        ("I walked over. [I smile] And then", "I walked over. [I smile]"),
        ("I walked over. (I nod) And then", "I walked over. (I nod)"),
        ("I walked over. {I smile} And then", "I walked over. {I smile}"),
        # Unbalanced closer should not be included
        ("I walked over.] something", "I walked over."),
    ],
)
def test_strip_partial_sentences_with_brackets(input, expected):
    assert strip_partial_sentences(input) == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        # CJK sentence terminators must count as valid endings; with the old
        # ASCII-only list the scan fell back to the closing quote and cut all
        # narration after the dialogue line.
        (
            '"今晚别走了。"她转身走向门口。',
            '"今晚别走了。"她转身走向门口。',
        ),
        (
            '*她抬头。*"站住。"*她举起枪，对准门缝。*',
            '*她抬头。*"站住。"*她举起枪，对准门缝。*',
        ),
        # corner brackets are balanced closers
        ("「今晚别走了。」", "「今晚别走了。」"),
        # a genuinely unfinished CJK tail is still stripped back to 。 / ！ / ？
        (
            '"站住。"她转身走向门口，',
            '"站住。"',
        ),
    ],
)
def test_strip_partial_sentences_cjk(input, expected):
    assert strip_partial_sentences(input) == expected
