import pytest

from rake import Rake

INVALID_ABBREVIATIONS = [
    "this is not an abbreviation",
    "bad abbrevation lenth (TOOMANY)",
    "not uppercase abbreviation (bad)",
    "contains space (go od)",
    "too short (TST)",
]

VALID_ABBREVIATIONS = [
    "this is a good abbreviation (TIAGA)",
    "this-is good (TIG)",
    "this-is good (TIG)",
]


@pytest.fixture
def rake():
    """Create Rake object and pass to calling functions"""
    yield Rake()


def test_init(rake):
    """Tests behaviour of Rake object"""
    assert rake is not None
    assert len(rake.stop_list) > 0
    assert len(rake.frequent_words) > 0
    assert len(rake.get_phrases("This is a test sentence with some key words")) > 0
    assert (
        len(rake.get_phrases("This is a test sentence with some key words", length=1))
        == 1
    )


def test_invalid_abbreviations(rake):
    """Tests for invalid abbreviations"""
    for text in INVALID_ABBREVIATIONS:
        abbreviation = rake.get_abbreviations(text)
        assert len(abbreviation) == 0


def test_valid_abbreviations(rake):
    """Tests for valid abbreviations"""
    for text in VALID_ABBREVIATIONS:
        abbreviation = rake.get_abbreviations(text)
        assert len(abbreviation) == 1
