from factrisk.core.text import (
    extract_numbers,
    normalized_distance,
    slot_correct,
    text_is_negative,
    unsupported_contrast,
)


def test_number_normalization_and_contrast() -> None:
    assert extract_numbers("There were twelve people and 3 cars.") == {"12", "3"}
    assert slot_correct("There were twelve people.", "12", "number")
    assert unsupported_contrast("There were thirteen people.", "12", "13", "number")


def test_negation() -> None:
    assert text_is_negative("The bridge is not open.")
    assert not text_is_negative("The bridge is open.")
    assert slot_correct("The bridge is not open.", "not open", "negation")


def test_normalized_distance_bounds() -> None:
    assert normalized_distance("a b c", "a b c") == 0.0
    assert 0.0 < normalized_distance("a b c", "a x c") <= 1.0

