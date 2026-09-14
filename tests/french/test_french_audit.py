import pytest

from factrisk.french.mentions import french_mentions, parse_integer
from factrisk.french.french_audit import reference_checks


@pytest.mark.parametrize('spelling,value', [
    ('zéro', 0), ('dix-sept', 17), ('soixante-dix', 70),
    ('soixante-et-onze', 71), ('quatre-vingts', 80),
    ('quatre-vingt-un', 81), ('quatre-vingt-dix-neuf', 99),
    ('deux cents', 200), ('deux cent un', 201),
    ('mille', 1000), ('deux mille', 2000),
    ('neuf cent quatre-vingt-dix-neuf mille neuf cent quatre-vingt-dix-neuf', 999999),
])
def test_independently_specified_french_integers(spelling, value):
    assert parse_integer(spelling) == value
    assert [m['value'] for m in french_mentions('Il reste '+spelling+' personnes.')] == [str(value)]


def test_reference_properties_cover_changed_missing_and_equivalent_values():
    checks = reference_checks({'reference': 'There are eighty-one chairs.', 'expected_slot': '81'})
    assert len(checks) == 5
    assert all(checks.values())


def test_article_not_silently_treated_as_a_number():
    # An unsupported run may reject the sentence rather than return an empty list.
    assert french_mentions('Une femme et un enfant.') in (None, [])
