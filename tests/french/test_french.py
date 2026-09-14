import pytest
from factrisk.french.mentions import french_mentions, parse_integer


@pytest.mark.parametrize('text,value',[
    ('vingt et un',21),('soixante et onze',71),('quatre-vingts',80),
    ('quatre-vingt-dix-neuf',99),('deux cents',200),('deux cent trois',203),
    ('mille neuf cent quatre-vingt-dix',1990),('deux mille vingt et un',2021),
])
def test_french_composition(text,value):
    assert parse_integer(text)==value
    assert french_mentions('Il y avait '+text+' personnes.')[0]['value']==str(value)


def test_french_conservative_boundaries():
    assert french_mentions('Une femme lit un livre.')==[]
    assert [m['value'] for m in french_mentions('deux ou trois livres')]==['2','3']
    assert french_mentions('deux trois livres') is None
    assert french_mentions('trois millions de livres') is None
