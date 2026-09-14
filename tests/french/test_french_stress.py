import pytest

from factrisk.french.french_stress import donor_design


@pytest.mark.parametrize('speakers',[list('aabbc'),list('aaabbcc'),list('abcdef'),list('abcdefghijk')])
def test_donors_are_bijective_distinct_and_clustered(speakers):
    rows=[dict(id=str(i),speaker_id=s) for i,s in enumerate(speakers)]
    donors,clusters=donor_design(rows)
    assert set(donors)=={r['id'] for r in rows}
    assert {r['id'] for r in donors.values()}==set(donors)
    for row in rows:
        donor=donors[row['id']]
        assert row['speaker_id']!=donor['speaker_id']
        assert clusters[row['id']]==clusters[donor['id']]
    assert donor_design(list(reversed(rows)))==(donors,clusters)


def test_impossible_speaker_design_fails():
    with pytest.raises(ValueError,match='infeasible'):
        donor_design([dict(id=str(i),speaker_id=s) for i,s in enumerate('aaabc')])
