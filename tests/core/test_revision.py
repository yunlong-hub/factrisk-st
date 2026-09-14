import pytest
from factrisk.core.numbers import number_mentions
from factrisk.core.text import extract_numbers, slot_correct
from factrisk.core.labels_v2 import label_fact
from factrisk.datasets.data import _donor_table


@pytest.mark.parametrize('text,expected', [
    ('twenty one patients', {'21'}), ('twenty-one patients', {'21'}),
    ('one hundred', {'100'}), ('one hundred and twenty-three', {'123'}),
    ('two thousand five hundred', {'2500'}), ('twenty first', {'21'}),
    ('one hundredth', {'100'}), ('one point zero five', {'1.05'}),
    ('1,234.50 and 22nd', {'1234.5', '22'}), ('minus twenty one', {'-21'}),
    ('one or two', {'1', '2'}), ('twenty, one', {'20', '1'}),
])
def test_compositional_numbers(text, expected):
    assert extract_numbers(text) == expected


def test_no_partial_number_match():
    assert not slot_correct('twenty one patients', 'twenty', 'number')
    assert number_mentions('There are twenty-one patients.')[0]['start'] == 10


def test_context_avoids_wrong_slot():
    value = label_fact('There are 30 cats and 20 dogs.', '20', '30', 'number',
                      'There are 20 cats and 30 dogs.')
    # Symmetric nearby context cannot be resolved safely by this heuristic.
    assert value['severe_fact_error'] is None
    assert value['needs_human_review']


def test_negation_scope_not_declared_verified():
    value = label_fact('The station is open but the museum is not.', 'not', '', 'negation',
                      'The station is not open but the museum is.')
    assert value['needs_human_review']
    assert value['label_status'] == 'scope_proxy_requires_review'


def test_donors_stay_within_split(tmp_path):
    pairs = []
    for i in range(6):
        p = tmp_path / f'{i}.wav'
        p.write_bytes(b'audio')
        pairs.append(dict(id=str(i), meta=dict(speaker_id=f's{i}'),
                          original=dict(audio=str(p)), counterfactual=dict(audio=str(p))))
    splits = {str(i): ['train', 'calibration', 'test'][i // 2] for i in range(6)}
    table = _donor_table(pairs, [], {}, 1, pair_splits=splits)
    for (pid, _), path in table.items():
        assert splits[pid] == splits[path.stem]
        assert pid != path.stem
