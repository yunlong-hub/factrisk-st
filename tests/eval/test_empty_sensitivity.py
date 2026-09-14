"""Observed-empty vs missing and actual numerical-rule boundary contracts."""
import math

import pytest

from factrisk.eval.empty_sensitivity import distance_features, observed_texts
from factrisk.core.labels_v2 import label_fact
from factrisk.core.numbers import number_mentions
from factrisk.pipeline.workflow import deployment_features


def test_blank_probe_is_observed_distance_not_missing():
    p = dict(translation='three cats', probe_translations=[{'translation': ''}, {'translation': 'three cats'}], sample_translations=[])
    assert distance_features(p, False)['perturb_mean_distance'] == 0
    assert distance_features(p, True)['perturb_mean_distance'] == .5
    assert distance_features(p, True)['perturb_max_distance'] == 1
    assert math.isnan(distance_features(p, True)['sample_mean_distance'])


def test_blank_sample_participates_without_changing_signature_contract():
    p = dict(translation='three cats', probe_translations=[], sample_translations=['three cats', '', 'three cats'])
    assert distance_features(p, False)['sample_mean_distance'] == 0
    assert distance_features(p, True)['sample_mean_distance'] == pytest.approx(2/3)
    # Deployment signatures ALREADY include observed blank samples.
    assert deployment_features(p['translation'], p['sample_translations'], 'three cats')['sample_number_disagreement'] == pytest.approx(1/3)


@pytest.mark.parametrize('prediction', [
    {}, {'translation': None}, {'translation': '', 'probe_translations': [{}]},
    {'translation': '', 'sample_translations': [None]},
])
def test_missing_is_not_coerced_to_empty(prediction):
    with pytest.raises(ValueError):
        observed_texts(prediction)


def test_empty_base_is_observed_and_counts_as_number_omission():
    assert observed_texts({'translation': ''}) == ([], [])
    result = label_fact('', '3', '', 'number', 'Three cats arrived.')
    assert result['severe_fact_error'] == 1
    assert result['label_reason'] == 'wrong_or_missing_target_number'


def test_ambiguous_multiple_numbers_are_unresolved():
    result = label_fact('Three cats and four cats.', '3', '', 'number', 'Three cats arrived.')
    assert result['severe_fact_error'] is None
    assert result['label_reason'] == 'number_target_context_ambiguous'


def test_decimal_is_one_mention_not_integer_substrings():
    assert [m['value'] for m in number_mentions('3.5 liters')] == ['3.5']
    assert label_fact('3.5 liters', '3.5', '', 'number', '3.5 liters')['severe_fact_error'] == 0


def test_fraction_is_outside_compositional_number_parser_scope():
    # Document actual scope: slash fraction is two mentions, not value 0.5.
    assert [m['value'] for m in number_mentions('1/2 liter')] == ['1', '2']
    result = label_fact('1/2 liter', '1/2', '', 'number', '1/2 liter')
    assert result['severe_fact_error'] is None
    assert result['label_reason'] == 'reference_target_not_unique'
