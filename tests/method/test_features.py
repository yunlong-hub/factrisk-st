from factrisk.method.features import critical_fact_labels, sentence_chrf


def test_empty_decoding_is_retained_but_missing_translation_rejected(monkeypatch):
    import pytest
    from factrisk.method.features import _feature_row
    from factrisk.core.labels_v2 import label_fact
    monkeypatch.setattr('factrisk.method.features._cached_signal_features', lambda path: {})
    row=dict(id='empty',pair_id='p',side='original',split='test',speaker_id='s',
        condition='noise_0db',evidence_insufficient=False,fact_type='number',
        tts_engine='natural_speech',reference='Twelve passengers.',expected_slot='12',
        contrast_slot='',audio='unused.wav')
    result=_feature_row(row,dict(translation=''),dict(cascade_translation='Twelve passengers.'),0.)
    assert result['translation']==''
    assert result['features']['output_length']==0
    assert result['features']['evidence_text_distance']==1
    assert label_fact('', '12', '', 'number', row['reference'])['severe_fact_error']==1
    with pytest.raises(ValueError, match='Missing or non-text'):
        _feature_row(row,{},None,None)


def test_sentence_chrf() -> None:
    assert sentence_chrf("the answer is twelve", "the answer is twelve") == 100.0
    assert sentence_chrf("the answer is twelve", "completely unrelated") < 30.0


def test_frozen_critical_fact_label() -> None:
    correct, unsupported, severe = critical_fact_labels(
        "There were twelve passengers.", "12", "13", "number"
    )
    assert (correct, unsupported, severe) == (True, False, False)
    correct, unsupported, severe = critical_fact_labels(
        "There were thirteen passengers.", "12", "13", "number"
    )
    assert (correct, unsupported, severe) == (False, True, True)
