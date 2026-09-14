import pytest

from factrisk.pipeline.environment_replay import assert_equal


def test_metric_replay_rejects_changed_counts_and_missing_fields():
    assert_equal({'risk': 0.1 + 1e-14, 'kept': 90, 'unresolved': None},
                 {'risk': 0.1, 'kept': 90, 'unresolved': None})
    with pytest.raises(ValueError, match='kept'):
        assert_equal({'kept': 89}, {'kept': 90})
    with pytest.raises(ValueError, match='key mismatch'):
        assert_equal({'kept': 90}, {'kept': 90, 'errors': 2})


def test_metric_replay_rejects_nonfinite_or_material_drift():
    with pytest.raises(ValueError):
        assert_equal(float('nan'), 0.1)
    with pytest.raises(ValueError):
        assert_equal(0.1 + 1e-8, 0.1)
