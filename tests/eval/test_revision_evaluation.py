import numpy as np
from factrisk.eval.evaluation import aurc, selective, keep_top, calibrate, apply_calibration
from factrisk.datasets.real_data import german_values
from factrisk.pipeline.workflow import deployment_features


def test_tied_aurc_does_not_depend_on_input_order():
    a=np.array([0.,0.,1.,1.]);r=np.zeros(4)
    assert aurc(a,r)==aurc(a[::-1],r)==.5


def test_unknown_risk_bounds_keep_full_denominator():
    result=selective(np.array([1.,np.nan,0.,0.]),np.array([True,True,True,False]))
    assert result['risk_lower']==1/3
    assert result['risk_upper']==2/3
    assert result['coverage']==.75


def test_calibration_is_monotone():
    p=np.array([.1,.2,.8,.9]);y=np.array([0,1,0,1])
    ab=calibrate(p,y)
    assert np.all(np.diff(apply_calibration(p,ab))>=0)


def test_both_fact_channels_without_task_type():
    result=deployment_features('There are not twenty one cars.', ['There are twenty cars.'], 'There are not 21 cars.')
    assert result['evidence_number_mismatch']==0
    assert result['evidence_negation_mismatch']==0
    assert result['sample_number_disagreement']==1
    assert result['sample_negation_disagreement']==1


def test_german_composition_and_article():
    assert german_values('ein Zug mit einundzwanzig Wagen')==['21']
    assert german_values('zweitausendvierhundert Menschen')==['2400']


def test_shared_unknowns_cancel_for_identical_selections():
    from factrisk.eval.analysis import difference_bounds
    y=np.array([1.,np.nan,0.]);keep=np.array([True,True,False])
    assert difference_bounds(y,keep,keep)==dict(lower=0.,upper=0.)


def test_shared_unknown_bounds_match_enumeration():
    from itertools import product
    from factrisk.eval.analysis import difference_bounds
    y=np.array([1.,np.nan,0.,np.nan])
    a=np.array([True,True,False,True]);b=np.array([False,True,True,False])
    values=[]
    for labels in product((0.,1.),repeat=2):
        complete=y.copy();complete[~np.isfinite(y)]=labels
        values.append(complete[a].mean()-complete[b].mean())
    bounds=difference_bounds(y,a,b)
    assert np.isclose(bounds['lower'],min(values))
    assert np.isclose(bounds['upper'],max(values))


def test_adversarial_label_budget_matches_exhaustive_assignments():
    from itertools import product
    from factrisk.eval.analysis import label_flip_sensitivity
    y=np.array([0.,1.,np.nan,1.]);a=np.array([True,False,True,False]);b=np.ones(4,bool)
    result=label_flip_sensitivity(y,a,b)
    for budget in range(4):
        values=[]
        for candidate in product((0.,1.),repeat=4):
            candidate=np.array(candidate)
            if np.sum(candidate[np.isfinite(y)]!=y[np.isfinite(y)])<=budget:
                values.append(candidate[a].mean()-candidate[b].mean())
        assert np.isclose(result['worst_case_difference_by_flip_budget'][budget],max(values))
