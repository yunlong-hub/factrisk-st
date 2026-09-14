import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from factrisk.french.french_evaluation import quality


def test_chrf_bootstrap_uses_shared_selections_and_donor_clusters(tmp_path):
    model=LogisticRegression().fit([[0],[1],[2],[3]],[0,0,1,1])
    cp=dict(model=model,features=['x'],calibration=np.array([1.,0.]),threshold=.5)
    for name in ('et_full','et_asr_evidence'):joblib.dump(cp,tmp_path/f'{name}.joblib')
    rows=[dict(id=str(i),speaker_id=str(i),cluster_id=str(i//2),
        translation=f'{i} cats',reference=f'{i} cats',features={'x':i}) for i in range(4)]
    result=quality(rows,tmp_path,replicates=10)
    assert result['clusters']==2
    assert result['cluster_unit']=='speaker_donor_component'
    assert result['base_chrf']==100
    assert result['methods']['et_full']['delta_ci95']==[0.,0.]
    assert result['full_minus_asr_evidence']['ci95']==[0.,0.]
