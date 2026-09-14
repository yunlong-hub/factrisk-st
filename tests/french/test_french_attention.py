import pytest
from factrisk.french.french_attention import feature_matrix


def test_attention_rows_join_by_id():
    rows=[{'id':'a'},{'id':'b'}]
    cached=[dict(id='b',feature_shape=[1,1,2],features=[3.,4.]),
        dict(id='a',feature_shape=[1,1,2],features=[1.,2.])]
    x,_=feature_matrix(rows,cached,[1,1,2])
    assert x.tolist()==[[1.,2.],[3.,4.]]


@pytest.mark.parametrize('cache',[
    [dict(id='a',feature_shape=[1],features=[0.])]*2,
    [dict(id='b',feature_shape=[1],features=[0.])],
    [dict(id='a',feature_shape=[2],features=[0.,0.])],
    [dict(id='a',feature_shape=[1],features=[float('nan')])]])
def test_attention_invalid_cache_rejected(cache):
    with pytest.raises(ValueError):feature_matrix([dict(id='a')],cache,[1])
