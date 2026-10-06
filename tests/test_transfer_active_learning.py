import numpy as np
from scripts.transfer_active_learning.run import projected,round_targets

def test_round_targets_match_final_budget():
    values=round_targets(14709,10)
    assert len(values)==11 and values[0]==1471 and values[-1]==14709
    assert np.all(np.diff(values)>0)

def test_projection_is_deterministic_and_bounded():
    values=np.arange(120,dtype=np.float32).reshape(10,12)
    first=projected(values,42,5);second=projected(values,42,5)
    assert first.shape==(10,5)
    np.testing.assert_array_equal(first,second)
