import numpy as np
from scripts.transfer_active_learning.run import ensemble_uncertainty, projected, round_targets

def test_round_targets_match_final_budget():
    values=round_targets(14709,10)
    assert len(values)==11 and values[0]==1471 and values[-1]==14709
    assert np.all(np.diff(values)>0)

def test_projection_is_deterministic_and_bounded():
    values=np.arange(120,dtype=np.float32).reshape(10,12)
    first=projected(values,42,5);second=projected(values,42,5)
    assert first.shape==(10,5)
    np.testing.assert_array_equal(first,second)

def test_ensemble_uncertainty_is_sample_variance(monkeypatch):
    predictions = iter([np.array([2., 4.]), np.array([3., 7.])])
    def fake_fit(*args, **kwargs):
        values = next(predictions)
        return None, None, None, None, None, lambda index: values
    monkeypatch.setattr("scripts.transfer_active_learning.run.fit", fake_fit)
    base = lambda index: np.array([1., 1.])
    values = ensemble_uncertainty(
        "mlp_small", None, None,
        {"train": np.arange(4), "validation": np.arange(2)},
        42, "cpu", {}, np.arange(2), base, members=3,
    )
    np.testing.assert_allclose(values, np.var([[1., 1.], [2., 4.], [3., 7.]], axis=0, ddof=1))
