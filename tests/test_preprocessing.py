import numpy as np

from smartecg.data.preprocessing import bandpass, znorm_per_lead, preprocess_record


def test_znorm_zero_mean_unit_std_per_lead():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((12, 1000)).astype(np.float32) * 3.0 + 5.0
    y = znorm_per_lead(x)
    assert np.allclose(y.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(y.std(axis=1), 1, atol=1e-2)


def test_bandpass_preserves_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1000, 12)).astype(np.float32)
    y = bandpass(x, fs=100)
    assert y.shape == x.shape


def test_preprocess_record_returns_n_t():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((1000, 12)).astype(np.float32)
    out = preprocess_record(sig, fs=100)
    assert out.shape == (12, 1000)
    assert out.dtype == np.float32
