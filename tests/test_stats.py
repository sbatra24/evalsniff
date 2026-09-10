import numpy as np

from stats import bootstrap_mean_ci, energy_distance, paired_energy_test, paired_permutation_test


def test_permutation_p_values_are_uniform_under_null():
    """Under the null, p-values should be roughly Uniform(0, 1)."""
    rng = np.random.default_rng(0)
    ps = []
    for i in range(200):
        a = rng.random(40) < 0.5
        b = rng.random(40) < 0.5
        ps.append(paired_permutation_test(a.astype(float), b.astype(float), n_perm=500, seed=i).p_value)
    ps = np.array(ps)
    # Kolmogorov-Smirnov style check against the uniform CDF (discreteness
    # of the statistic makes p-values conservative, so allow some slack).
    grid = np.linspace(0.05, 0.95, 19)
    empirical = np.array([(ps <= g).mean() for g in grid])
    assert np.all(empirical <= grid + 0.12)
    assert (ps < 0.05).mean() <= 0.09
    assert (ps < 0.5).mean() >= 0.3


def test_permutation_detects_a_shift():
    rng = np.random.default_rng(1)
    a = (rng.random(120) < 0.95).astype(float)
    b = (rng.random(120) < 0.55).astype(float)
    res = paired_permutation_test(a, b, n_perm=5000, seed=0)
    assert res.statistic > 0.3
    assert res.p_value < 0.001


def test_permutation_p_value_never_zero_and_symmetric():
    a = np.ones(30)
    b = np.zeros(30)
    res = paired_permutation_test(a, b, n_perm=1000, seed=0)
    assert 0 < res.p_value <= 2 / 1001
    rev = paired_permutation_test(b, a, n_perm=1000, seed=0)
    assert rev.p_value == res.p_value and rev.statistic == -res.statistic


def test_energy_distance_properties():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 3))
    assert abs(energy_distance(x, x)) < 1e-9
    y = rng.normal(size=(50, 3)) + 3.0
    assert energy_distance(x, y) > 1.0


def test_energy_test_null_and_shift():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 2))
    y = rng.normal(size=(80, 2))
    null = paired_energy_test(x, y, n_perm=300, seed=0)
    assert null.p_value > 0.01
    y2 = y.copy()
    y2[:, 0] += 1.0
    shift = paired_energy_test(x, y2, n_perm=300, seed=0)
    assert shift.p_value < 0.01


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(4)
    values = rng.normal(loc=0.3, scale=1.0, size=200)
    mean, lo, hi = bootstrap_mean_ci(values, n_boot=1000, seed=0)
    assert lo < mean < hi
    assert lo < 0.3 < hi
    assert hi - lo < 0.5
