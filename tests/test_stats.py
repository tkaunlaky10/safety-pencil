"""Unit tests for pencil.stats against known values (scipy references and hand-computed examples)."""

import math

import numpy as np
import pytest
from scipy import stats as sps

from pencil.stats import (
    best_at_threshold,
    cohen_kappa,
    cohen_kappa_ci,
    holm,
    holm_reject,
    mcnemar_exact,
    paired_bootstrap_diff,
    spearman_bootstrap_ci,
    wilson_ci,
)

NB = 500  # bootstrap replicates in tests (keeps the file fast)


# ------------------------------------------------------------------ Wilson


@pytest.mark.parametrize(
    "k,n,alpha",
    [(8, 10, 0.05), (8, 10, 0.10), (3, 17, 0.05), (97, 100, 0.05), (0, 10, 0.05), (10, 10, 0.05), (1, 1, 0.05)],
)
def test_wilson_matches_scipy(k, n, alpha):
    p, lo, hi = wilson_ci(k, n, alpha=alpha)
    ref = sps.binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="wilson")
    assert p == pytest.approx(k / n)
    assert lo == pytest.approx(ref.low, abs=1e-12)
    assert hi == pytest.approx(ref.high, abs=1e-12)
    assert 0.0 <= lo <= p <= hi <= 1.0


def test_wilson_8_of_10_hand_value():
    # z = 1.959964; (0.8 + z^2/20 -/+ z*sqrt(0.016 + z^2/400)) / (1 + z^2/10)
    _, lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4901625, abs=1e-6)
    assert hi == pytest.approx(0.9433178, abs=1e-6)


def test_wilson_edges():
    assert all(math.isnan(v) for v in wilson_ci(0, 0))
    p, lo, hi = wilson_ci(0, 25)
    assert p == 0.0 and lo == 0.0 and hi > 0.0
    p, lo, hi = wilson_ci(25, 25)
    assert p == 1.0 and hi == 1.0 and lo < 1.0
    assert wilson_ci(np.int64(8), np.int64(10)) == wilson_ci(8, 10)
    for bad in [(11, 10), (-1, 10), (2.5, 10), (1, -1)]:
        with pytest.raises(ValueError):
            wilson_ci(*bad)
    with pytest.raises(ValueError):
        wilson_ci(1, 10, alpha=0.0)


# ------------------------------------------------------------------ McNemar


def _paired(n11, b, c, n00):
    a = [1] * n11 + [1] * b + [0] * c + [0] * n00
    bb = [1] * n11 + [0] * b + [1] * c + [0] * n00
    return a, bb


def test_mcnemar_counts_and_p():
    a, bb = _paired(20, 7, 2, 11)
    r = mcnemar_exact(a, bb)
    assert (r["b"], r["c"], r["n"]) == (7, 2, 40)
    assert r["p_value"] == pytest.approx(sps.binomtest(7, 9, 0.5).pvalue, rel=1e-12)
    assert r["p_value"] == pytest.approx(sps.binomtest(2, 9, 0.5).pvalue, rel=1e-12)
    # hand value: 2 * P(X <= 2 | 9, 1/2) = 2 * (1 + 9 + 36) / 512
    assert r["p_value"] == pytest.approx(92 / 512, rel=1e-12)


def test_mcnemar_bools_order_and_symmetry():
    a, bb = _paired(5, 3, 12, 4)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(a))
    a_b = np.array(a, dtype=bool)[perm]
    b_b = np.array(bb, dtype=bool)[perm]
    r = mcnemar_exact(a_b, b_b)
    r_swap = mcnemar_exact(b_b, a_b)
    assert (r["b"], r["c"]) == (3, 12)
    assert (r_swap["b"], r_swap["c"]) == (12, 3)
    assert r["p_value"] == pytest.approx(r_swap["p_value"])
    assert r["p_value"] == pytest.approx(sps.binomtest(3, 15, 0.5).pvalue)


def test_mcnemar_no_discordant_and_errors():
    r = mcnemar_exact([1, 0, 1], [1, 0, 1])
    assert r == {"b": 0, "c": 0, "p_value": 1.0, "n": 3}
    with pytest.raises(ValueError):
        mcnemar_exact([1, 0], [1, 0, 1])
    with pytest.raises(ValueError):
        mcnemar_exact([1, 2, 0], [1, 0, 0])


# ------------------------------------------------------------------ Spearman bootstrap


def _xy(n=48, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 0.7 * x + 0.5 * rng.standard_normal(n)
    return np.round(x, 1), np.round(y, 1)  # rounding creates ties


def test_spearman_point_matches_scipy_and_ci_contains_it():
    x, y = _xy()
    r = spearman_bootstrap_ci(x, y, n_boot=NB, seed=0)
    assert r["rho"] == pytest.approx(sps.spearmanr(x, y).statistic, abs=1e-12)
    assert r["n"] == 48 and r["n_boot_used"] == NB
    assert -1.0 <= r["lo"] <= r["rho"] <= r["hi"] <= 1.0
    assert r["lo"] > 0.0


def test_spearman_bootstrap_replicates_match_scipy():
    # The vectorised replicate computation must equal spearmanr on the same resample.
    x, y = _xy(n=20)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 20, size=(NB, 20))
    ref = np.array([sps.spearmanr(x[i], y[i]).statistic for i in idx])
    lo, hi = np.quantile(ref, [0.025, 0.975])
    r = spearman_bootstrap_ci(x, y, n_boot=NB, seed=0)
    assert r["lo"] == pytest.approx(lo, abs=1e-12)
    assert r["hi"] == pytest.approx(hi, abs=1e-12)


def test_spearman_reproducible_with_seed():
    x, y = _xy()
    r1 = spearman_bootstrap_ci(x, y, n_boot=NB, seed=7)
    r2 = spearman_bootstrap_ci(x, y, n_boot=NB, seed=7)
    r3 = spearman_bootstrap_ci(x, y, n_boot=NB, seed=8)
    assert r1 == r2
    assert (r1["lo"], r1["hi"]) != (r3["lo"], r3["hi"])


def test_spearman_degenerate_resamples_skipped():
    x = np.array([0.0] * 9 + [1.0])  # ~35% of resamples have constant x
    y = np.arange(10.0)
    r = spearman_bootstrap_ci(x, y, n_boot=NB, seed=0)
    assert 0 < r["n_boot_used"] < NB
    assert np.isfinite(r["lo"]) and np.isfinite(r["hi"])
    const = spearman_bootstrap_ci(np.ones(10), y, n_boot=NB, seed=0)
    assert math.isnan(const["rho"]) and math.isnan(const["lo"]) and const["n_boot_used"] == 0
    with pytest.raises(ValueError):
        spearman_bootstrap_ci([1.0, 2.0], [1.0], n_boot=NB)
    with pytest.raises(ValueError):
        spearman_bootstrap_ci([1.0, np.nan, 3.0], [1.0, 2.0, 3.0], n_boot=NB)


# ------------------------------------------------------------------ paired bootstrap


def test_paired_bootstrap_1d():
    rng = np.random.default_rng(0)
    b = rng.random(200)
    a = b + 0.1 + 0.05 * rng.standard_normal(200)
    r = paired_bootstrap_diff(a, b, n_boot=NB, seed=0)
    assert r["diff"] == pytest.approx(a.mean() - b.mean())
    assert r["lo"] <= r["diff"] <= r["hi"]
    assert r["lo"] > 0.0
    assert r["p_two_sided"] < 0.01
    assert r == paired_bootstrap_diff(a, b, n_boot=NB, seed=0)


def test_paired_bootstrap_matches_hand_loop():
    rng = np.random.default_rng(5)
    a, b = rng.random(30), rng.random(30)
    r = paired_bootstrap_diff(a, b, n_boot=NB, seed=0)
    idx = np.random.default_rng(0).integers(0, 30, size=(NB, 30))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    assert r["lo"] == pytest.approx(np.quantile(d, 0.025))
    assert r["hi"] == pytest.approx(np.quantile(d, 0.975))
    assert r["p_two_sided"] == pytest.approx(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def test_paired_bootstrap_identical_inputs():
    a = np.random.default_rng(0).random(50)
    r = paired_bootstrap_diff(a, a.copy(), n_boot=NB)
    assert r == {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p_two_sided": 1.0}


def test_paired_bootstrap_2d_resamples_items():
    # With equal k per item, mean over the flattened rows equals the mean of row means, and the
    # same seed draws the same item indices, so 2-D and row-mean 1-D results must agree.
    rng = np.random.default_rng(2)
    A = rng.random((40, 7)) + 0.05
    B = rng.random((40, 7))
    r2 = paired_bootstrap_diff(A, B, n_boot=NB, seed=0)
    r1 = paired_bootstrap_diff(A.mean(axis=1), B.mean(axis=1), n_boot=NB, seed=0)
    for key in ("diff", "lo", "hi", "p_two_sided"):
        assert r2[key] == pytest.approx(r1[key], abs=1e-12)
    rm = paired_bootstrap_diff(A, B, stat=np.median, n_boot=NB, seed=0)
    assert rm["diff"] == pytest.approx(np.median(A) - np.median(B))
    assert rm["lo"] <= rm["hi"]
    with pytest.raises(ValueError):
        paired_bootstrap_diff(A, B[:-1], n_boot=NB)


# ------------------------------------------------------------------ best_at_threshold


def test_best_at_threshold():
    pts = [
        {"mu": 0.0, "safety": 0.90, "utility": 0.80},
        {"mu": 0.5, "safety": 0.97, "utility": 0.70},  # boundary counts (inclusive)
        {"mu": 1.0, "safety": 0.99, "utility": 0.65},
        {"mu": 2.0, "safety": 1.00, "utility": 0.40},
        {"mu": 3.0, "safety": float("nan"), "utility": 0.99},
    ]
    assert best_at_threshold(pts, 0.97) is pts[1]
    assert best_at_threshold(pts, 97 / 100) is pts[1]
    assert best_at_threshold(pts, 0.98) is pts[2]
    assert best_at_threshold(pts, 0.0) is pts[0]
    assert best_at_threshold(pts, 1.01) is None
    assert best_at_threshold([], 0.97) is None


# ------------------------------------------------------------------ Holm


def test_holm_textbook_example():
    # Four hypotheses at alpha = 0.05: sorted p = .005, .01, .03, .04 against .0125, .0167, .025:
    # the first two are rejected, then .03 > .025 stops the procedure.
    p = [0.01, 0.04, 0.03, 0.005]
    adj = holm(p)
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm_reject(p, 0.05) == [True, False, False, True]


def test_holm_monotone_clipped_and_edges():
    assert holm([0.5, 0.6, 0.7]) == [1.0, 1.0, 1.0]
    p = np.random.default_rng(0).random(12) ** 3
    adj = np.array(holm(p))
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= 0)
    assert np.all(adj >= p) and np.all(adj <= 1.0)
    assert holm([0.02]) == [0.02]
    assert holm([]) == [] and holm_reject([]) == []
    assert holm([0.01, 0.01]) == [0.02, 0.02]
    with pytest.raises(ValueError):
        holm([0.1, 1.2])
    with pytest.raises(ValueError):
        holm([0.1, float("nan")])


# ------------------------------------------------------------------ Cohen's kappa


def _from_table(table, labels):
    """Expand a confusion table (rows: rater 1, cols: rater 2) into two label lists."""
    y1, y2 = [], []
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            y1 += [li] * table[i][j]
            y2 += [lj] * table[i][j]
    return y1, y2


def _sklearn_kappa(table):
    # sklearn.metrics.cohen_kappa_score(weights=None), written out on the confusion matrix.
    cm = np.asarray(table, dtype=float)
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / cm.sum()
    w = np.ones_like(cm) - np.eye(cm.shape[0])
    return 1.0 - (w * cm).sum() / (w * expected).sum()


def test_kappa_textbook_2x2():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> p_o = .7, p_e = .5, kappa = .4
    y1, y2 = _from_table([[20, 5], [10, 15]], ["yes", "no"])
    assert cohen_kappa(y1, y2) == pytest.approx(0.4, abs=1e-12)


def test_kappa_matches_sklearn_formula_3x3():
    table = [[12, 3, 1], [2, 9, 4], [0, 5, 14]]
    labels = ["refuse", "comply", "partial"]
    y1, y2 = _from_table(table, labels)
    perm = np.random.default_rng(0).permutation(len(y1))
    y1 = [y1[i] for i in perm]
    y2 = [y2[i] for i in perm]
    assert cohen_kappa(y1, y2) == pytest.approx(_sklearn_kappa(table), abs=1e-12)
    assert cohen_kappa(y1, y2, labels=labels + ["unused"]) == pytest.approx(_sklearn_kappa(table))
    assert cohen_kappa(y2, y1) == pytest.approx(cohen_kappa(y1, y2))
    with pytest.raises(ValueError):
        cohen_kappa(y1, y2, labels=["refuse", "comply"])
    with pytest.raises(ValueError):
        cohen_kappa(y1, y2[:-1])


def test_kappa_edges():
    assert cohen_kappa([1, 0, 1, 1], [1, 0, 1, 1]) == 1.0
    assert math.isnan(cohen_kappa(["a"] * 5, ["a"] * 5))  # p_e = 1
    assert math.isnan(cohen_kappa([], []))
    assert cohen_kappa([True, False], [False, True]) == pytest.approx(-1.0)


def test_kappa_ci():
    table = [[30, 6], [8, 26]]
    y1, y2 = _from_table(table, [1, 0])
    perm = np.random.default_rng(1).permutation(len(y1))
    y1 = [y1[i] for i in perm]
    y2 = [y2[i] for i in perm]
    r = cohen_kappa_ci(y1, y2, n_boot=NB, seed=0)
    assert r["kappa"] == pytest.approx(_sklearn_kappa(table))
    assert r["n"] == 70 and r["n_boot_used"] == NB
    assert -1.0 <= r["lo"] <= r["kappa"] <= r["hi"] <= 1.0
    assert r == cohen_kappa_ci(y1, y2, n_boot=NB, seed=0)


def test_kappa_ci_degenerate():
    # Perfect agreement with a rare label: resamples drawing only "a" have p_e = 1 and are skipped.
    y = ["a"] * 9 + ["b"]
    r = cohen_kappa_ci(y, list(y), n_boot=NB, seed=0)
    assert r["kappa"] == 1.0 and r["lo"] == 1.0 and r["hi"] == 1.0
    assert 0 < r["n_boot_used"] < NB
    r = cohen_kappa_ci(["a"] * 6, ["a"] * 6, n_boot=NB)
    assert math.isnan(r["kappa"]) and math.isnan(r["lo"]) and r["n_boot_used"] == 0
