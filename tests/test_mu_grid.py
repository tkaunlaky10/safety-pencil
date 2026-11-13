"""The scaled repair's μ grid rule (pencil.mu_grid)."""

import pytest

from pencil.mu_grid import mu_for_regret_fractions

FRACTIONS = [0.02, 0.05, 0.1, 0.2, 0.4]


def test_grid_hits_each_regret_fraction_of_a_known_curve():
    """R(μ) = R0 / (1 + μ/c): the exact answer is μ = c (1/f − 1)."""
    c, R0 = 0.7, 3.0
    grid = mu_for_regret_fractions(lambda mu: R0 / (1 + mu / c), FRACTIONS)
    for fraction, mu, achieved in zip(FRACTIONS, grid["mu"], grid["achieved"]):
        assert mu == pytest.approx(c * (1 / fraction - 1), rel=1e-2)  # 8 bisection steps: within a factor 10^(1/256)
        assert achieved == pytest.approx(fraction, rel=1e-2)  # 8 bisection steps: within a factor 10^(1/256)
    assert grid["R0"] == R0 and grid["mu"] == sorted(grid["mu"], reverse=True)


def test_grid_takes_the_first_crossing_of_a_non_monotone_curve():
    def bumpy(mu):  # falls through 0.4 at μ = 1.5, rises back above it after μ = 10, falls again after μ = 100
        if mu <= 10:
            return 1 / (1 + mu)
        if mu < 100:
            return 0.45
        return 0.45 * 100 / mu

    grid = mu_for_regret_fractions(bumpy, [0.4])
    assert grid["mu"][0] == pytest.approx(1.5, rel=1e-2)  # 8 bisection steps: within a factor 10^(1/256)


def test_grid_refuses_when_there_is_nothing_to_repair_or_the_target_is_out_of_reach():
    with pytest.raises(ValueError, match="nothing to repair"):
        mu_for_regret_fractions(lambda mu: 0.0, FRACTIONS)
    with pytest.raises(ValueError, match="never falls"):
        mu_for_regret_fractions(lambda mu: 1.0, FRACTIONS)
