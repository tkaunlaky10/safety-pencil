"""The scaled repair's μ grid, fixed from predictions before anything is measured (13 Nov, post-freeze).

For each target fraction f, μ(f) is where the predicted task regret with scaling falls to f of its value at μ = 0:

    R(μ) / R(0) = f,     R(μ) = Σ_modules t_k(μ)² R_k(μ)    (ModelRepair.predicted, no forward passes)

R(μ) goes from R(0) at μ = 0 (full removal) to 0 as μ → ∞. With scaling it need not be monotone, so each μ(f)
is the first crossing going up a coarse log grid (one μ per decade), refined by bisection in log μ. A dip below f
that rises back between two coarse points is not seen; the coarse curve is saved with the grid so this can be
checked.
"""

from __future__ import annotations

import math

COARSE_MU = [10.0**exponent for exponent in range(-6, 9)]  # 1e-6 … 1e8, one per decade
BISECTION_STEPS = 8  # inside one decade: μ to within a factor 10^(1/256), under 1%; each step costs one prediction


def mu_for_regret_fractions(predicted_R, fractions, coarse_mu=COARSE_MU, bisection_steps=BISECTION_STEPS) -> dict:
    """{"mu": [μ(f) per fraction], "fractions", "achieved": [R(μ)/R(0)], "R0", "coarse": [[μ, R/R(0)], ...]}.

    predicted_R(μ) returns the predicted total task regret at μ (μ = 0 allowed)."""
    R0 = predicted_R(0.0)
    if not R0 > 0:
        raise ValueError(f"predicted regret at μ = 0 is {R0}: nothing to repair")
    coarse = [(mu, predicted_R(mu) / R0) for mu in coarse_mu]
    chosen, achieved = [], []
    for fraction in fractions:
        lo, hi = _first_crossing(coarse, fraction)
        mu = _bisect_log(lambda m: predicted_R(m) / R0, fraction, lo, hi, bisection_steps)
        chosen.append(mu)
        achieved.append(predicted_R(mu) / R0)
    return {
        "mu": chosen,
        "fractions": list(fractions),
        "achieved": achieved,
        "R0": R0,
        "coarse": [list(pair) for pair in coarse],
    }


def _first_crossing(coarse: list[tuple[float, float]], fraction: float) -> tuple[float, float]:
    """(μ_lo, μ_hi): adjacent coarse μ with ratio(μ_lo) > fraction ≥ ratio(μ_hi), the first such pair."""
    if coarse[0][1] <= fraction:
        raise ValueError(f"R(μ)/R(0) is already {coarse[0][1]:.3g} at μ = {coarse[0][0]:g}, below {fraction}")
    for (mu_lo, ratio_lo), (mu_hi, ratio_hi) in zip(coarse, coarse[1:]):
        if ratio_lo > fraction >= ratio_hi:
            return mu_lo, mu_hi
    raise ValueError(f"R(μ)/R(0) never falls to {fraction} by μ = {coarse[-1][0]:g}")


def _bisect_log(ratio_at, fraction: float, lo: float, hi: float, steps: int) -> float:
    """Bisection in log μ for ratio_at(μ) = fraction on [lo, hi], where ratio(lo) > fraction ≥ ratio(hi)."""
    for _ in range(steps):
        middle = math.sqrt(lo * hi)
        if ratio_at(middle) > fraction:
            lo = middle
        else:
            hi = middle
    return math.sqrt(lo * hi)
