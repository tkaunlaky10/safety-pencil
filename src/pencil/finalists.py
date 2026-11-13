"""The fixed rule that picks the finalists of the test-tier comparison, from dev results only.

μ* is the smallest μ of the sweep that keeps at least KEEP_TASK_GAIN of the fine-tune's task gain on dev:

    NLL(μ*) ≤ NLL(θ1) + (1 − KEEP_TASK_GAIN) · (NLL(θ0) − NLL(θ1))

(smaller μ repairs harder, so this is the strongest repair that still keeps the task). μ whose trust-region check
failed (rule 11: "do not evaluate this point") are not candidates. Each compared baseline's finalist is its
setting whose dev task NLL is closest to NLL(μ*), so every method is compared at about the same task score.
Fixed on 12 Nov before any finalist was evaluated (PREREGISTRATION.md deviation log).

Backup μ* (13 Nov, before the scaled run): if no accepted μ meets the limit, μ* is the accepted μ with the lowest
dev task NLL, marked "backup": True so the results and the summary say "backup μ*". If no μ is accepted at all,
there is no μ*: pencil_mu_star raises NoAcceptedMu (the stage has already stopped after the sweep, see
09_pick_test_settings.py --any-passing).

Matched-NLL baselines (13 Nov, for P6): a baseline's grid rarely lands on NLL(μ*), so its knob is refined toward
that NLL by regula falsi inside the bracketing pair of measured settings (log μ for RegMean and diagonal Fisher,
linear α for WiSE-FT), at most MATCH_STEPS extra settings, stopping within MATCH_TOLERANCE. It looks only at task
NLL, never at S.
"""

from __future__ import annotations

import math

KEEP_TASK_GAIN = 0.9
MATCH_TOLERANCE = 0.005  # nats per token: a baseline setting this close to NLL(μ*) counts as matched
MATCH_STEPS = 3  # extra settings measured per baseline while matching


def task_nll_limit(nll_theta0: float, nll_theta1: float) -> float:
    """The largest dev task NLL that still keeps KEEP_TASK_GAIN of the gain NLL(θ0) − NLL(θ1)."""
    return nll_theta1 + (1 - KEEP_TASK_GAIN) * (nll_theta0 - nll_theta1)


def accepted(point: dict) -> bool:
    """A sweep point that was measured and passed rule 11's trust-region check."""
    return "task_nll" in point and not point.get("trust_region_failed", False)


class NoAcceptedMu(ValueError):
    """No μ of the sweep passed rule 11, so there is no μ* (not even a backup)."""


def pencil_mu_star(points: list[dict], nll_theta0: float, nll_theta1: float) -> dict:
    """{"mu", "task_nll", "limit", "backup"}: the smallest accepted μ within the task NLL limit, or, when none
    is within it, the accepted μ with the lowest dev task NLL (backup = True). Raises NoAcceptedMu when no μ
    passed rule 11."""
    limit = task_nll_limit(nll_theta0, nll_theta1)
    accepted_points = [point for point in points if accepted(point)]
    if not accepted_points:
        raise NoAcceptedMu("no μ passed rule 11: there is no μ*")
    within_limit = [point for point in accepted_points if point["task_nll"] <= limit]
    if within_limit:
        best = min(within_limit, key=lambda point: point["mu"])
        return {"mu": best["mu"], "task_nll": best["task_nll"], "limit": limit, "backup": False}
    best = min(accepted_points, key=lambda point: point["task_nll"])
    return {"mu": best["mu"], "task_nll": best["task_nll"], "limit": limit, "backup": True}


def matched(row: dict, target_nll: float) -> bool:
    """A measured row within MATCH_TOLERANCE of target_nll."""
    return "task_nll" in row and abs(row["task_nll"] - target_nll) <= MATCH_TOLERANCE


def next_setting_for_nll(rows: list[dict], target_nll: float, log_scale: bool) -> float | None:
    """The next knob value to try: regula falsi between the two adjacent measured settings whose task NLLs
    straddle target_nll, in log space when log_scale. None when no adjacent pair straddles it (the target is
    outside the measured range) or a row is already matched."""
    measured = sorted((row for row in rows if "task_nll" in row), key=lambda row: row["setting"])
    if any(matched(row, target_nll) for row in measured):
        return None
    for left, right in zip(measured, measured[1:]):
        gap_left = left["task_nll"] - target_nll
        gap_right = right["task_nll"] - target_nll
        if gap_left * gap_right < 0:
            x_left, x_right = left["setting"], right["setting"]
            if log_scale:
                x_left, x_right = math.log(x_left), math.log(x_right)
            x_next = x_left + (x_right - x_left) * gap_left / (gap_left - gap_right)
            return math.exp(x_next) if log_scale else x_next
    return None


def frontier_setting(rows: list[dict], target_nll: float) -> dict | None:
    """The post-hoc frontier match (13 Nov): the measured row with the lowest mean S among those with task NLL
    ≤ target_nll + MATCH_TOLERANCE, or None if no row is that good on the task."""
    eligible = [row for row in rows if "task_nll" in row and row["task_nll"] <= target_nll + MATCH_TOLERANCE]
    if not eligible:
        return None
    return min(eligible, key=lambda row: row["S_mean"])


def baseline_rows(entry) -> list[dict]:
    """A baseline's measured rows from its result entry: a list of rows, or {"unavailable": ..., "rows": [...]}
    for a method that failed part-way (its rows so far)."""
    if isinstance(entry, dict):
        return entry.get("rows", [])
    return entry


def closest_setting(rows: list[dict], target_nll: float) -> dict:
    """The baseline row whose dev task NLL is closest to target_nll; rows that failed (no task_nll) are skipped."""
    measured = [row for row in rows if "task_nll" in row]
    if not measured:
        raise ValueError("the baseline has no measured setting")
    return min(measured, key=lambda row: abs(row["task_nll"] - target_nll))
