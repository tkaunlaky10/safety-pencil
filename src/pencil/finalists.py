"""The fixed rule that picks the finalists of the test-tier comparison, from dev results only.

μ* is the smallest μ of the sweep that keeps at least KEEP_TASK_GAIN of the fine-tune's task gain on dev:

    NLL(μ*) ≤ NLL(θ1) + (1 − KEEP_TASK_GAIN) · (NLL(θ0) − NLL(θ1))

(smaller μ repairs harder, so this is the strongest repair that still keeps the task). μ whose trust-region check
failed (rule 11: "do not evaluate this point") are not candidates. Each compared baseline's finalist is its
setting whose dev task NLL is closest to NLL(μ*), so every method is compared at about the same task score.
Fixed on 12 Nov before any finalist was evaluated (PREREGISTRATION.md deviation log).
"""

from __future__ import annotations

KEEP_TASK_GAIN = 0.9


def task_nll_limit(nll_theta0: float, nll_theta1: float) -> float:
    """The largest dev task NLL that still keeps KEEP_TASK_GAIN of the gain NLL(θ0) − NLL(θ1)."""
    return nll_theta1 + (1 - KEEP_TASK_GAIN) * (nll_theta0 - nll_theta1)


def accepted(point: dict) -> bool:
    """A sweep point that was measured and passed rule 11's trust-region check."""
    return "task_nll" in point and not point.get("trust_region_failed", False)


def pencil_mu_star(points: list[dict], nll_theta0: float, nll_theta1: float) -> dict:
    """{"mu", "task_nll", "limit"}: the smallest μ of the sweep within the task NLL limit."""
    limit = task_nll_limit(nll_theta0, nll_theta1)
    candidates = []
    for point in points:
        if accepted(point) and point["task_nll"] <= limit:
            candidates.append(point)
    if not candidates:
        raise ValueError(f"no μ keeps {KEEP_TASK_GAIN:.0%} of the task gain (dev task NLL ≤ {limit:.4f})")
    best = min(candidates, key=lambda point: point["mu"])
    return {"mu": best["mu"], "task_nll": best["task_nll"], "limit": limit}


def closest_setting(rows: list[dict], target_nll: float) -> dict:
    """The baseline row whose dev task NLL is closest to target_nll; rows that failed (no task_nll) are skipped."""
    measured = [row for row in rows if "task_nll" in row]
    if not measured:
        raise ValueError("the baseline has no measured setting")
    return min(measured, key=lambda row: abs(row["task_nll"] - target_nll))
