"""Statistics for the pre-registered predictions (CLAUDE.md, "Predictions and pass criteria"); AuditKIT has none.

- Wilson score intervals for rates (refusal, unsafe, over-refusal, MMLU accuracy);
- the exact McNemar test for two checkpoints scored on the same prompts;
- percentile bootstrap intervals for Spearman's ρ (P2, P3), for paired differences over prompts (P6) and for κ;
- Holm's correction across the three languages, and Cohen's κ for judge-versus-human agreement.

``alpha`` is the two-sided error rate. Bootstraps draw from ``numpy.random.default_rng(seed)``, so a call is
reproducible. Results are plain Python numbers, ready for JSON.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import Any

import numpy as np
from scipy import stats as sps

# Bootstrap indices are drawn in chunks of about this many elements, so memory stays bounded for large n.
_CHUNK_ELEMENTS = 2_000_000


# --------------------------------------------------------------------------------------- helpers


def _check_alpha(alpha: float) -> float:
    """alpha as a float, checked to lie strictly between 0 and 1."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


def _check_n_boot(n_boot: int) -> int:
    """n_boot as an int, checked to be a positive whole number."""
    if int(n_boot) != n_boot or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot}")
    return int(n_boot)


def _as_int(value: Any, name: str) -> int:
    """value as an int; raises if it is not a whole number (so 2.0 is accepted and 2.5 is not)."""
    as_float = float(value)
    if not as_float.is_integer():
        raise ValueError(f"{name} must be an integer, got {value}")
    return int(as_float)


def _binary_1d(values: Any, name: str) -> np.ndarray:
    """A 1-D bool array from bools or 0/1 numbers; raises on anything else."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.dtype == bool:
        return array
    if array.size and not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 or booleans")
    return array.astype(bool)


def _boot_indices(rng: np.random.Generator, n: int, n_boot: int) -> Iterator[np.ndarray]:
    """Resampling indices (with replacement) as (chunk, n) arrays, n_boot rows in total."""
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // max(n, 1))
    rows_done = 0
    while rows_done < n_boot:
        chunk_rows = min(rows_per_chunk, n_boot - rows_done)
        yield rng.integers(0, n, size=(chunk_rows, n))
        rows_done += chunk_rows


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    """The α/2 and 1 − α/2 quantiles of the bootstrap samples (nan, nan when there are none)."""
    if samples.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------- Wilson


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """A rate k/n with its Wilson score interval, returned as (p̂, lo, hi).

    p̂ = k/n and z = Φ⁻¹(1 − α/2); the interval is centre ± half-width with
    centre = (p̂ + z²/2n) / (1 + z²/n) and half-width = z / (1 + z²/n) · √(p̂(1 − p̂)/n + z²/4n²).
    No continuity correction, so it equals scipy's ``binomtest(k, n).proportion_ci(method="wilson")``.
    All nan when n == 0; lo is exactly 0 when k == 0 and hi exactly 1 when k == n.
    """
    alpha = _check_alpha(alpha)
    k = _as_int(successes, "successes")
    n = _as_int(n, "n")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"successes must be in [0, {n}], got {k}")
    if n == 0:
        nan = float("nan")
        return nan, nan, nan

    z = float(sps.norm.ppf(1.0 - alpha / 2.0))
    p_hat = k / n
    z_squared_over_n = z * z / n
    denominator = 1.0 + z_squared_over_n
    centre = (p_hat + z_squared_over_n / 2.0) / denominator
    half_width = (z / denominator) * np.sqrt(p_hat * (1.0 - p_hat) / n + z_squared_over_n / (4.0 * n))

    # Exact 0 and 1 at the edges, where floating point could otherwise leave a tiny gap.
    lo = 0.0 if k == 0 else max(0.0, centre - half_width)
    hi = 1.0 if k == n else min(1.0, centre + half_width)
    return float(p_hat), float(lo), float(hi)


# --------------------------------------------------------------------------------------- McNemar


def mcnemar_exact(a_correct: Sequence, b_correct: Sequence) -> dict[str, Any]:
    """Exact two-sided McNemar test: do methods A and B differ on the same items (paired 0/1 outcomes)?

    b counts the items A got right and B wrong, c the reverse. Under the null each discordant item is a fair
    coin, so the p-value is the two-sided exact binomial test of b successes in b + c trials at ½
    (``scipy.stats.binomtest``), and 1.0 when b + c == 0. Returns {"b", "c", "p_value", "n"}, n = all items.
    """
    a_right = _binary_1d(a_correct, "a_correct")
    b_right = _binary_1d(b_correct, "b_correct")
    if a_right.shape != b_right.shape:
        raise ValueError(f"a_correct and b_correct differ in length: {a_right.size} vs {b_right.size}")
    b = int(np.count_nonzero(a_right & ~b_right))
    c = int(np.count_nonzero(~a_right & b_right))
    return {"b": b, "c": c, "p_value": _mcnemar_p_value(b, c), "n": int(a_right.size)}


def _mcnemar_p_value(b: int, c: int) -> float:
    """Two-sided exact binomial p-value of b successes in b + c fair-coin trials; 1.0 when b + c == 0."""
    if b + c == 0:
        return 1.0
    return float(sps.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


# --------------------------------------------------------------------------------------- Spearman


def _rowwise_spearman(x_rows: np.ndarray, y_rows: np.ndarray) -> np.ndarray:
    """Spearman's ρ of each row of two (R, n) arrays: the Pearson correlation of the average ranks."""
    rank_x = sps.rankdata(x_rows, axis=1)
    rank_y = sps.rankdata(y_rows, axis=1)
    rank_x -= rank_x.mean(axis=1, keepdims=True)
    rank_y -= rank_y.mean(axis=1, keepdims=True)
    covariance = (rank_x * rank_y).sum(axis=1)
    rho = covariance / np.sqrt((rank_x * rank_x).sum(axis=1) * (rank_y * rank_y).sum(axis=1))
    return np.clip(rho, -1.0, 1.0)


def _constant_rows(rows: np.ndarray) -> np.ndarray:
    """True for each row of a 2-D array whose entries are all equal."""
    return np.all(rows == rows[:, :1], axis=1)


def spearman_bootstrap_ci(
    x: Sequence[float], y: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, Any]:
    """Spearman's rank correlation ρ of x and y, with a percentile bootstrap interval over the (x, y) pairs.

    ρ is the Pearson correlation of the ranks, ties averaged (``scipy.stats.spearmanr``). Resamples in which x
    or y is constant (ρ undefined) are skipped. Returns {"rho", "lo", "hi", "n", "n_boot_used"}; rho is nan
    for constant input, and lo and hi are nan when no resample is usable.
    """
    alpha = _check_alpha(alpha)
    n_boot = _check_n_boot(n_boot)
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or x_values.shape != y_values.shape:
        raise ValueError(f"x and y must be 1-D of equal length, got {x_values.shape} and {y_values.shape}")
    if not (np.isfinite(x_values).all() and np.isfinite(y_values).all()):
        raise ValueError("x and y must be finite")
    n = x_values.size
    if n < 2:
        raise ValueError(f"need at least 2 pairs, got {n}")

    is_constant = np.all(x_values == x_values[0]) or np.all(y_values == y_values[0])
    rho = float("nan") if is_constant else float(sps.spearmanr(x_values, y_values).statistic)

    rho_chunks: list[np.ndarray] = []
    for indices in _boot_indices(np.random.default_rng(seed), n, n_boot):
        x_boot = x_values[indices]
        y_boot = y_values[indices]
        usable = ~(_constant_rows(x_boot) | _constant_rows(y_boot))
        if usable.any():
            rho_chunks.append(_rowwise_spearman(x_boot[usable], y_boot[usable]))
    boot_rhos = np.concatenate(rho_chunks) if rho_chunks else np.empty(0)
    lo, hi = _percentile_ci(boot_rhos, alpha)
    return {"rho": rho, "lo": lo, "hi": hi, "n": int(n), "n_boot_used": int(boot_rhos.size)}


# --------------------------------------------------------------------------------------- paired bootstrap


def paired_bootstrap_diff(
    a: Any,
    b: Any,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """How much stat(a) exceeds stat(b) on the same items (e.g. two methods on the same prompts), with a
    paired percentile bootstrap interval.

    Row i of ``a`` and of ``b`` is item i; rows may be scalars (1-D) or vectors of any length (2-D, e.g.
    per-token values). Each replicate resamples item indices, shared by a and b to keep the pairing, and
    computes d* = stat(a*) − stat(b*) on the flattened rows. Returns {"diff", "lo", "hi", "p_two_sided"},
    where the p-value is the bootstrap sign test min(1, 2 · min(P*(d* ≤ 0), P*(d* ≥ 0))).
    """
    alpha = _check_alpha(alpha)
    n_boot = _check_n_boot(n_boot)
    a_values = np.asarray(a, dtype=float)
    b_values = np.asarray(b, dtype=float)
    if a_values.ndim not in (1, 2) or b_values.ndim not in (1, 2):
        raise ValueError(f"a and b must be 1-D or 2-D, got shapes {a_values.shape} and {b_values.shape}")
    if a_values.shape[0] != b_values.shape[0]:
        raise ValueError(f"a and b must have the same number of items, got {a_values.shape[0]} and {b_values.shape[0]}")
    n = a_values.shape[0]
    if n == 0:
        raise ValueError("need at least one item")

    diff = float(stat(a_values.ravel()) - stat(b_values.ravel()))
    rng = np.random.default_rng(seed)
    replicate_diffs = []
    for chunk in _boot_indices(rng, n, n_boot):
        for resample in chunk:
            replicate_diffs.append(stat(a_values[resample].ravel()) - stat(b_values[resample].ravel()))
    replicate_diffs = np.array(replicate_diffs)

    lo, hi = _percentile_ci(replicate_diffs, alpha)
    p_two_sided = 2.0 * min(np.mean(replicate_diffs <= 0.0), np.mean(replicate_diffs >= 0.0))
    return {"diff": diff, "lo": lo, "hi": hi, "p_two_sided": float(min(1.0, p_two_sided))}


def paired_bootstrap_token_mean_diff(
    a_sums: Any, b_sums: Any, counts: Any, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """How much a token-weighted mean Σa/Σn exceeds Σb/Σn on the same items, with a paired percentile bootstrap.

    Item i has per-token sums a_sums[i] and b_sums[i] over the same counts[i] tokens (e.g. one prompt's summed KL
    under two models: the measured S is Σ kl_sum / Σ n). Each replicate resamples item indices, shared by a and
    b, and computes d* = (Σa* − Σb*) / Σn*. Returns {"diff", "lo", "hi", "p_two_sided"} (bootstrap sign test)."""
    alpha = _check_alpha(alpha)
    n_boot = _check_n_boot(n_boot)
    a_values = np.asarray(a_sums, dtype=float)
    b_values = np.asarray(b_sums, dtype=float)
    n_values = np.asarray(counts, dtype=float)
    if not (a_values.ndim == b_values.ndim == n_values.ndim == 1):
        raise ValueError("a_sums, b_sums and counts must be 1-D")
    if not (len(a_values) == len(b_values) == len(n_values)) or len(a_values) == 0:
        raise ValueError("a_sums, b_sums and counts need the same, non-zero number of items")
    if np.any(n_values <= 0):
        raise ValueError("every item needs a positive token count")

    differences = a_values - b_values
    diff = float(differences.sum() / n_values.sum())
    rng = np.random.default_rng(seed)
    replicate_diffs = []
    for chunk in _boot_indices(rng, len(differences), n_boot):
        replicate_diffs.extend(differences[chunk].sum(axis=1) / n_values[chunk].sum(axis=1))
    replicate_diffs = np.array(replicate_diffs)

    lo, hi = _percentile_ci(replicate_diffs, alpha)
    p_two_sided = 2.0 * min(np.mean(replicate_diffs <= 0.0), np.mean(replicate_diffs >= 0.0))
    return {"diff": diff, "lo": lo, "hi": hi, "p_two_sided": float(min(1.0, p_two_sided))}


# --------------------------------------------------------------------------------------- frontier


def best_at_threshold(points: Sequence[dict], min_safety: float) -> dict | None:
    """The highest-utility point with ``safety >= min_safety``, e.g. the best task score at refusal ≥ 0.97.

    Returns the same dict object, or None if nothing qualifies. Points with a non-finite utility or a nan
    safety never qualify; the earliest point wins ties.
    """
    eligible = []
    for point in points:
        safe_enough = float(point["safety"]) >= min_safety
        if safe_enough and np.isfinite(float(point["utility"])):
            eligible.append(point)
    return max(eligible, key=lambda point: float(point["utility"]), default=None)


# --------------------------------------------------------------------------------------- Holm


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    With m p-values sorted as p_(1) ≤ … ≤ p_(m), the j-th smallest becomes min(1, max_{i ≤ j} (m − i + 1) p_(i))
    (Holm 1979). Rejecting where the adjusted value ≤ alpha is Holm's procedure.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        return []
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must be in [0, 1] and not nan")
    m = p.size
    order = np.argsort(p, kind="stable")
    multipliers = m - np.arange(m)  # m, m - 1, ..., 1 for the sorted p-values
    running_max = np.maximum.accumulate(multipliers * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(1.0, running_max)
    return [float(value) for value in adjusted]


def holm_reject(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Which hypotheses Holm's procedure rejects at family-wise error rate ``alpha``, in the input order.

    Hypothesis i is rejected when its Holm-adjusted p-value (see ``holm``) is ≤ alpha.
    """
    alpha = _check_alpha(alpha)
    return [bool(value <= alpha) for value in holm(p_values)]


# --------------------------------------------------------------------------------------- Cohen's kappa


def _encode_pair(
    y1: Sequence[Hashable], y2: Sequence[Hashable], labels: Sequence[Hashable] | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Both raters' labels as integer codes, and the number of categories.

    Categories are numbered in order of first appearance in ``labels`` (or in y1 then y2 when labels is None).
    """
    y1, y2 = list(y1), list(y2)
    if len(y1) != len(y2):
        raise ValueError(f"y1 and y2 differ in length: {len(y1)} vs {len(y2)}")
    categories = y1 + y2 if labels is None else labels
    code_of: dict[Hashable, int] = {}
    for label in categories:
        code_of.setdefault(label, len(code_of))
    unknown = {label for label in y1 + y2 if label not in code_of}
    if unknown:
        raise ValueError(f"values not in labels: {sorted(repr(label) for label in unknown)}")
    codes_1 = np.fromiter((code_of[label] for label in y1), dtype=np.int64, count=len(y1))
    codes_2 = np.fromiter((code_of[label] for label in y2), dtype=np.int64, count=len(y2))
    return codes_1, codes_2, len(code_of)


def _kappa_from_codes(codes_1: np.ndarray, codes_2: np.ndarray, n_categories: int) -> np.ndarray:
    """Row-wise Cohen's κ of two (R, n) code arrays in exact integer arithmetic; nan where chance agreement is 1.

    κ = (p_o − p_e) / (1 − p_e) = (n·A − S) / (n² − S), with A the number of agreements and
    S = Σ_k n1_k · n2_k over the two raters' per-category counts.
    """
    n = codes_1.shape[1]
    agreements = (codes_1 == codes_2).sum(axis=1)
    marginal_products = np.zeros(codes_1.shape[0], dtype=np.int64)
    for category in range(n_categories):
        marginal_products += (codes_1 == category).sum(axis=1) * (codes_2 == category).sum(axis=1)
    numerator = n * agreements - marginal_products
    denominator = n * n - marginal_products
    kappa = np.full(codes_1.shape[0], np.nan)
    defined = denominator > 0
    kappa[defined] = numerator[defined] / denominator[defined]
    return kappa


def cohen_kappa(y1: Sequence[Hashable], y2: Sequence[Hashable], labels: Sequence[Hashable] | None = None) -> float:
    """Unweighted Cohen's κ: how much two raters agree on nominal labels for the same items, beyond chance.

    κ = (p_o − p_e) / (1 − p_e), p_o the observed and p_e the chance agreement (Cohen 1960); equal to
    ``sklearn.metrics.cohen_kappa_score``. With ``labels``, a value outside them raises rather than being
    dropped. nan with no items, or when both raters gave every item one and the same label.
    """
    codes_1, codes_2, n_categories = _encode_pair(y1, y2, labels)
    if codes_1.size == 0:
        return float("nan")
    return float(_kappa_from_codes(codes_1[None, :], codes_2[None, :], n_categories)[0])


def cohen_kappa_ci(
    y1: Sequence[Hashable], y2: Sequence[Hashable], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, Any]:
    """Cohen's κ (see ``cohen_kappa``) with a percentile bootstrap interval over items.

    Resamples where κ is undefined are skipped. Returns {"kappa", "lo", "hi", "n", "n_boot_used"}.
    """
    alpha = _check_alpha(alpha)
    n_boot = _check_n_boot(n_boot)
    codes_1, codes_2, n_categories = _encode_pair(y1, y2, None)
    n = codes_1.size
    if n == 0:
        nan = float("nan")
        return {"kappa": nan, "lo": nan, "hi": nan, "n": 0, "n_boot_used": 0}
    kappa = float(_kappa_from_codes(codes_1[None, :], codes_2[None, :], n_categories)[0])
    kappa_chunks = []
    for indices in _boot_indices(np.random.default_rng(seed), n, n_boot):
        chunk_kappas = _kappa_from_codes(codes_1[indices], codes_2[indices], n_categories)
        kappa_chunks.append(chunk_kappas[~np.isnan(chunk_kappas)])
    boot_kappas = np.concatenate(kappa_chunks)
    lo, hi = _percentile_ci(boot_kappas, alpha)
    return {"kappa": kappa, "lo": lo, "hi": hi, "n": int(n), "n_boot_used": int(boot_kappas.size)}
