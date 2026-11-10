"""Statistics for the pre-registered predictions (CLAUDE.md, "Predictions and pass criteria").

Wilson 95% intervals for rates, exact McNemar on paired prompts, bootstrap intervals (Spearman for
P2/P3, paired differences over prompts for P6), Holm correction across the three languages, and
Cohen's kappa for judge agreement. AuditKIT has none of these, so they live here.

Conventions:
  * ``alpha`` is the two-sided error rate: intervals are central (1 - alpha) intervals.
  * Every bootstrap draws from ``numpy.random.default_rng(seed)``, so a call is reproducible for a
    given (inputs, n_boot, seed). Bootstrap intervals are percentile intervals.
  * Functions return plain Python floats / ints / bools so results can go straight into JSON.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import Any

import numpy as np
from scipy import stats as sps

# Upper bound on elements per bootstrap index chunk, so memory stays bounded for large n.
_CHUNK_ELEMS = 2_000_000


# --------------------------------------------------------------------------------------- helpers

def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


def _check_n_boot(n_boot: int) -> int:
    if int(n_boot) != n_boot or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot}")
    return int(n_boot)


def _as_int(x: Any, name: str) -> int:
    xf = float(x)
    if not xf.is_integer():
        raise ValueError(f"{name} must be an integer, got {x}")
    return int(xf)


def _binary_1d(x: Any, name: str) -> np.ndarray:
    """1-D bool array from bools or 0/1 numbers; raises on anything else."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype == bool:
        return arr
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0/1 or booleans")
    return arr.astype(bool)


def _boot_indices(rng: np.random.Generator, n: int, n_boot: int) -> Iterator[np.ndarray]:
    """Yield (chunk, n) arrays of resampling indices, n_boot rows in total."""
    rows = max(1, _CHUNK_ELEMS // max(n, 1))
    done = 0
    while done < n_boot:
        m = min(rows, n_boot - done)
        yield rng.integers(0, n, size=(m, n))
        done += m


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    if samples.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------- Wilson

def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    With p = successes / n and z = Phi^{-1}(1 - alpha/2), the interval is

        (p + z^2/(2n) -/+ z * sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n),

    without continuity correction (matches ``scipy.stats.binomtest(k, n).proportion_ci(
    confidence_level=1-alpha, method="wilson")``).

    Args:
        successes: number of successes k, an integer with 0 <= k <= n.
        n: number of trials, a non-negative integer.
        alpha: two-sided error rate; the interval has nominal coverage 1 - alpha.

    Returns:
        ``(p_hat, lo, hi)`` as floats. If n == 0 all three are nan. If k == 0, lo is exactly 0.0;
        if k == n, hi is exactly 1.0. Bounds are clipped to [0, 1].

    Raises:
        ValueError: if k or n is not an integer, n < 0, k outside [0, n], or alpha not in (0, 1).
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
    p = k / n
    z2n = z * z / n
    denom = 1.0 + z2n
    center = (p + z2n / 2.0) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z2n / (4.0 * n))
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return float(p), float(lo), float(hi)


# --------------------------------------------------------------------------------------- McNemar

def mcnemar_exact(a_correct: Sequence, b_correct: Sequence) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired binary outcomes.

    Item i contributes to the discordant counts
        b = #{i : a_i = 1 and b_i = 0}   (A right, B wrong)
        c = #{i : a_i = 0 and b_i = 1}   (A wrong, B right).
    Under H0 (equal marginal rates), b | b + c ~ Binomial(b + c, 1/2), and the p-value is
    ``scipy.stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue`` (identical to using
    min(b, c), by symmetry). If b + c == 0 the p-value is 1.0.

    Args:
        a_correct, b_correct: 1-D sequences of equal length with booleans or 0/1 values; element i
            of each is the outcome of method A / method B on the same item i (e.g. the same prompt).

    Returns:
        ``{"b": int, "c": int, "p_value": float, "n": int}`` where n is the number of paired items
        (concordant and discordant).

    Raises:
        ValueError: if inputs are not 1-D, differ in length, or contain values other than 0/1/bool.
    """
    a = _binary_1d(a_correct, "a_correct")
    bb = _binary_1d(b_correct, "b_correct")
    if a.shape != bb.shape:
        raise ValueError(f"a_correct and b_correct differ in length: {a.size} vs {bb.size}")
    b = int(np.count_nonzero(a & ~bb))
    c = int(np.count_nonzero(~a & bb))
    p = 1.0 if b + c == 0 else float(sps.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    return {"b": b, "c": c, "p_value": min(1.0, p), "n": int(a.size)}


# --------------------------------------------------------------------------------------- Spearman

def spearman_bootstrap_ci(x: Sequence[float], y: Sequence[float], n_boot: int = 2000,
                          alpha: float = 0.05, seed: int = 0) -> dict[str, Any]:
    """Spearman rank correlation with a percentile bootstrap interval over pairs.

    The point estimate is ``scipy.stats.spearmanr(x, y)`` (Pearson correlation of average ranks,
    so ties are handled). Each bootstrap replicate resamples the n pairs (x_i, y_i) with
    replacement and recomputes rho the same way. Replicates in which the resampled x or the
    resampled y is constant (rho undefined) are skipped, not counted as failures; the interval is
    the (alpha/2, 1 - alpha/2) quantiles of the remaining replicates.

    Args:
        x, y: 1-D finite numeric sequences of equal length n >= 2.
        n_boot: number of bootstrap replicates drawn (including skipped ones).
        alpha: two-sided error rate of the interval.
        seed: seed for ``numpy.random.default_rng``.

    Returns:
        ``{"rho": float, "lo": float, "hi": float, "n": int, "n_boot_used": int}``. n is the number
        of pairs and n_boot_used the number of non-degenerate replicates. If x or y is constant,
        rho is nan; if no replicate is usable, lo and hi are nan.

    Raises:
        ValueError: on shape mismatch, non-1-D or non-finite input, n < 2, or bad alpha/n_boot.
    """
    alpha = _check_alpha(alpha)
    n_boot = _check_n_boot(n_boot)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.ndim != 1 or ya.ndim != 1 or xa.shape != ya.shape:
        raise ValueError(f"x and y must be 1-D of equal length, got {xa.shape} and {ya.shape}")
    if not (np.isfinite(xa).all() and np.isfinite(ya).all()):
        raise ValueError("x and y must be finite")
    n = xa.size
    if n < 2:
        raise ValueError(f"need at least 2 pairs, got {n}")

    if np.all(xa == xa[0]) or np.all(ya == ya[0]):
        rho = float("nan")
    else:
        rho = float(sps.spearmanr(xa, ya).statistic)

    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    for idx in _boot_indices(rng, n, n_boot):
        xs, ys = xa[idx], ya[idx]
        ok = ~(np.all(xs == xs[:, :1], axis=1) | np.all(ys == ys[:, :1], axis=1))
        if not ok.any():
            continue
        rx = sps.rankdata(xs[ok], axis=1)
        ry = sps.rankdata(ys[ok], axis=1)
        rx -= rx.mean(axis=1, keepdims=True)
        ry -= ry.mean(axis=1, keepdims=True)
        r = (rx * ry).sum(axis=1) / np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
        kept.append(np.clip(r, -1.0, 1.0))
    boots = np.concatenate(kept) if kept else np.empty(0)
    lo, hi = _percentile_ci(boots, alpha)
    return {"rho": rho, "lo": lo, "hi": hi, "n": int(n), "n_boot_used": int(boots.size)}


# --------------------------------------------------------------------------------------- paired bootstrap

def paired_bootstrap_diff(a: Any, b: Any, stat: Callable[[np.ndarray], float] = np.mean,
                          n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap over items for the difference stat(a) - stat(b).

    Item i is row i of ``a`` and row i of ``b`` (e.g. the same prompt under two methods). ``a`` and
    ``b`` are 1-D arrays of shape (items,) or 2-D arrays of shape (items, k), e.g. per-token values
    per prompt; their k may differ but their item counts must match. Each replicate draws item
    indices i_1..i_n with replacement (the same indices for a and b, which keeps the pairing),
    takes rows a[i_1..i_n] and b[i_1..i_n], flattens each, and records
    d* = stat(a_resampled.ravel()) - stat(b_resampled.ravel()).

    The point estimate is d = stat(a.ravel()) - stat(b.ravel()). The interval is the percentile
    interval of the d* values. The p-value is the bootstrap sign p-value

        p_two_sided = min(1, 2 * min(P*(d* <= 0), P*(d* >= 0))),

    where P* is the fraction of replicates; it is small when the bootstrap distribution sits
    clearly on one side of 0, and equals 1 when every d* is 0.

    Args:
        a, b: numeric arrays (1-D or 2-D) with the same length along axis 0, at least 1 item.
        stat: function mapping a 1-D array to a scalar (default ``np.mean``).
        n_boot: number of bootstrap replicates.
        alpha: two-sided error rate of the interval.
        seed: seed for ``numpy.random.default_rng``.

    Returns:
        ``{"diff": float, "lo": float, "hi": float, "p_two_sided": float}``.

    Raises:
        ValueError: if a or b is not 1-D/2-D, item counts differ, there are no items, or
            alpha/n_boot is invalid.
    """
    alpha = _check_alpha(alpha)
    n_boot = _check_n_boot(n_boot)
    aa = np.asarray(a, dtype=float)
    ba = np.asarray(b, dtype=float)
    if aa.ndim not in (1, 2) or ba.ndim not in (1, 2):
        raise ValueError(f"a and b must be 1-D or 2-D, got shapes {aa.shape} and {ba.shape}")
    if aa.shape[0] != ba.shape[0]:
        raise ValueError(f"a and b must have the same number of items, got {aa.shape[0]} and {ba.shape[0]}")
    n = aa.shape[0]
    if n == 0:
        raise ValueError("need at least one item")

    diff = float(stat(aa.ravel()) - stat(ba.ravel()))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    j = 0
    for idx in _boot_indices(rng, n, n_boot):
        for row in idx:
            out[j] = stat(aa[row].ravel()) - stat(ba[row].ravel())
            j += 1
    lo, hi = _percentile_ci(out, alpha)
    p = 2.0 * min(np.mean(out <= 0.0), np.mean(out >= 0.0))
    return {"diff": diff, "lo": lo, "hi": hi, "p_two_sided": float(min(1.0, p))}


# --------------------------------------------------------------------------------------- frontier

def best_at_threshold(points: Sequence[dict], min_safety: float) -> dict | None:
    """Best-utility frontier point subject to a safety floor (e.g. task score at refusal >= 0.97).

    Among ``points`` with ``point["safety"] >= min_safety`` (inclusive; plain float comparison, so a
    rate computed as k/n compares exactly with the same decimal literal) and a finite
    ``point["utility"]``, return the one with the largest utility. Points whose safety is nan never
    qualify. Ties go to the earliest point in the input order.

    Args:
        points: one method's frontier, as dicts with at least the keys "safety" and "utility".
        min_safety: the safety floor.

    Returns:
        The qualifying point (the same dict object that was passed in), or None if none qualifies.
    """
    best = None
    best_u = -np.inf
    for pt in points:
        s, u = float(pt["safety"]), float(pt["utility"])
        if s >= min_safety and np.isfinite(u) and (best is None or u > best_u):
            best, best_u = pt, u
    return best


# --------------------------------------------------------------------------------------- Holm

def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    Sort the m p-values ascending, p_(1) <= ... <= p_(m). The adjusted value of the j-th smallest is
        p~_(j) = min(1, max_{i <= j} (m - i + 1) * p_(i)),
    so adjusted values are monotone in the raw ones and never exceed 1. Ties keep a stable order
    (they get equal adjusted values anyway). Rejecting H_j when p~_j <= alpha is exactly Holm's
    procedure at family-wise level alpha.

    Args:
        p_values: raw p-values in [0, 1] (any length, including 0).

    Returns:
        List of adjusted p-values, element j corresponding to p_values[j].

    Raises:
        ValueError: if any p-value is nan or outside [0, 1].
    """
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        return []
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must be in [0, 1] and not nan")
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = (m - np.arange(m)) * p[order]
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return [float(v) for v in adj]


def holm_reject(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni rejections at family-wise error rate ``alpha``, in the input order.

    Element j is True iff holm(p_values)[j] <= alpha.
    """
    alpha = _check_alpha(alpha)
    return [bool(v <= alpha) for v in holm(p_values)]


# --------------------------------------------------------------------------------------- Cohen's kappa

def _encode_pair(y1: Sequence[Hashable], y2: Sequence[Hashable],
                 labels: Sequence[Hashable] | None) -> tuple[np.ndarray, np.ndarray, int]:
    y1 = list(y1)
    y2 = list(y2)
    if len(y1) != len(y2):
        raise ValueError(f"y1 and y2 differ in length: {len(y1)} vs {len(y2)}")
    if labels is None:
        cats: dict[Hashable, int] = {}
        for v in y1 + y2:
            cats.setdefault(v, len(cats))
    else:
        cats = {}
        for v in labels:
            cats.setdefault(v, len(cats))
        unknown = {v for v in y1 + y2 if v not in cats}
        if unknown:
            raise ValueError(f"values not in labels: {sorted(map(repr, unknown))}")
    c1 = np.fromiter((cats[v] for v in y1), dtype=np.int64, count=len(y1))
    c2 = np.fromiter((cats[v] for v in y2), dtype=np.int64, count=len(y2))
    return c1, c2, len(cats)


def _kappa_from_codes(c1: np.ndarray, c2: np.ndarray, k: int) -> np.ndarray:
    """Row-wise kappa for (R, n) code arrays, in exact integer arithmetic; nan where p_e = 1.

    kappa = (p_o - p_e) / (1 - p_e) = (n * A - S) / (n^2 - S), with A the number of agreements and
    S = sum_k n1_k * n2_k the product of the two raters' marginal counts.
    """
    n = c1.shape[1]
    agree = (c1 == c2).sum(axis=1)
    s = np.zeros(c1.shape[0], dtype=np.int64)
    for lab in range(k):
        s += (c1 == lab).sum(axis=1) * (c2 == lab).sum(axis=1)
    num = n * agree - s
    den = n * n - s
    out = np.full(c1.shape[0], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def cohen_kappa(y1: Sequence[Hashable], y2: Sequence[Hashable],
                labels: Sequence[Hashable] | None = None) -> float:
    """Cohen's kappa (unweighted) for two raters assigning nominal labels to the same items.

    kappa = (p_o - p_e) / (1 - p_e), where p_o is the fraction of items with y1_i == y2_i and
    p_e = sum_k (n1_k / n) * (n2_k / n) is the chance agreement from each rater's marginal label
    frequencies. This equals ``sklearn.metrics.cohen_kappa_score`` (weights=None). It is computed
    in exact integer arithmetic.

    Args:
        y1, y2: equal-length sequences of hashable labels (strings, ints, bools, ...); element i of
            each is the label the rater gave item i.
        labels: optional label set. If given, every observed value must be in it (otherwise
            ValueError; items are never silently dropped). Unused labels do not change kappa.

    Returns:
        kappa as a float in [-1, 1]; nan if there are no items or if p_e == 1 (both raters used one
        and the same label for every item, so kappa is undefined).

    Raises:
        ValueError: on a length mismatch or a value outside ``labels``.
    """
    c1, c2, k = _encode_pair(y1, y2, labels)
    if c1.size == 0:
        return float("nan")
    return float(_kappa_from_codes(c1[None, :], c2[None, :], k)[0])


def cohen_kappa_ci(y1: Sequence[Hashable], y2: Sequence[Hashable], n_boot: int = 2000,
                   alpha: float = 0.05, seed: int = 0) -> dict[str, Any]:
    """Cohen's kappa with a percentile bootstrap interval over items.

    Each replicate resamples the n items (pairs (y1_i, y2_i)) with replacement and recomputes
    ``cohen_kappa``. Replicates where kappa is undefined (p_e == 1) are skipped; the interval is the
    (alpha/2, 1 - alpha/2) quantiles of the remaining replicates.

    Args:
        y1, y2: equal-length sequences of hashable labels.
        n_boot: number of bootstrap replicates drawn (including skipped ones).
        alpha: two-sided error rate of the interval.
        seed: seed for ``numpy.random.default_rng``.

    Returns:
        ``{"kappa": float, "lo": float, "hi": float, "n": int, "n_boot_used": int}``. kappa is nan
        when undefined on the full data; lo and hi are nan when no replicate is usable (or n == 0).

    Raises:
        ValueError: on a length mismatch or bad alpha/n_boot.
    """
    alpha = _check_alpha(alpha)
    n_boot = _check_n_boot(n_boot)
    c1, c2, k = _encode_pair(y1, y2, None)
    n = c1.size
    nan = float("nan")
    if n == 0:
        return {"kappa": nan, "lo": nan, "hi": nan, "n": 0, "n_boot_used": 0}
    kappa = float(_kappa_from_codes(c1[None, :], c2[None, :], k)[0])
    rng = np.random.default_rng(seed)
    kept = []
    for idx in _boot_indices(rng, n, n_boot):
        r = _kappa_from_codes(c1[idx], c2[idx], k)
        kept.append(r[~np.isnan(r)])
    boots = np.concatenate(kept)
    lo, hi = _percentile_ci(boots, alpha)
    return {"kappa": kappa, "lo": lo, "hi": hi, "n": int(n), "n_boot_used": int(boots.size)}
