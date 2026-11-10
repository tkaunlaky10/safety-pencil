"""Statistics for the pre-registered predictions (CLAUDE.md, "Predictions and pass criteria"); AuditKIT has none.

Wilson intervals for rates, exact McNemar on paired prompts, bootstrap intervals (Spearman for P2/P3, paired
differences over prompts for P6), Holm correction across languages, and Cohen's kappa for judge agreement.

``alpha`` is the two-sided error rate. Bootstraps draw from ``numpy.random.default_rng(seed)`` and give
percentile intervals, so a call is reproducible. Results are plain Python numbers, ready for JSON.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import Any

import numpy as np
from scipy import stats as sps

_CHUNK_ELEMS = 2_000_000  # bootstrap indices drawn per chunk, so memory stays bounded for large n


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
    """Resampling indices as (chunk, n) arrays, n_boot rows in total."""
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
    """(p_hat, lo, hi): the Wilson score interval without continuity correction, equal to scipy's
    ``binomtest(k, n).proportion_ci(method="wilson")``. All nan when n == 0; lo is exactly 0 when k == 0
    and hi exactly 1 when k == n."""
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
    """Exact two-sided McNemar test on paired 0/1 outcomes of methods A and B on the same items.

    b counts items A got right and B wrong, c the reverse; the p-value is the two-sided binomial test of
    b out of b + c at 1/2 (1.0 when b + c == 0). Returns {"b", "c", "p_value", "n"}, n = all items.
    """
    a = _binary_1d(a_correct, "a_correct")
    bb = _binary_1d(b_correct, "b_correct")
    if a.shape != bb.shape:
        raise ValueError(f"a_correct and b_correct differ in length: {a.size} vs {bb.size}")
    b = int(np.count_nonzero(a & ~bb))
    c = int(np.count_nonzero(~a & bb))
    p = 1.0 if b + c == 0 else float(sps.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    return {"b": b, "c": c, "p_value": p, "n": int(a.size)}


# --------------------------------------------------------------------------------------- Spearman


def _rowwise_spearman(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Spearman rho of each row of two (R, n) arrays: the Pearson correlation of average ranks."""
    rx = sps.rankdata(xs, axis=1)
    ry = sps.rankdata(ys, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    r = (rx * ry).sum(axis=1) / np.sqrt((rx * rx).sum(axis=1) * (ry * ry).sum(axis=1))
    return np.clip(r, -1.0, 1.0)


def spearman_bootstrap_ci(
    x: Sequence[float], y: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, Any]:
    """Spearman rho (``scipy.stats.spearmanr``, ties averaged) with a percentile bootstrap interval over pairs.

    Resamples in which x or y is constant (rho undefined) are skipped. Returns {"rho", "lo", "hi", "n",
    "n_boot_used"}; rho is nan for constant input, lo and hi are nan when no resample is usable.
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

    constant = np.all(xa == xa[0]) or np.all(ya == ya[0])
    rho = float("nan") if constant else float(sps.spearmanr(xa, ya).statistic)
    kept: list[np.ndarray] = []
    for idx in _boot_indices(np.random.default_rng(seed), n, n_boot):
        xs, ys = xa[idx], ya[idx]
        ok = ~(np.all(xs == xs[:, :1], axis=1) | np.all(ys == ys[:, :1], axis=1))
        if ok.any():
            kept.append(_rowwise_spearman(xs[ok], ys[ok]))
    boots = np.concatenate(kept) if kept else np.empty(0)
    lo, hi = _percentile_ci(boots, alpha)
    return {"rho": rho, "lo": lo, "hi": hi, "n": int(n), "n_boot_used": int(boots.size)}


# --------------------------------------------------------------------------------------- paired bootstrap


def paired_bootstrap_diff(
    a: Any,
    b: Any,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap over items for stat(a) - stat(b), e.g. the same prompts under two methods.

    Row i of ``a`` and of ``b`` is item i; rows may be scalars (1-D) or vectors of any length (2-D, e.g.
    per-token values). Each replicate resamples item indices, shared by a and b to keep the pairing, and
    applies ``stat`` to the flattened rows. Returns {"diff", "lo", "hi", "p_two_sided"}, where the p-value
    is the bootstrap sign test min(1, 2 * min(P*(d* <= 0), P*(d* >= 0))).
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
    out = np.array(
        [stat(aa[row].ravel()) - stat(ba[row].ravel()) for idx in _boot_indices(rng, n, n_boot) for row in idx]
    )
    lo, hi = _percentile_ci(out, alpha)
    p = 2.0 * min(np.mean(out <= 0.0), np.mean(out >= 0.0))
    return {"diff": diff, "lo": lo, "hi": hi, "p_two_sided": float(min(1.0, p))}


# --------------------------------------------------------------------------------------- frontier


def best_at_threshold(points: Sequence[dict], min_safety: float) -> dict | None:
    """The highest-utility point (the same dict object) with ``safety >= min_safety`` and a finite
    ``utility``, e.g. the best task score at refusal >= 0.97; the earliest wins ties, nan safety never
    qualifies, and None if nothing qualifies."""
    eligible = [pt for pt in points if float(pt["safety"]) >= min_safety and np.isfinite(float(pt["utility"]))]
    return max(eligible, key=lambda pt: float(pt["utility"]), default=None)


# --------------------------------------------------------------------------------------- Holm


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order: the j-th smallest becomes
    min(1, max_{i <= j} (m - i + 1) p_(i)). Rejecting where the adjusted value <= alpha is Holm's procedure."""
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        return []
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must be in [0, 1] and not nan")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    adj[order] = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    return [float(v) for v in adj]


def holm_reject(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni rejections at family-wise error rate ``alpha``, in the input order."""
    alpha = _check_alpha(alpha)
    return [bool(v <= alpha) for v in holm(p_values)]


# --------------------------------------------------------------------------------------- Cohen's kappa


def _encode_pair(
    y1: Sequence[Hashable], y2: Sequence[Hashable], labels: Sequence[Hashable] | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Both raters' labels as integer codes, and the number of categories."""
    y1, y2 = list(y1), list(y2)
    if len(y1) != len(y2):
        raise ValueError(f"y1 and y2 differ in length: {len(y1)} vs {len(y2)}")
    cats: dict[Hashable, int] = {}
    for v in y1 + y2 if labels is None else labels:
        cats.setdefault(v, len(cats))
    unknown = {v for v in y1 + y2 if v not in cats}
    if unknown:
        raise ValueError(f"values not in labels: {sorted(map(repr, unknown))}")
    c1 = np.fromiter((cats[v] for v in y1), dtype=np.int64, count=len(y1))
    c2 = np.fromiter((cats[v] for v in y2), dtype=np.int64, count=len(y2))
    return c1, c2, len(cats)


def _kappa_from_codes(c1: np.ndarray, c2: np.ndarray, k: int) -> np.ndarray:
    """Row-wise kappa of (R, n) code arrays in exact integer arithmetic; nan where chance agreement is 1.

    kappa = (p_o - p_e) / (1 - p_e) = (n A - S) / (n^2 - S), with A the number of agreements and
    S = sum_k n1_k n2_k over the two raters' marginal counts.
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


def cohen_kappa(y1: Sequence[Hashable], y2: Sequence[Hashable], labels: Sequence[Hashable] | None = None) -> float:
    """Unweighted Cohen's kappa of two raters' nominal labels for the same items, equal to
    ``sklearn.metrics.cohen_kappa_score``. With ``labels``, a value outside them raises rather than being
    dropped. nan with no items, or when both raters gave every item one and the same label."""
    c1, c2, k = _encode_pair(y1, y2, labels)
    if c1.size == 0:
        return float("nan")
    return float(_kappa_from_codes(c1[None, :], c2[None, :], k)[0])


def cohen_kappa_ci(
    y1: Sequence[Hashable], y2: Sequence[Hashable], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, Any]:
    """Cohen's kappa with a percentile bootstrap interval over items; resamples where kappa is undefined are
    skipped. Returns {"kappa", "lo", "hi", "n", "n_boot_used"}."""
    alpha = _check_alpha(alpha)
    n_boot = _check_n_boot(n_boot)
    c1, c2, k = _encode_pair(y1, y2, None)
    n = c1.size
    if n == 0:
        nan = float("nan")
        return {"kappa": nan, "lo": nan, "hi": nan, "n": 0, "n_boot_used": 0}
    kappa = float(_kappa_from_codes(c1[None, :], c2[None, :], k)[0])
    kept = []
    for idx in _boot_indices(np.random.default_rng(seed), n, n_boot):
        r = _kappa_from_codes(c1[idx], c2[idx], k)
        kept.append(r[~np.isnan(r)])
    boots = np.concatenate(kept)
    lo, hi = _percentile_ci(boots, alpha)
    return {"kappa": kappa, "lo": lo, "hi": hi, "n": int(n), "n_boot_used": int(boots.size)}
