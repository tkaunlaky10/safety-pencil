"""Generalised eigenproblems F_h x = λ F_t′ x for one factor pair, in float64.

F_t′ is the damped task factor (VERIFICATION.md §3.2):
  covariant  F_t + γ F_h            keeps the eigenvectors and exact reparameterisation invariance (default)
  euclidean  F_t + γ mean(diag) I   a trust region; breaks invariance, so only the rule-11 fallback
  none       F_t                    valid only when null(F_t) ⊆ null(F_h); tests only

`gen_eig` works on range(F_h + F_t): directions null in both factors (e.g. 1/w for a LayerNorm-fed input)
carry no information and would make Cholesky fail. Factors summed in fp32 are only null to ~1e-8 of the top
eigenvalue, so set rel_tol above that floor (`auto_rel_tol`). Mirrors tests/test_math.py::gen_eig_range_np.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DAMPINGS = ("covariant", "euclidean", "none")


@dataclass
class Pencil:
    """One side's eigenpairs. Only X and F_t′X are kept (n × k), never the n × n factors."""

    lam: torch.Tensor  # (k,) eigenvalues, ascending, clamped at 0
    X: torch.Tensor  # (n, k) eigenvectors; X^T F_t' X = I
    FX: torch.Tensor  # (n, k) F_t' X: all the repair needs of the damped task factor (n x k, not n x n)
    range_dim: int  # dimension of range(F_h + F_t)
    dropped: int = 0  # range directions where the damped task factor was not numerically positive

    def to(self, dtype=None, device=None) -> Pencil:
        return Pencil(
            self.lam.to(device=device, dtype=dtype),
            self.X.to(device=device, dtype=dtype),
            self.FX.to(device=device, dtype=dtype),
            self.range_dim,
            self.dropped,
        )


def damp(F_h: torch.Tensor, F_t: torch.Tensor, damping: str, gamma: float) -> torch.Tensor:
    """F_t′ for the given damping mode."""
    if damping == "covariant":
        return F_t + gamma * F_h
    if damping == "euclidean":
        n = F_t.shape[0]
        return F_t + gamma * F_t.diagonal().mean() * torch.eye(n, dtype=F_t.dtype, device=F_t.device)
    if damping == "none":
        return F_t
    raise ValueError(f"damping must be one of {DAMPINGS}, got {damping!r}")


def gen_eig(
    F_h: torch.Tensor,
    F_t: torch.Tensor,
    damping: str = "covariant",
    gamma: float = 1e-3,
    top_k: int | None = None,
    rel_tol: float = 1e-10,
) -> Pencil:
    """Solve F_h x = λ F_t′ x on range(F_h + F_t). Returns ascending eigenpairs (the top_k largest if set)
    with Xᵀ F_t′ X = I, computed in float64 on F_h's device."""
    F_h = _sym(F_h.to(torch.float64))
    F_t = _sym(F_t.to(torch.float64))
    if damping == "covariant" and gamma <= 0:
        raise ValueError("covariant damping needs gamma > 0 (undamped factor pencils can be singular)")
    w, U = torch.linalg.eigh(F_h + F_t)
    keep = w > rel_tol * w.max().clamp_min(torch.finfo(torch.float64).tiny)
    U = U[:, keep]
    F_td = damp(F_h, F_t, damping, gamma)
    Ah = U.T @ F_h @ U
    At = _sym(U.T @ F_td @ U)
    # in exact arithmetic At >= min(1, gamma) * rel_tol * max(w) on the range (covariant damping)
    floor = 0.5 * min(1.0, gamma if damping == "covariant" else 1.0) * rel_tol * float(w.max())
    Li, dropped = _whitener(At, floor)
    lam, V = torch.linalg.eigh(_sym(Li @ Ah @ Li.T))
    X = U @ (Li.T @ V)
    lam = lam.clamp_min(0)
    if top_k is not None and top_k < lam.numel():
        lam, X = lam[-top_k:].clone(), X[:, -top_k:].clone()  # clone: a view would keep the full n x n
    return Pencil(lam=lam, X=X, FX=F_td @ X, range_dim=int(keep.sum()), dropped=dropped)


def gen_eig_topk(
    F_h: torch.Tensor,
    F_t: torch.Tensor,
    k: int,
    damping: str = "covariant",
    gamma: float = 1e-3,
    jitter: float = 1e-10,
    eig_dtype: torch.dtype = torch.float32,
    device: str | torch.device | None = None,
) -> Pencil:
    """Top-k pairs of F_h x = lam F_t' x for a large factor with no exact null space (the MLP-width sides:
    down_proj input, gate/up output). Replaces gen_eig there, whose three n x n float64 eigendecompositions
    cost about a minute each at n = 11008 on an L4.

    Cholesky of F_t' + jitter·mean(diag)·I in float64 (the jitter only guards numerically singular
    directions; these factors are not touched by the V/O gauge), whitening by triangular solves in float64,
    one eigh of the whitened matrix in `eig_dtype` (float32 is accurate for the top pairs, which are all we
    keep), then X = L⁻ᵀ V in float64. X^T F_t' X = I up to eig_dtype rounding."""
    dev = torch.device(device) if device is not None else F_h.device
    F_h = _sym(F_h.to(dev, torch.float64))
    F_t = _sym(F_t.to(dev, torch.float64))
    if damping == "covariant" and gamma <= 0:
        raise ValueError("covariant damping needs gamma > 0")
    F_td = damp(F_h, F_t, damping, gamma)
    n = F_td.shape[0]
    eye = torch.eye(n, dtype=F_td.dtype, device=dev)
    L, info = torch.linalg.cholesky_ex(F_td + jitter * F_td.diagonal().mean() * eye)
    if int(info) != 0:
        raise RuntimeError("F_t' is not positive definite even with jitter; use gen_eig (range restriction)")
    Z = torch.linalg.solve_triangular(L, F_h, upper=False)  # L⁻¹ F_h
    Hw = _sym(torch.linalg.solve_triangular(L, Z.T.contiguous(), upper=False))  # L⁻¹ F_h L⁻ᵀ
    del Z
    lam, V = torch.linalg.eigh(Hw.to(eig_dtype))
    lam, V = lam[-k:].to(torch.float64).clamp_min(0), V[:, -k:].to(torch.float64)
    X = torch.linalg.solve_triangular(L.T.contiguous(), V, upper=True)  # L⁻ᵀ V
    return Pencil(lam=lam.clone(), X=X, FX=F_td @ X, range_dim=n, dropped=0)


def _whitener(At: torch.Tensor, floor: float) -> tuple[torch.Tensor, int]:
    """Li with Li At Liᵀ = I. Cholesky when it succeeds (exact match with the numpy reference); otherwise
    an eigendecomposition that drops directions with eigenvalue <= floor (numerically non-positive)."""
    L, info = torch.linalg.cholesky_ex(At)
    if int(info) == 0:
        eye = torch.eye(L.shape[0], dtype=L.dtype, device=L.device)
        return torch.linalg.solve_triangular(L, eye, upper=False), 0
    s, Q = torch.linalg.eigh(At)
    ok = s > floor
    return (Q[:, ok] / s[ok].sqrt()).T, int((~ok).sum())


def noise_floor(F: torch.Tensor, null_vec: torch.Tensor) -> float:
    """Rayleigh quotient of a vector that is null in exact arithmetic, over F's top eigenvalue: the factor's
    numerical floor."""
    F = _sym(F.to(torch.float64))
    u = null_vec.to(F) / null_vec.norm()
    top = torch.linalg.eigvalsh(F).max()
    return float((u @ F @ u).abs() / top) if top > 0 else 0.0


def auto_rel_tol(floors, base: float = 1e-10, factor: float = 10.0) -> float:
    """max(base, factor × the largest measured floor)."""
    floors = list(floors)
    return max(base, factor * max(floors)) if floors else base


def _sym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)
