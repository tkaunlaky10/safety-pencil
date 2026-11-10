"""Generalised eigenproblems F_h x = λ F_t′ x for one Kronecker factor pair, in float64.

F stands for either side of a layer: the input factors (A_h, A_t, eigenvalues α) or the output factors
(G_h, G_t, eigenvalues γ). F_t′ is the damped task factor (VERIFICATION.md §3.2):

    covariant  F_t′ = F_t + γ_c F_h            keeps the eigenvectors and exact reparameterisation invariance
    euclidean  F_t′ = F_t + γ_e mean(diag) I   a trust region; breaks invariance, so only the rule-11 fallback
    none       F_t′ = F_t                      valid only when null(F_t) ⊆ null(F_h); tests only

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
    """One side's eigenpairs of F_h x = λ F_t′ x. Only X and F_t′X are kept (n × k), never the n × n factors."""

    lam: torch.Tensor  # (k,) eigenvalues, ascending, clamped at 0
    X: torch.Tensor  # (n, k) eigenvectors, normalised so that Xᵀ F_t′ X = I
    FX: torch.Tensor  # (n, k) F_t′ X: all the repair needs of the damped task factor
    range_dim: int  # dimension of range(F_h + F_t)
    dropped: int = 0  # range directions where the damped task factor was not numerically positive

    def to(self, dtype=None, device=None) -> Pencil:
        """A copy with the tensors moved to `dtype` and `device`."""
        return Pencil(
            lam=self.lam.to(device=device, dtype=dtype),
            X=self.X.to(device=device, dtype=dtype),
            FX=self.FX.to(device=device, dtype=dtype),
            range_dim=self.range_dim,
            dropped=self.dropped,
        )


def damp(F_h: torch.Tensor, F_t: torch.Tensor, damping: str, gamma: float) -> torch.Tensor:
    """The damped task factor F_t′ for one of the DAMPINGS (see the module docstring)."""
    if damping == "covariant":
        return F_t + gamma * F_h
    if damping == "euclidean":
        size = F_t.shape[0]
        identity = torch.eye(size, dtype=F_t.dtype, device=F_t.device)
        return F_t + gamma * F_t.diagonal().mean() * identity
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
    """Solve F_h x = λ F_t′ x on range(F_h + F_t), in float64 on F_h's device.

    Returns ascending eigenpairs (only the top_k largest if top_k is set) with Xᵀ F_t′ X = I. Steps: restrict
    both factors to U = the eigenvectors of F_h + F_t above rel_tol × max; whiten the restricted F_t′ with
    L⁻¹ (L Lᵀ = Uᵀ F_t′ U); take the ordinary eigenpairs (λ, V) of L⁻¹ Uᵀ F_h U L⁻ᵀ; map back X = U L⁻ᵀ V.
    """
    F_h = _symmetrize(F_h.to(torch.float64))
    F_t = _symmetrize(F_t.to(torch.float64))
    if damping == "covariant" and gamma <= 0:
        raise ValueError("covariant damping needs gamma > 0 (undamped factor pencils can be singular)")

    U, top_eigenvalue = _range_basis(F_h + F_t, rel_tol)
    F_t_damped = damp(F_h, F_t, damping, gamma)
    F_h_on_range = U.T @ F_h @ U
    F_t_damped_on_range = _symmetrize(U.T @ F_t_damped @ U)

    floor = _positivity_floor(damping, gamma, rel_tol, top_eigenvalue)
    whitener, dropped = _whitener(F_t_damped_on_range, floor)
    lam, V = torch.linalg.eigh(_symmetrize(whitener @ F_h_on_range @ whitener.T))
    X = U @ (whitener.T @ V)
    lam = lam.clamp_min(0)

    if top_k is not None and top_k < lam.numel():
        # eigh is ascending, so the top k are the last columns. Clone: a view would keep the full n × n alive.
        lam = lam[-top_k:].clone()
        X = X[:, -top_k:].clone()
    return Pencil(lam=lam, X=X, FX=F_t_damped @ X, range_dim=U.shape[1], dropped=dropped)


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
    """Top-k pairs of F_h x = λ F_t′ x for a large factor with no exact null space.

    Used on the MLP-width sides (down_proj input, gate/up output) instead of gen_eig, whose three n × n
    float64 eigendecompositions cost about a minute each at n = 11008 on an L4. Steps:
      1. Cholesky L Lᵀ = F_t′ + jitter · mean(diag) · I in float64. The jitter only guards numerically
         singular directions; these factors are not touched by the V/O gauge.
      2. Whiten by triangular solves in float64: L⁻¹ F_h L⁻ᵀ.
      3. One eigh of the whitened matrix in `eig_dtype` (float32 is accurate for the top pairs, which are
         all we keep).
      4. X = L⁻ᵀ V in float64, so Xᵀ F_t′ X = I up to eig_dtype rounding.
    """
    device = F_h.device if device is None else torch.device(device)
    F_h = _symmetrize(F_h.to(device, torch.float64))
    F_t = _symmetrize(F_t.to(device, torch.float64))
    if damping == "covariant" and gamma <= 0:
        raise ValueError("covariant damping needs gamma > 0")

    F_t_damped = damp(F_h, F_t, damping, gamma)
    size = F_t_damped.shape[0]
    identity = torch.eye(size, dtype=F_t_damped.dtype, device=device)
    jittered = F_t_damped + jitter * F_t_damped.diagonal().mean() * identity
    L, info = torch.linalg.cholesky_ex(jittered)
    if int(info) != 0:
        raise RuntimeError("F_t' is not positive definite even with jitter; use gen_eig (range restriction)")

    half_whitened = torch.linalg.solve_triangular(L, F_h, upper=False)  # L⁻¹ F_h
    whitened = torch.linalg.solve_triangular(L, half_whitened.T.contiguous(), upper=False)  # L⁻¹ F_h L⁻ᵀ
    whitened = _symmetrize(whitened)
    del half_whitened  # n × n float64: free it before the eigh

    lam, V = torch.linalg.eigh(whitened.to(eig_dtype))
    lam = lam[-k:].to(torch.float64).clamp_min(0)
    V = V[:, -k:].to(torch.float64)
    X = torch.linalg.solve_triangular(L.T.contiguous(), V, upper=True)  # L⁻ᵀ V
    return Pencil(lam=lam.clone(), X=X, FX=F_t_damped @ X, range_dim=size, dropped=0)


def _range_basis(F_sum: torch.Tensor, rel_tol: float) -> tuple[torch.Tensor, float]:
    """(U, top eigenvalue) where U holds the eigenvectors of F_sum = F_h + F_t with eigenvalue above
    rel_tol × the top eigenvalue: an orthonormal basis of the factors' joint range."""
    eigenvalues, eigenvectors = torch.linalg.eigh(F_sum)
    top = eigenvalues.max()
    keep = eigenvalues > rel_tol * top.clamp_min(torch.finfo(torch.float64).tiny)
    return eigenvectors[:, keep], float(top)


def _positivity_floor(damping: str, gamma: float, rel_tol: float, top_eigenvalue: float) -> float:
    """Eigenvalues of the restricted F_t′ at or below this are numerically non-positive.

    In exact arithmetic, covariant damping gives F_t′ ≥ min(1, γ) · rel_tol · max eig(F_h + F_t) on the
    range; half of that bound leaves room for rounding."""
    damping_scale = 1.0
    if damping == "covariant":
        damping_scale = min(1.0, gamma)
    return 0.5 * damping_scale * rel_tol * top_eigenvalue


def _whitener(F: torch.Tensor, floor: float) -> tuple[torch.Tensor, int]:
    """(W, number of dropped directions) with W F Wᵀ = I.

    Uses W = L⁻¹ from a Cholesky factorisation when it succeeds (an exact match with the numpy reference);
    otherwise an eigendecomposition that drops directions with eigenvalue ≤ floor."""
    L, info = torch.linalg.cholesky_ex(F)
    if int(info) == 0:
        identity = torch.eye(L.shape[0], dtype=L.dtype, device=L.device)
        return torch.linalg.solve_triangular(L, identity, upper=False), 0
    eigenvalues, Q = torch.linalg.eigh(F)
    positive = eigenvalues > floor
    whitener = (Q[:, positive] / eigenvalues[positive].sqrt()).T
    return whitener, int((~positive).sum())


def noise_floor(F: torch.Tensor, null_vec: torch.Tensor) -> float:
    """A factor's numerical floor: the Rayleigh quotient ûᵀ F û of a vector that is null in exact arithmetic,
    divided by F's top eigenvalue (0 for an all-zero factor)."""
    F = _symmetrize(F.to(torch.float64))
    u_hat = null_vec.to(F) / null_vec.norm()
    top_eigenvalue = torch.linalg.eigvalsh(F).max()
    if not top_eigenvalue > 0:
        return 0.0
    return float((u_hat @ F @ u_hat).abs() / top_eigenvalue)


def auto_rel_tol(floors, base: float = 1e-10, factor: float = 10.0) -> float:
    """The range threshold for factors with a measured floor: max(base, factor × the largest floor)."""
    floors = list(floors)
    if not floors:
        return base
    return max(base, factor * max(floors))


def _symmetrize(M: torch.Tensor) -> torch.Tensor:
    """½ (M + Mᵀ): removes the rounding asymmetry before an eigh or Cholesky."""
    return 0.5 * (M + M.T)
