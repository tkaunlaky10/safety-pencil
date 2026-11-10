"""Our baselines, on the same factors as the pencil, so a comparison isolates the curvature model.

Each maps a module's fine-tune update ΔW to a repaired ΔW′ (merged weights are W0 + ΔW′):
  RegMean / regmean_delta   RegMean (Jin et al. 2023): the K-FAC merge with G = I
  diag_fisher_delta         diagonal Fisher merge (Matena & Raffel 2022), per weight
  alignmerge_transform      AlignMerge-style merge (arXiv 2512.16245): the pencil with Euclidean rank-r
                            harmful factors (apply with LazyFactors.with_transform, then ModelRepair)
  safe_lora_delta           a minimal Safe LoRA (Hsu et al. 2024), the invariance negative control
  safe_delta_hessians       our input factors in the form SafeTune's Safe Delta takes
The standard Fisher merge is ModelRepair with general-instruction factors at θ0 instead of harmful ones.
SafeTune's own methods run in scripts/07_baselines.py.

Under the V/O gauge, RegMean is invariant (its factors transform covariantly); the diagonal Fisher merge,
the AlignMerge-style projection and Safe LoRA are not.
"""

from __future__ import annotations

import torch

from .geneig import gen_eig, gen_eig_topk

F64 = torch.float64


def regmean_delta(dW, A_h, A_t, mu: float, rcond: float = 1e-10) -> torch.Tensor:
    """ΔW′ = ΔW − ΔW A_h (A_h + μ A_t)⁺, directly (the reference for RegMean). μ is on another scale from the
    pencil's: sweep [0.3, 1000]."""
    dW, A_h, A_t = dW.to(F64), A_h.to(F64), A_t.to(F64)
    w, U = torch.linalg.eigh(0.5 * (A_h + mu * A_t + (A_h + mu * A_t).T))
    keep = w > rcond * w.max()
    pinv = (U[:, keep] / w[keep]) @ U[:, keep].T
    return dW - dW @ A_h @ pinv


class RegMean:
    """RegMean for a whole μ sweep from one eigendecomposition per input factor.

    With A_h x = β (A_h + A_t) x (Xᵀ S X = I for S = A_h + A_t, β in [0, 1]), on the range of S:
        ΔW′ = ΔW − (ΔW S X) diag(β / (μ + (1−μ)β)) Xᵀ.
    That pencil is gen_eig with covariant damping γ = 1. MLP-wide inputs keep the top_k pairs (the rest have
    small β and shrink by about β/μ)."""

    def __init__(
        self,
        fac_h,
        fac_t,
        row_set: str = "all",
        *,
        rcond: float = 1e-10,
        top_k: int | None = 512,
        fast_topk_min: int = 4096,
        device="cpu",
        eig_device=None,
        basis_dtype=torch.float32,
    ):
        self.fh, self.ft, self.row_set, self.rcond = fac_h, fac_t, row_set, rcond
        self.top_k, self.fast_topk_min = top_k, fast_topk_min
        self.device, self.eig_device, self.basis_dtype = device, eig_device or device, basis_dtype
        self._bases = {}

    def basis(self, module: str):
        key = self.fh.a_key[module]
        if key not in self._bases:
            A_h = self.fh.pair(module, self.row_set)[0].to(F64)
            A_t = self.ft.pair(module, self.row_set)[0].to(F64)
            if self.top_k and self.top_k < A_h.shape[0] and A_h.shape[0] >= self.fast_topk_min:
                p = gen_eig_topk(A_h, A_t, self.top_k, "covariant", 1.0, device=self.eig_device)
            else:
                p = gen_eig(A_h.to(self.eig_device), A_t.to(self.eig_device), "covariant", 1.0, None, self.rcond)
            self._bases[key] = p.to(dtype=self.basis_dtype, device=self.device)
        return self._bases[key]

    def delta(self, module: str, dW: torch.Tensor, mu: float) -> torch.Tensor:
        p = self.basis(module)
        dW, beta = dW.to(self.device, F64), p.lam.to(F64)
        return dW - ((dW @ p.FX.to(F64)) * (beta / (mu + (1.0 - mu) * beta))) @ p.X.to(F64).T


def kfac_diagonal(A: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """diag(A ⊗ G) as a d_out × d_in matrix. The fallback when the factor pass didn't collect the
    per-sequence diagonal (04_factors.py --diag); say so in the results."""
    return torch.outer(G.diagonal(), A.diagonal())


def diag_fisher_delta(dW, D_h, D_t, mu: float, floor: float = 1e-8) -> torch.Tensor:
    """θ = (D_h θ0 + μ D_t θ1) / (D_h + μ D_t) per weight, i.e. ΔW′ = μ D_t ΔW / (D_h + μ D_t), with each
    diagonal floored at `floor` × its max. Sweep μ over [1e-3, 3]."""
    D_h = D_h.to(F64).clamp_min(floor * float(D_h.max()))
    D_t = D_t.to(F64).clamp_min(floor * float(D_t.max()))
    return mu * D_t * dW.to(F64) / (D_h + mu * D_t)


def euclidean_projection(F: torch.Tensor, r: int) -> torch.Tensor:
    """P F P with P the Euclidean projector onto F's top-r eigenvectors."""
    F = F.to(F64)
    _, U = torch.linalg.eigh(0.5 * (F + F.T))
    P = U[:, -r:] @ U[:, -r:].T
    return P @ F @ P


def alignmerge_transform(r: int):
    """Harmful factors → their Euclidean rank-r projections. The AlignMerge objective
    ½‖Δ′ − Δ‖²_H + (λ/2) Δ′ᵀ P F_h P Δ′ is minimised by (H + λ P F_h P)⁻¹ H Δ: the pencil with F_h → P F_h P and
    μ = 1/λ, and in K-FAC form P F_h P = (P_in A_h P_in) ⊗ (P_out G_h P_out)."""

    def transform(kind, name, F):
        return F if kind == "D" else euclidean_projection(F, min(r, F.shape[0]))

    return transform


def safe_lora_delta(dW, W_aligned, W_base, threshold: float = 0.5) -> tuple[torch.Tensor, float]:
    """Safe LoRA: V = W_aligned − W_base, C = V Vᵀ / ‖V‖_F; replace ΔW by C ΔW when cos(C ΔW, ΔW) < threshold.
    Returns (ΔW′, cosine). C ΔW = (C B) A, so this equals projecting lora_B."""
    dW = dW.to(F64)
    V = W_aligned.to(F64) - W_base.to(F64)
    proj = (V @ V.T / V.norm()) @ dW
    cos = float((proj * dW).sum() / (proj.norm() * dW.norm()).clamp_min(1e-30))
    return (proj if cos < threshold else dW), cos


def safe_delta_hessians(fac_h, row_set: str = "all") -> dict[str, torch.Tensor]:
    """{"<module>.weight": A_h} in float32 for SafeDeltaTrainer(..., hessian_inputs=...) (our SafeTune patch):
    the per-layer input Hessian from the same harmful-prompt pass at θ0 the pencil uses."""
    return {f"{m}.weight": fac_h.pair(m, row_set)[0].float() for m in fac_h.modules}
