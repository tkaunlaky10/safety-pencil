"""Our baselines, on the same inputs as the pencil so a comparison isolates the curvature model.

Every function maps a module's fine-tune update ΔW to a repaired ΔW′ (merged weights are W0 + ΔW′):

- regmean_delta        RegMean (Jin et al. 2023): the K-FAC merge with G = I, input side only.
- diag_fisher_delta    diagonal Fisher merge (Matena & Raffel 2022), per weight.
- alignmerge_factors   AlignMerge-style merge (arXiv 2512.16245): the pencil with the harmful factors
                       replaced by their Euclidean rank-r projections. Feed the result to ModelRepair.
- safe_lora_delta      a minimal Safe LoRA (Hsu et al. 2024) projection, the invariance negative control.
- safe_delta_hessians  our A factors in the form SafeTune's Safe Delta takes (`hessian_inputs`).

The standard Fisher merge needs no code: run ModelRepair with general-instruction factors at θ0 in place
of the harmful ones. SafeTune's own methods (ReSta, LoX, WiSE-FT, SafeMERGE, ...) run through
`safetune.runner.recover`; see scripts/07_baselines.py.

Gauge behaviour (scripts/05_invariance.py): RegMean is a left/right matrix product with factors that
transform covariantly, so it is V/O-gauge invariant like the pencil; the diagonal Fisher merge, the
AlignMerge-style Euclidean projection and Safe LoRA are not.
"""
from __future__ import annotations

import torch

from .factors import Factors


def _pinv_psd(M: torch.Tensor, rcond: float = 1e-10) -> torch.Tensor:
    M = 0.5 * (M + M.T)
    w, U = torch.linalg.eigh(M)
    keep = w > rcond * w.max()
    return (U[:, keep] / w[keep]) @ U[:, keep].T


def regmean_delta(dW: torch.Tensor, A_h: torch.Tensor, A_t: torch.Tensor, mu: float, rcond: float = 1e-10) -> torch.Tensor:
    """ΔW′ = ΔW − ΔW A_h (A_h + μ A_t)⁺. μ lives on another scale from the pencil's: sweep [0.3, 1000]."""
    dW, A_h, A_t = dW.double(), A_h.double(), A_t.double()
    return dW - dW @ A_h @ _pinv_psd(A_h + mu * A_t, rcond)


def kfac_diagonal(A: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """diag(A ⊗ G) as a d_out × d_in matrix: diag(G) diag(A)ᵀ. The fallback when the true diagonal
    (KFACCollector(diag=True)) doesn't fit; say so in the results."""
    return torch.outer(G.diagonal(), A.diagonal())


def diag_fisher_delta(dW: torch.Tensor, D_h: torch.Tensor, D_t: torch.Tensor, mu: float, floor: float = 1e-8) -> torch.Tensor:
    """θ = (D_h θ0 + μ D_t θ1) / (D_h + μ D_t) per weight, so ΔW′ = μ D_t ΔW / (D_h + μ D_t).
    Each diagonal is floored at `floor` × its max. Sweep μ over [1e-3, 3]."""
    D_h = D_h.double().clamp_min(floor * float(D_h.max()))
    D_t = D_t.double().clamp_min(floor * float(D_t.max()))
    return mu * D_t * dW.double() / (D_h + mu * D_t)


def euclidean_projection(F: torch.Tensor, r: int) -> torch.Tensor:
    """P F P with P the Euclidean projector onto F's top-r eigenvectors."""
    w, U = torch.linalg.eigh(0.5 * (F.double() + F.double().T))
    Ur = U[:, -r:]
    P = Ur @ Ur.T
    return P @ F.double() @ P


def alignmerge_factors(fac_h: Factors, r: int) -> Factors:
    """Harmful factors with each A and G replaced by its Euclidean rank-r projection.

    The AlignMerge objective ½‖Δ′ − Δ‖²_H + (λ/2) Δ′ᵀ P F_h P Δ′ has minimiser (H + λ P F_h P)⁻¹ H Δ, which is
    the pencil repair with F_h → P F_h P and μ = 1/λ. In K-FAC form P = P_in ⊗ P_out, so P F_h P is
    (P_in A_h P_in) ⊗ (P_out G_h P_out). Run ModelRepair(alignmerge_factors(fac_h, r), fac_t)."""
    A = {k: {rs: euclidean_projection(t, min(r, t.shape[0])) for rs, t in v.items()} for k, v in fac_h.A.items()}
    G = {k: {rs: euclidean_projection(t, min(r, t.shape[0])) for rs, t in v.items()} for k, v in fac_h.G.items()}
    return Factors(A=A, G=G, a_key=dict(fac_h.a_key), n_rows=dict(fac_h.n_rows), n_loss=fac_h.n_loss,
                   n_seqs=fac_h.n_seqs, meta={**fac_h.meta, "alignmerge_rank": r})


def alignmerge_transform(r: int):
    """The LazyFactors version of alignmerge_factors: Euclidean rank-r projection applied layer by layer."""
    def fn(kind, key, F):
        return F if kind == "D" else euclidean_projection(F, min(r, F.shape[0]))
    return fn


def safe_lora_delta(dW: torch.Tensor, W_aligned: torch.Tensor, W_base: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, float]:
    """Safe LoRA: V = W_aligned − W_base, C = V Vᵀ / ‖V‖_F. Replace ΔW by C ΔW when cos(C ΔW, ΔW) < threshold.
    Returns (ΔW′, cosine). Prefer SafeTune's SafeLoRATrainer on the PEFT model for the main table; this
    version exists for the offline invariance test."""
    V = (W_aligned.double() - W_base.double())
    C = V @ V.T / V.norm()
    proj = C @ dW.double()
    cos = float((proj * dW.double()).sum() / (proj.norm() * dW.double().norm()).clamp_min(1e-30))
    return (proj if cos < threshold else dW.double()), cos


def safe_delta_hessians(fac_h: Factors, row_set: str = "all") -> dict[str, torch.Tensor]:
    """{"<module>.weight": A_h} in float32: the per-layer input Hessian Safe Delta expects, from the same
    harmful-prompt pass at θ0 the pencil uses (the paper builds it from a safety set on the aligned model).
    Pass as SafeDeltaTrainer(..., hessian_inputs=...) (needs our SafeTune patch)."""
    return {f"{m}.weight": fac_h.pair(m, row_set)[0].float() for m in fac_h.modules}


class RegMean:
    """RegMean for a whole μ sweep from one eigendecomposition per input factor.

    With the pencil A_h x = β (A_h + A_t) x (X normalised so Xᵀ(A_h + A_t)X = I, β in [0, 1]), on the range
    of A_h + A_t:  (A_h + μA_t)⁻¹ = X diag(1/(μ + (1−μ)β)) Xᵀ  and  A_h = S X diag(β) Xᵀ S, so
        ΔW′ = ΔW − (ΔW S X) diag(β / (μ + (1−μ)β)) Xᵀ.
    That pencil is gen_eig with covariant damping γ = 1. MLP-width inputs (down_proj) keep the top_k pairs
    (the discarded ones have small β and shrink by about β/μ), via gen_eig_topk. Bases are cached per A key."""

    def __init__(self, fac_h, fac_t, row_set: str = "all", *, rcond: float = 1e-10, top_k: int | None = 512,
                 fast_topk_min: int = 4096, device="cpu", eig_device=None, basis_dtype=torch.float32):
        self.fh, self.ft, self.rs, self.rcond = fac_h, fac_t, row_set, rcond
        self.top_k, self.fast_topk_min = top_k, fast_topk_min
        self.device, self.eig_device, self.basis_dtype = device, eig_device or device, basis_dtype
        self._b = {}

    def basis(self, module: str):
        from .geneig import gen_eig, gen_eig_topk
        key = self.fh.a_key[module]
        if key not in self._b:
            A_h = self.fh.pair(module, self.rs)[0].double()
            A_t = self.ft.pair(module, self.rs)[0].double()
            n = A_h.shape[0]
            if self.top_k and n >= self.fast_topk_min and self.top_k < n:
                p = gen_eig_topk(A_h, A_t, self.top_k, "covariant", 1.0, device=self.eig_device)
            else:
                p = gen_eig(A_h.to(self.eig_device), A_t.to(self.eig_device), "covariant", 1.0, None, self.rcond)
            self._b[key] = p.to(dtype=self.basis_dtype, device=self.device)
        return self._b[key]

    def delta(self, module: str, dW: torch.Tensor, mu: float) -> torch.Tensor:
        p = self.basis(module)
        dW = dW.to(self.device, torch.float64)
        beta = p.lam.double()
        shrink = beta / (mu + (1.0 - mu) * beta)
        return dW - ((dW @ p.FX.double()) * shrink) @ p.X.double().T
