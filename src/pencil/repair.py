"""Kronecker pencil repair, forecast and predicted frontier for one linear layer.

Conventions (CLAUDE.md, "Kronecker (K-FAC) form"): W is (d_out, d_in); A factors are d_in-sized
(input activations), G factors d_out-sized (output gradients); F ≈ A ⊗ G.

    C = Yᵀ G_t′ ΔW A_t′ X,   M_ji = μ / (γ_j α_i + μ),   ΔW′ = ΔW − Y ((1 − M) ∘ C) Xᵀ

The subtraction form leaves components of ΔW along exactly-null directions untouched, and with a
truncated basis (top-k pairs on a side) the discarded pairs pass through unrepaired.
`tests/test_math.py::pencil_repair_np` is the numpy reference; `tests/test_torch_equivalence.py`
checks this file against it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .geneig import Pencil, damp, gen_eig, gen_eig_topk


@dataclass
class LayerFactors:
    """K-FAC factors for one linear layer: harmful side at θ0, task side at θ1."""
    A_h: torch.Tensor
    A_t: torch.Tensor
    G_h: torch.Tensor
    G_t: torch.Tensor


@dataclass
class LayerBasis:
    inp: Pencil                # input side: alpha (= inp.lam), X
    out: Pencil                # output side: gamma (= out.lam), Y

    @property
    def alpha(self):
        return self.inp.lam

    @property
    def gamma(self):
        return self.out.lam


def layer_basis(f: LayerFactors, *, damping: str = "covariant", gamma: float = 1e-3,
                top_k_in: int | None = None, top_k_out: int | None = None, rel_tol: float = 1e-10) -> LayerBasis:
    return LayerBasis(inp=gen_eig(f.A_h, f.A_t, damping, gamma, top_k_in, rel_tol),
                      out=gen_eig(f.G_h, f.G_t, damping, gamma, top_k_out, rel_tol))


def coords(dW: torch.Tensor, b: LayerBasis) -> torch.Tensor:
    """C = Yᵀ G_t′ ΔW A_t′ X, the update in pencil coordinates (shape k_out × k_in)."""
    dW = dW.to(torch.float64)
    return b.out.FX.to(dW).T @ dW @ b.inp.FX.to(dW)


def shrinkage(b: LayerBasis, mu: float) -> torch.Tensor:
    return mu / (torch.outer(b.gamma, b.alpha).to(torch.float64) + mu)


def repair_delta(dW: torch.Tensor, b: LayerBasis, mu: float) -> torch.Tensor:
    """ΔW′ = ΔW − Y ((1 − M) ∘ C) Xᵀ in float64."""
    dW64 = dW.to(torch.float64)
    C = coords(dW64, b)
    M = shrinkage(b, mu)
    return dW64 - b.out.X.to(dW64) @ ((1.0 - M) * C) @ b.inp.X.to(dW64).T


def repair_layer(W0: torch.Tensor, dW: torch.Tensor, A_h: torch.Tensor, A_t: torch.Tensor,
                 G_h: torch.Tensor, G_t: torch.Tensor, mu: float, *, damping: str = "covariant",
                 gamma: float = 1e-3, top_k_in: int | None = None, top_k_out: int | None = None,
                 basis: LayerBasis | None = None) -> tuple[torch.Tensor, float]:
    """Return (W0 + ΔW′ in W0's dtype, step_ratio = ‖ΔW′ − ΔW‖_F / ‖ΔW‖_F).

    Pass a precomputed `basis` to sweep μ without re-solving the eigenproblems."""
    b = basis or layer_basis(LayerFactors(A_h, A_t, G_h, G_t), damping=damping, gamma=gamma,
                             top_k_in=top_k_in, top_k_out=top_k_out)
    dW64 = dW.to(torch.float64)
    new = repair_delta(dW64, b, mu)
    denom = torch.linalg.norm(dW64)
    ratio = float(torch.linalg.norm(new - dW64) / denom) if denom > 0 else 0.0
    W_new = (W0.to(torch.float64) + new).to(W0.dtype)
    return W_new, ratio


def forecast(dW: torch.Tensor, A_h: torch.Tensor, G_h: torch.Tensor) -> float:
    """½ tr(G_h ΔW A_h ΔWᵀ) = ½ vec(ΔW)ᵀ (A_h ⊗ G_h) vec(ΔW): the quadratic safety drift.

    Needs only the harmful factors (equal to ½ Σ α_i γ_j C_ij² for any task factors and any covariant
    damping; VERIFICATION.md Lemma 3.2 ii). Use it as a ranking, not a calibrated number."""
    dW = dW.to(torch.float64)
    return 0.5 * float(torch.trace(G_h.to(torch.float64) @ dW @ A_h.to(torch.float64) @ dW.T))


def frontier(C: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, mus) -> tuple[list[float], list[float]]:
    """Predicted S(μ) = ½ Σ λ (μ/(λ+μ))² C² and R(μ) = ½ Σ (λ/(λ+μ))² C², with λ_ji = γ_j α_i.

    R(μ) assumes θ1 is a task minimum, which fine-tunes are not: label it "predicted" and plot the
    measured task regret instead (CLAUDE.md)."""
    lam = torch.outer(gamma, alpha).to(torch.float64)
    C2 = C.to(torch.float64) ** 2
    S, R = [], []
    for mu in mus:
        S.append(0.5 * float((lam * (mu / (lam + mu)) ** 2 * C2).sum()))
        R.append(0.5 * float(((lam / (lam + mu)) ** 2 * C2).sum()))
    return S, R


def leverage_values(b: LayerBasis) -> torch.Tensor:
    """All λ_ji = γ_j α_i for the layer (k_out × k_in)."""
    return torch.outer(b.gamma, b.alpha)


# ---------------------------------------------------------------- whole-model repair
MLP_OUT = ("gate_proj", "up_proj")      # output side is MLP-width
MLP_IN = ("down_proj",)                 # input side is MLP-width
RESIDUAL_WRITERS = ("o_proj", "down_proj")   # output G has the known null vector 1 (see factors.py)


class ModelRepair:
    """Pencil repair over every module that has a delta and factors.

    fac_h: harmful factors at θ0; fac_t: task factors at θ1. Either in-memory `Factors` or streaming
    `LazyFactors` (pencil.factors); both are read one layer at a time, in layer order.

    Work is split so that a μ sweep is cheap and memory stays bounded at Tiny Aya scale:
    - `basis(n)`: the two pencils of a module, solved once and kept compact (eigenvalues, X and F_t′X, n×k).
      The input side is shared per A key and row set (q/k/v/gate/up share one). MLP-width sides with
      top_k_mlp use `gen_eig_topk` (Cholesky + one eigh in eig_dtype) once n >= fast_topk_min.
    - `prepare(deltas)`: per module, the pencil coordinates C of ΔW and the few scalars every μ needs.
    - `repair(W0, deltas, mu)`: closed-form per module from the cached C; no factor access.

    row_set: one row set for every module, or {leaf: row_set, "default": row_set}.
    rel_tol: range threshold for the factors that had a known null vector (shared input A, o/down G);
    rel_tol_other (default rel_tol) for the rest, which include the two the V/O gauge transforms
    (o_proj input, v_proj output): keep that one small (1e-10) so the gauge can't move eigenvalues across it.
    For real (fp32-summed) factors, project the known nulls out first and set rel_tol from their measured
    floors (runtime.prepare_factors).
    """

    def __init__(self, fac_h, fac_t, *, row_set: str | dict = "all", damping: str = "covariant",
                 gamma: float = 1e-3, top_k_mlp: int | None = None, rel_tol: float = 1e-10,
                 rel_tol_other: float | None = None, device: str | torch.device = "cpu",
                 eig_device: str | torch.device | None = None, basis_dtype: torch.dtype = torch.float64,
                 eig_dtype: torch.dtype = torch.float32, fast_topk_min: int = 4096):
        self.fh, self.ft = fac_h, fac_t
        self.row_set, self.damping, self.gamma = row_set, damping, gamma
        self.top_k_mlp, self.rel_tol, self.device = top_k_mlp, rel_tol, torch.device(device)
        self.rel_tol_other = rel_tol if rel_tol_other is None else rel_tol_other
        self.eig_device = torch.device(eig_device) if eig_device is not None else self.device
        self.basis_dtype, self.eig_dtype, self.fast_topk_min = basis_dtype, eig_dtype, fast_topk_min
        self._in: dict[tuple, Pencil] = {}
        self._out: dict[str, Pencil] = {}
        self._prep: dict[str, dict] = {}
        self._prep_for = None

    # -- bookkeeping
    @staticmethod
    def _leaf(name):
        return name.rsplit(".", 1)[-1]

    @staticmethod
    def order(names) -> list[str]:
        """Layer order (then name), so streaming factors load each layer once."""
        import re
        def key(n):
            m = re.search(r"layers\.(\d+)\.", n)
            return (int(m.group(1)) if m else -1, n)
        return sorted(names, key=key)

    def rs(self, name: str) -> str:
        if isinstance(self.row_set, str):
            return self.row_set
        return self.row_set.get(self._leaf(name), self.row_set.get("default", "all"))

    def factors(self, name):
        """(A_h, A_t, G_h, G_t) in float64 on `device`."""
        rs = self.rs(name)
        A_h, G_h = self.fh.pair(name, rs)
        A_t, G_t = self.ft.pair(name, rs)
        return tuple(t.to(self.device, torch.float64) for t in (A_h, A_t, G_h, G_t))

    def modules(self, deltas) -> list[str]:
        missing = [n for n in deltas if n not in self.fh.a_key or n not in self.ft.a_key]
        if missing:
            raise KeyError(f"no factors for {len(missing)} modules, e.g. {missing[:3]}")
        return self.order(deltas)

    def _solve(self, F_h, F_t, top_k, tol):
        if top_k is not None and F_h.shape[0] >= self.fast_topk_min and top_k < F_h.shape[0]:
            p = gen_eig_topk(F_h, F_t, top_k, self.damping, self.gamma, eig_dtype=self.eig_dtype,
                             device=self.eig_device)
        else:
            p = gen_eig(F_h.to(self.eig_device), F_t.to(self.eig_device), self.damping, self.gamma, top_k, tol)
        return p.to(dtype=self.basis_dtype, device=self.device)

    def basis(self, name: str) -> LayerBasis:
        key = (self.fh.a_key[name], self.rs(name))
        leaf = self._leaf(name)
        if key not in self._in or name not in self._out:
            A_h, A_t, G_h, G_t = self.factors(name)
            if key not in self._in:
                k = self.top_k_mlp if leaf in MLP_IN else None
                tol = self.rel_tol if key[0] != name else self.rel_tol_other          # shared input: known null
                self._in[key] = self._solve(A_h, A_t, k, tol)
            if name not in self._out:
                k = self.top_k_mlp if leaf in MLP_OUT else None
                tol = self.rel_tol if leaf in RESIDUAL_WRITERS else self.rel_tol_other  # o/down output: known null
                self._out[name] = self._solve(G_h, G_t, k, tol)
        return LayerBasis(inp=self._in[key], out=self._out[name])

    # -- per-delta preparation
    def prepare(self, deltas: dict[str, torch.Tensor]) -> None:
        """Coordinates and scalars per module. With a complete pencil on the range, F_h is diagonal in pencil
        coordinates and zero outside, and truncated pairs pass through unchanged, so
            S(ΔW′) = S(ΔW) − ½ Σ_kept λC² + ½ Σ_kept λ(MC)²,  ½ tr(G_t′ δ A_t′ δᵀ) = ½ Σ_kept ((1−M)C)²."""
        if self._prep_for is deltas:
            return
        self._prep = {}
        for n in self.modules(deltas):
            b = self.basis(n)
            d = deltas[n].to(self.device, torch.float64)
            A_h, A_t, G_h, _ = self.factors(n)
            C = coords(d, b)
            lam = torch.outer(b.gamma, b.alpha).to(torch.float64)
            A_td = damp(A_h, A_t, self.damping, self.gamma)
            Y = b.out.X.to(torch.float64)
            self._prep[n] = {"C": C, "lam": lam, "S_before": forecast(d, A_h, G_h),
                             "S_kept": 0.5 * float((lam * C ** 2).sum()),
                             "base_in": float(torch.trace(d @ A_td @ d.T)), "YtY": Y.T @ Y,
                             "d_norm": float(d.norm())}
        self._prep_for = deltas

    def forecast(self, deltas: dict[str, torch.Tensor]) -> dict[str, float]:
        """Per-module ½ tr(G_h ΔW A_h ΔWᵀ). Harmful factors only; no eigenproblems."""
        return forecast_many(self.fh, {"_": deltas}, self.rs, self.device)["_"]

    def _module(self, n, d, mu):
        P, b = self._prep[n], self.basis(n)
        M = mu / (P["lam"] + mu)
        Q = (1.0 - M) * P["C"]
        step = -(b.out.X.to(torch.float64) @ Q @ b.inp.X.to(torch.float64).T)
        info = {
            "step_ratio": float(step.norm()) / P["d_norm"] if P["d_norm"] > 0 else 0.0,
            # the same ratio on task inputs, ‖δ A_t′^½‖ / ‖ΔW A_t′^½‖: big Euclidean steps along directions
            # the inputs never visit are harmless (they show up only in step_ratio)
            "input_step_ratio": (max(float(torch.trace(Q.T @ P["YtY"] @ Q)), 0.0) / P["base_in"]) ** 0.5
            if P["base_in"] > 0 else 0.0,
            "S_before_pred": P["S_before"],
            "S_after_pred": P["S_before"] - P["S_kept"] + 0.5 * float((P["lam"] * (M * P["C"]) ** 2).sum()),
            "R_pred": 0.5 * float((Q ** 2).sum()),
            "range_in": b.inp.range_dim, "range_out": b.out.range_dim, "dropped": b.inp.dropped + b.out.dropped,
        }
        return step, info

    def repair(self, W0: dict[str, torch.Tensor], deltas: dict[str, torch.Tensor], mu: float,
               modules: list[str] | None = None) -> tuple[dict[str, torch.Tensor], dict]:
        """Return ({module: W0 + ΔW′ in W0's dtype, on W0's device}, report). Modules not in `modules` keep
        W0 + ΔW (the targeted-repair control)."""
        self.prepare(deltas)
        chosen = set(self.modules(deltas) if modules is None else modules)
        new, rep = {}, {"mu": mu, "modules": {}}
        for n in self.modules(deltas):
            d = deltas[n].to(self.device, torch.float64)
            if n in chosen:
                step, rep["modules"][n] = self._module(n, d, mu)
                d = d + step
            new[n] = (W0[n].to(self.device, torch.float64) + d).to(dtype=W0[n].dtype, device=W0[n].device)
        m = rep["modules"].values()
        for k in ("S_before_pred", "S_after_pred", "R_pred"):
            rep[k] = sum(x[k] for x in m)
        rep["max_step_ratio"] = max((x["step_ratio"] for x in m), default=0.0)
        rep["max_input_step_ratio"] = max((x["input_step_ratio"] for x in m), default=0.0)
        return new, rep

    def steps(self, deltas: dict[str, torch.Tensor], mu: float, modules: list[str] | None = None) -> dict[str, torch.Tensor]:
        """δ = ΔW′ − ΔW (float64) for the repaired modules only (all by default)."""
        self.prepare(deltas)
        chosen = self.modules(deltas) if modules is None else self.order(modules)
        return {n: self._module(n, deltas[n].to(self.device, torch.float64), mu)[0] for n in chosen}

    def frontier(self, deltas: dict[str, torch.Tensor], mus) -> dict[str, list[float]]:
        """Predicted S(μ) and R(μ) summed over modules. R(μ) is "predicted (assumes a task minimum)": never
        plot it as data."""
        self.prepare(deltas)
        S, R = [], []
        for mu in mus:
            s = r = 0.0
            for P in self._prep.values():
                M = mu / (P["lam"] + mu)
                s += P["S_before"] - P["S_kept"] + 0.5 * float((P["lam"] * (M * P["C"]) ** 2).sum())
                r += 0.5 * float((((1 - M) * P["C"]) ** 2).sum())
            S.append(s)
            R.append(r)
        return {"mu": list(mus), "S_pred": S, "R_pred_assumes_task_minimum": R}


def forecast_many(fac_h, deltas_by_ckpt: dict[str, dict], row_set, device="cpu") -> dict[str, dict[str, float]]:
    """Forecasts for several checkpoints in one pass over the harmful factors (layer order), so streaming
    factors are read once. row_set: a string or a callable module -> row set. Low-rank deltas
    (delta.LoRADelta) use ½ tr((BᵀG_hB)(A A_h Aᵀ)) without materialising ΔW."""
    from .delta import LoRADelta
    rs_of = row_set if callable(row_set) else (lambda n: row_set)
    names = ModelRepair.order({n for d in deltas_by_ckpt.values() for n in d})
    out = {c: {} for c in deltas_by_ckpt}
    for n in names:
        if n not in fac_h.a_key:
            continue
        A_h, G_h = (t.to(device, torch.float64) for t in fac_h.pair(n, rs_of(n)))
        for c, deltas in deltas_by_ckpt.items():
            if n not in deltas:
                continue
            d = deltas[n]
            if isinstance(d, LoRADelta):
                B, A = d.B.to(device, torch.float64), d.A.to(device, torch.float64)
                out[c][n] = 0.5 * float(torch.trace((B.T @ G_h @ B) @ (A @ A_h @ A.T)))
            else:
                out[c][n] = forecast(d.to(device), A_h, G_h)
    return out


def model_weights(model, names, device=None) -> dict[str, torch.Tensor]:
    """{module: weight} clones for the named linear modules (on `device`, default the weight's own)."""
    mods = dict(model.named_modules())
    return {n: mods[n].weight.detach().to(device or mods[n].weight.device, copy=True) for n in names}


def task_grad_diagnostic(model, batches, steps: dict[str, torch.Tensor], R_pred: float) -> dict[str, float]:
    """|gᵀδ| / (½ δᵀĤδ) with g the task-loss gradient at θ1 (mean token cross-entropy on `batches`) and
    ½ δᵀĤδ = R_pred from the K-FAC task curvature, over the modules in `steps`. Above 0.1 the predicted
    R(μ) is not trustworthy (CLAUDE.md). gᵀδ is accumulated per batch in float64, so bf16 gradients never
    sum across batches."""
    mods = dict(model.named_modules())
    params = {n: mods[n].weight for n in steps}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params.values():
        p.requires_grad_(True)
        p.grad = None
    dev = next(iter(params.values())).device
    gdot, n_tok = 0.0, 0
    for b in batches:
        ids, att, lm = b["input_ids"].to(dev), b["attention_mask"].to(dev), b["loss_mask"].to(dev).bool()
        logits = model(input_ids=ids, attention_mask=att).logits
        tgt = ids[:, 1:][lm[:, :-1]]
        lp = logits[:, :-1][lm[:, :-1]].float().log_softmax(-1)
        (-lp.gather(-1, tgt[:, None]).sum()).backward()
        for n, p in params.items():
            gdot += float((p.grad.double() * steps[n].to(dev)).sum())
            p.grad = None
        n_tok += int(lm.sum())
    for p in params.values():
        p.requires_grad_(False)
    gdot /= max(n_tok, 1)
    return {"g_dot_delta": gdot, "half_dHd": R_pred, "ratio": abs(gdot) / R_pred if R_pred > 0 else float("inf")}


def trust_region_repair(fac_h, fac_t, W0, deltas, mu, *, gamma_c: float = 1e-3, gamma_e_grid=(1e-3, 1e-2, 1e-1),
                        max_step_ratio: float = 1.0, actual_S=None, repairers: dict | None = None,
                        **kw) -> tuple[dict, dict, dict]:
    """CLAUDE.md rule 11. Try covariant damping first; if any module's step ratio exceeds `max_step_ratio`
    (or `actual_S(new_weights) >= actual_S(None)` when that callable is given; it should return S on dev,
    with None meaning the unrepaired model), fall back to Euclidean damping, smallest γ_e first.
    Pass the same `repairers` dict across μ values so each damping setting's bases are solved once.
    Returns (new_weights, report, setting). The setting records every attempt; Euclidean damping breaks
    exact invariance, so the invariance error must be reported at the chosen γ_e."""
    repairers = {} if repairers is None else repairers
    S0 = actual_S(None) if actual_S else None
    attempts = []
    for damping, gamma in [("covariant", gamma_c)] + [("euclidean", g) for g in gamma_e_grid]:
        if (damping, gamma) not in repairers:
            repairers[(damping, gamma)] = ModelRepair(fac_h, fac_t, damping=damping, gamma=gamma, **kw)
        new, report = repairers[(damping, gamma)].repair(W0, deltas, mu)
        S = actual_S(new) if actual_S else None
        ok = report["max_step_ratio"] <= max_step_ratio and (S is None or S < S0)
        attempts.append({"damping": damping, "gamma": gamma, "max_step_ratio": report["max_step_ratio"],
                         "max_input_step_ratio": report["max_input_step_ratio"], "S_actual": S, "S_before": S0,
                         "passed": ok})
        if ok:
            return new, report, {"damping": damping, "gamma": gamma, "attempts": attempts}
    return new, report, {"damping": damping, "gamma": gamma, "attempts": attempts, "failed": True}
