"""The pencil repair: per layer in closed form, then over a whole model.

Per linear layer W (d_out × d_in) with K-FAC factors F ≈ A ⊗ G (A input-sized, G output-sized):

    solve A_h x = α A_t′ x and G_h y = γ G_t′ y         (geneig; Xᵀ A_t′ X = I, Yᵀ G_t′ Y = I)
    C  = Yᵀ G_t′ ΔW A_t′ X                              the update in pencil coordinates
    M  = μ / (γ_j α_i + μ)                              per-pair shrinkage
    ΔW′ = ΔW − Y ((1 − M) ∘ C) Xᵀ                       subtraction form: null directions pass through

and the forecast of the safety drift is ½ tr(G_h ΔW A_h ΔWᵀ). tests/test_math.py::pencil_repair_np is the
reference; tests/test_torch_equivalence.py checks this file against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .delta import LoRADelta, as_dense
from .geneig import Pencil, damp, gen_eig, gen_eig_topk
from .modules import MLP_WIDE_INPUT, MLP_WIDE_OUTPUT, RESIDUAL_WRITERS, layer_order, leaf

F64 = torch.float64


def row_set_for(row_set: str | dict, module: str) -> str:
    """row_set is one name for every module, or {leaf: name, "default": name}."""
    if isinstance(row_set, str):
        return row_set
    return row_set.get(leaf(module), row_set.get("default", "all"))


# ---------------------------------------------------------------- one layer
@dataclass
class LayerBasis:
    inp: Pencil  # input side: α = inp.lam, X = inp.X
    out: Pencil  # output side: γ = out.lam, Y = out.X

    @property
    def alpha(self):
        return self.inp.lam

    @property
    def gamma(self):
        return self.out.lam


def layer_basis(
    A_h,
    A_t,
    G_h,
    G_t,
    *,
    damping: str = "covariant",
    gamma: float = 1e-3,
    top_k_in: int | None = None,
    top_k_out: int | None = None,
    rel_tol: float = 1e-10,
) -> LayerBasis:
    return LayerBasis(
        inp=gen_eig(A_h, A_t, damping, gamma, top_k_in, rel_tol),
        out=gen_eig(G_h, G_t, damping, gamma, top_k_out, rel_tol),
    )


def coords(dW: torch.Tensor, b: LayerBasis) -> torch.Tensor:
    """C = Yᵀ G_t′ ΔW A_t′ X (k_out × k_in)."""
    dW = dW.to(F64)
    return b.out.FX.to(dW).T @ dW @ b.inp.FX.to(dW)


def repair_delta(dW: torch.Tensor, b: LayerBasis, mu: float) -> torch.Tensor:
    """ΔW′ = ΔW − Y ((1 − M) ∘ C) Xᵀ in float64."""
    dW = dW.to(F64)
    M = mu / (torch.outer(b.gamma, b.alpha).to(F64) + mu)
    return dW - b.out.X.to(dW) @ ((1.0 - M) * coords(dW, b)) @ b.inp.X.to(dW).T


def repair_layer(
    W0,
    dW,
    A_h,
    A_t,
    G_h,
    G_t,
    mu: float,
    *,
    damping: str = "covariant",
    gamma: float = 1e-3,
    top_k_in: int | None = None,
    top_k_out: int | None = None,
) -> tuple[torch.Tensor, float]:
    """(W0 + ΔW′ in W0's dtype, step ratio ‖ΔW′ − ΔW‖ / ‖ΔW‖) for one layer."""
    b = layer_basis(A_h, A_t, G_h, G_t, damping=damping, gamma=gamma, top_k_in=top_k_in, top_k_out=top_k_out)
    dW = dW.to(F64)
    new = repair_delta(dW, b, mu)
    ratio = float((new - dW).norm() / dW.norm()) if dW.norm() > 0 else 0.0
    return (W0.to(F64) + new).to(W0.dtype), ratio


def forecast(dW: torch.Tensor, A_h: torch.Tensor, G_h: torch.Tensor) -> float:
    """½ tr(G_h ΔW A_h ΔWᵀ): the quadratic safety drift. Needs only the harmful factors (VERIFICATION.md
    Lemma 3.2 ii). A ranking, not a calibrated number."""
    dW = dW.to(F64)
    return 0.5 * float(torch.trace(G_h.to(F64) @ dW @ A_h.to(F64) @ dW.T))


def frontier(C: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, mus) -> tuple[list[float], list[float]]:
    """Predicted S(μ) = ½ Σ λ (μ/(λ+μ))² C² and R(μ) = ½ Σ (λ/(λ+μ))² C², λ_ji = γ_j α_i. R assumes θ1 is a
    task minimum, which fine-tunes are not: plot the measured task regret, never this R (CLAUDE.md)."""
    lam, C2 = torch.outer(gamma, alpha).to(F64), C.to(F64) ** 2
    S = [0.5 * float((lam * (mu / (lam + mu)) ** 2 * C2).sum()) for mu in mus]
    R = [0.5 * float(((lam / (lam + mu)) ** 2 * C2).sum()) for mu in mus]
    return S, R


# ---------------------------------------------------------------- the whole model
class ModelRepair:
    """The repair over every module that has a delta and factors.

    fac_h: harmful factors at θ0; fac_t: task factors at θ1 (factor_store.Factors or LazyFactors).

    Work is split so a μ sweep is cheap and memory stays bounded at Tiny Aya scale:
      basis(module)    the two pencils, solved once and kept compact. The input side is cached per input
                       factor and row set (q/k/v/gate/up share it). MLP-wide sides keep top_k_mlp pairs via
                       gen_eig_topk once they are at least fast_topk_min wide.
      prepare(deltas)  per module: pencil coordinates C and the scalars every μ needs.
      repair(...)      closed form per module from the cache; no factor reads.

    row_set: one row set for all modules, or {leaf: row_set, "default": row_set}.
    rel_tol: range threshold for factors that had a known null vector (shared input A, o/down G), set from
    their measured floor. rel_tol_other (default rel_tol) for the rest, which include the two the V/O gauge
    transforms (o_proj input, v_proj output): keep it small (1e-10) so the gauge can't move eigenvalues
    across it."""

    def __init__(
        self,
        fac_h,
        fac_t,
        *,
        row_set: str | dict = "all",
        damping: str = "covariant",
        gamma: float = 1e-3,
        top_k_mlp: int | None = None,
        rel_tol: float = 1e-10,
        rel_tol_other: float | None = None,
        device: str | torch.device = "cpu",
        eig_device: str | torch.device | None = None,
        basis_dtype: torch.dtype = F64,
        eig_dtype: torch.dtype = torch.float32,
        fast_topk_min: int = 4096,
    ):
        self.fh, self.ft = fac_h, fac_t
        self.row_set, self.damping, self.gamma, self.top_k_mlp = row_set, damping, gamma, top_k_mlp
        self.rel_tol = rel_tol
        self.rel_tol_other = rel_tol if rel_tol_other is None else rel_tol_other
        self.device = torch.device(device)
        self.eig_device = torch.device(eig_device) if eig_device is not None else self.device
        self.basis_dtype, self.eig_dtype, self.fast_topk_min = basis_dtype, eig_dtype, fast_topk_min
        self._inputs: dict[tuple, Pencil] = {}
        self._outputs: dict[str, Pencil] = {}
        self._prepared: dict[str, dict] = {}
        self._prepared_for = None

    # -- factors and bases
    def rs(self, module: str) -> str:
        return row_set_for(self.row_set, module)

    def factors(self, module: str):
        """(A_h, A_t, G_h, G_t) in float64."""
        A_h, G_h = self.fh.pair(module, self.rs(module))
        A_t, G_t = self.ft.pair(module, self.rs(module))
        return tuple(t.to(self.device, F64) for t in (A_h, A_t, G_h, G_t))

    def modules(self, deltas) -> list[str]:
        missing = [n for n in deltas if n not in self.fh.a_key or n not in self.ft.a_key]
        if missing:
            raise KeyError(f"no factors for {len(missing)} modules, e.g. {missing[:3]}")
        return layer_order(deltas)

    def _solve(self, F_h, F_t, top_k, rel_tol) -> Pencil:
        if top_k is not None and top_k < F_h.shape[0] and F_h.shape[0] >= self.fast_topk_min:
            p = gen_eig_topk(
                F_h, F_t, top_k, self.damping, self.gamma, eig_dtype=self.eig_dtype, device=self.eig_device
            )
        else:
            p = gen_eig(F_h.to(self.eig_device), F_t.to(self.eig_device), self.damping, self.gamma, top_k, rel_tol)
        return p.to(dtype=self.basis_dtype, device=self.device)

    def basis(self, module: str) -> LayerBasis:
        in_key = (self.fh.a_key[module], self.rs(module))
        if in_key not in self._inputs or module not in self._outputs:
            A_h, A_t, G_h, G_t = self.factors(module)
            if in_key not in self._inputs:
                shared = in_key[0] != module  # the shared input has a known null vector
                self._inputs[in_key] = self._solve(
                    A_h,
                    A_t,
                    self.top_k_mlp if leaf(module) in MLP_WIDE_INPUT else None,
                    self.rel_tol if shared else self.rel_tol_other,
                )
            if module not in self._outputs:
                writer = leaf(module) in RESIDUAL_WRITERS  # o/down outputs have a known null vector
                self._outputs[module] = self._solve(
                    G_h,
                    G_t,
                    self.top_k_mlp if leaf(module) in MLP_WIDE_OUTPUT else None,
                    self.rel_tol if writer else self.rel_tol_other,
                )
        return LayerBasis(inp=self._inputs[in_key], out=self._outputs[module])

    # -- per-delta cache
    def prepare(self, deltas: dict) -> None:
        """Cache per module what every μ needs. With the pencil complete on the range, F_h is diagonal in
        pencil coordinates and zero outside, and truncated pairs pass through unchanged, so
            S(ΔW′) = S(ΔW) − ½ Σ λC² + ½ Σ λ(MC)²   and   ½ tr(G_t′ δ A_t′ δᵀ) = ½ Σ ((1−M)C)²
        with the sums over kept pairs."""
        if self._prepared_for is deltas:
            return
        self._prepared = {}
        for n in self.modules(deltas):
            b = self.basis(n)
            dW = as_dense(deltas[n], F64, self.device)
            A_h, A_t, G_h, _ = self.factors(n)
            C, lam, Y = coords(dW, b), torch.outer(b.gamma, b.alpha).to(F64), b.out.X.to(F64)
            self._prepared[n] = {
                "C": C,
                "lam": lam,
                "YtY": Y.T @ Y,
                "dW_norm": float(dW.norm()),
                "S_before": forecast(dW, A_h, G_h),
                "S_kept": 0.5 * float((lam * C**2).sum()),
                "input_norm": float(torch.trace(dW @ damp(A_h, A_t, self.damping, self.gamma) @ dW.T)),
            }
        self._prepared_for = deltas

    def _repair_module(self, module: str, mu: float) -> tuple[torch.Tensor, dict]:
        """(δ = ΔW′ − ΔW, per-module report) from the cache."""
        P, b = self._prepared[module], self.basis(module)
        M = mu / (P["lam"] + mu)
        Q = (1.0 - M) * P["C"]
        step = -(b.out.X.to(F64) @ Q @ b.inp.X.to(F64).T)
        input_step = max(float(torch.trace(Q.T @ P["YtY"] @ Q)), 0.0)  # ‖δ A_t′^½‖² (Xᵀ A_t′ X = I)
        return step, {
            "step_ratio": float(step.norm()) / P["dW_norm"] if P["dW_norm"] > 0 else 0.0,
            # the same ratio on task inputs: big Euclidean steps along directions the inputs never visit
            # are harmless and show up only in step_ratio
            "input_step_ratio": (input_step / P["input_norm"]) ** 0.5 if P["input_norm"] > 0 else 0.0,
            "S_before_pred": P["S_before"],
            "S_after_pred": P["S_before"] - P["S_kept"] + 0.5 * float((P["lam"] * (M * P["C"]) ** 2).sum()),
            "R_pred": 0.5 * float((Q**2).sum()),
            "range_in": b.inp.range_dim,
            "range_out": b.out.range_dim,
            "dropped": b.inp.dropped + b.out.dropped,
        }

    def repair(self, W0: dict, deltas: dict, mu: float, modules: list[str] | None = None) -> tuple[dict, dict]:
        """({module: W0 + ΔW′ in W0's dtype and device}, report). Modules outside `modules` get W0 + ΔW
        (the targeted-repair control)."""
        self.prepare(deltas)
        chosen = set(self.modules(deltas) if modules is None else modules)
        new, per_module = {}, {}
        for n in self.modules(deltas):
            dW = as_dense(deltas[n], F64, self.device)
            if n in chosen:
                step, per_module[n] = self._repair_module(n, mu)
                dW = dW + step
            new[n] = (W0[n].to(self.device, F64) + dW).to(dtype=W0[n].dtype, device=W0[n].device)
        rows = per_module.values()
        report = {
            "mu": mu,
            "modules": per_module,
            **{k: sum(r[k] for r in rows) for k in ("S_before_pred", "S_after_pred", "R_pred")},
            "max_step_ratio": max((r["step_ratio"] for r in rows), default=0.0),
            "max_input_step_ratio": max((r["input_step_ratio"] for r in rows), default=0.0),
        }
        return new, report

    def steps(self, deltas: dict, mu: float, modules: list[str] | None = None) -> dict[str, torch.Tensor]:
        """δ = ΔW′ − ΔW (float64) for the repaired modules only (all by default)."""
        self.prepare(deltas)
        return {
            n: self._repair_module(n, mu)[0]
            for n in (self.modules(deltas) if modules is None else layer_order(modules))
        }

    def frontier(self, deltas: dict, mus) -> dict[str, list[float]]:
        """Predicted S(μ) and R(μ) summed over modules. R is "predicted (assumes a task minimum)"."""
        self.prepare(deltas)
        S, R = [], []
        for mu in mus:
            reports = [self._repair_module(n, mu)[1] for n in self._prepared]
            S.append(sum(r["S_after_pred"] for r in reports))
            R.append(sum(r["R_pred"] for r in reports))
        return {"mu": list(mus), "S_pred": S, "R_pred_assumes_task_minimum": R}


def trust_region_repair(
    fac_h,
    fac_t,
    W0: dict,
    deltas: dict,
    mu: float,
    *,
    gamma_c: float = 1e-3,
    gamma_e_grid=(1e-3, 1e-2, 1e-1),
    max_step_ratio: float = 1.0,
    actual_S=None,
    repairers: dict | None = None,
    **kw,
) -> tuple[dict, dict, dict]:
    """CLAUDE.md rule 11: covariant damping first; if any step ratio exceeds `max_step_ratio`, or the measured
    drift doesn't fall (`actual_S(new_weights)` vs `actual_S(None)`, on dev), fall back to Euclidean damping,
    smallest γ_e first. Pass one `repairers` dict across μ values so each setting's bases are solved once.
    Returns (new_weights, report, setting); setting["attempts"] records every try."""
    repairers = {} if repairers is None else repairers
    S_before = actual_S(None) if actual_S else None
    attempts = []
    for damping, gamma in [("covariant", gamma_c)] + [("euclidean", g) for g in gamma_e_grid]:
        if (damping, gamma) not in repairers:
            repairers[(damping, gamma)] = ModelRepair(fac_h, fac_t, damping=damping, gamma=gamma, **kw)
        new, report = repairers[(damping, gamma)].repair(W0, deltas, mu)
        S = actual_S(new) if actual_S else None
        passed = report["max_step_ratio"] <= max_step_ratio and (S is None or S < S_before)
        attempts.append(
            {
                "damping": damping,
                "gamma": gamma,
                "max_step_ratio": report["max_step_ratio"],
                "max_input_step_ratio": report["max_input_step_ratio"],
                "S_actual": S,
                "S_before": S_before,
                "passed": passed,
            }
        )
        if passed:
            return new, report, {"damping": damping, "gamma": gamma, "attempts": attempts}
    return new, report, {"damping": damping, "gamma": gamma, "attempts": attempts, "failed": True}


def forecast_many(
    fac_h, deltas_by_checkpoint: dict[str, dict], row_set: str | dict = "all", device="cpu"
) -> dict[str, dict[str, float]]:
    """Per-module forecasts for several checkpoints in one pass over the harmful factors (layer order, so
    streamed factors are read once). Low-rank deltas use ½ tr((BᵀG_hB)(A A_h Aᵀ)) without forming ΔW."""
    out = {c: {} for c in deltas_by_checkpoint}
    for n in layer_order({n for d in deltas_by_checkpoint.values() for n in d}):
        if n not in fac_h.a_key:
            continue
        A_h, G_h = (t.to(device, F64) for t in fac_h.pair(n, row_set_for(row_set, n)))
        for c, deltas in deltas_by_checkpoint.items():
            if n not in deltas:
                continue
            d = deltas[n]
            if isinstance(d, LoRADelta):
                B, A = d.B.to(device, F64), d.A.to(device, F64)
                out[c][n] = 0.5 * float(torch.trace((B.T @ G_h @ B) @ (A @ A_h @ A.T)))
            else:
                out[c][n] = forecast(d.to(device), A_h, G_h)
    return out


def task_grad_diagnostic(model, batches, steps: dict[str, torch.Tensor], R_pred: float) -> dict[str, float]:
    """|gᵀδ| / (½ δᵀĤδ), with g the mean token cross-entropy gradient at θ1 on `batches` and ½ δᵀĤδ = R_pred
    (K-FAC task curvature), over the modules in `steps`. Above 0.1 the predicted R(μ) is not trustworthy
    (CLAUDE.md). gᵀδ is summed per batch in float64, so bf16 gradients never accumulate across batches."""
    weights = {n: model.get_submodule(n).weight for n in steps}
    for p in model.parameters():
        p.requires_grad_(False)
    for w in weights.values():
        w.requires_grad_(True)
        w.grad = None
    dev = next(iter(weights.values())).device
    g_dot, n_tok = 0.0, 0
    for b in batches:
        ids, att, lm = b["input_ids"].to(dev), b["attention_mask"].to(dev), b["loss_mask"].to(dev).bool()
        logp = model(input_ids=ids, attention_mask=att).logits[:, :-1][lm[:, :-1]].float().log_softmax(-1)
        (-logp.gather(-1, ids[:, 1:][lm[:, :-1]][:, None]).sum()).backward()
        for n, w in weights.items():
            g_dot += float((w.grad.to(F64) * steps[n].to(dev)).sum())
            w.grad = None
        n_tok += int(lm.sum())
    for w in weights.values():
        w.requires_grad_(False)
    g_dot /= max(n_tok, 1)
    return {"g_dot_delta": g_dot, "half_dHd": R_pred, "ratio": abs(g_dot) / R_pred if R_pred > 0 else float("inf")}
