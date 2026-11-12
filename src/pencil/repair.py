"""The pencil repair: per layer in closed form, then over a whole model.

Per linear layer W (d_out × d_in) with K-FAC factors F ≈ A ⊗ G (A input-sized, G output-sized) and the
damped task factors A_t′ = A_t + γ_c A_h, G_t′ = G_t + γ_c G_h:

    solve A_h x = α A_t′ x and G_h y = γ G_t′ y         (geneig; Xᵀ A_t′ X = I, Yᵀ G_t′ Y = I)
    λ_ji = γ_j α_i                                      safety curvature per unit of task curvature
    C  = Yᵀ G_t′ ΔW A_t′ X                              the update in pencil coordinates
    M  = μ / (λ + μ)                                    per-pair shrinkage
    ΔW′ = ΔW − Y ((1 − M) ∘ C) Xᵀ                       subtraction form: null directions pass through

and the forecast of the safety drift is ½ tr(G_h ΔW A_h ΔWᵀ) = ½ Σ λ C². tests/test_math.py::pencil_repair_np
is the reference; tests/test_torch_equivalence.py checks this file against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .delta import LoRADelta, as_dense
from .geneig import Pencil, damp, gen_eig, gen_eig_topk
from .modules import MLP_WIDE_INPUT, MLP_WIDE_OUTPUT, RESIDUAL_WRITERS, layer_order, leaf

F64 = torch.float64
SUMMED_REPORT_KEYS = ("S_before_pred", "S_after_pred", "R_pred")  # per-module report values summed over modules


def row_set_for(row_set: str | dict, module: str) -> str:
    """The row set a module's factors come from. `row_set` is one name for every module, or a dict
    {leaf: name, "default": name} (e.g. {"default": "full", "v_proj": "full_reduce"})."""
    if isinstance(row_set, str):
        return row_set
    default = row_set.get("default", "all")
    return row_set.get(leaf(module), default)


# ---------------------------------------------------------------- one layer
@dataclass
class LayerBasis:
    """A layer's two pencils: the input side (α, X) and the output side (γ, Y)."""

    inp: Pencil  # input side: α = inp.lam, X = inp.X, A_t′X = inp.FX
    out: Pencil  # output side: γ = out.lam, Y = out.X, G_t′Y = out.FX

    @property
    def alpha(self) -> torch.Tensor:
        return self.inp.lam

    @property
    def gamma(self) -> torch.Tensor:
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
    """Solve both sides of one layer: A_h x = α A_t′ x and G_h y = γ G_t′ y."""
    input_side = gen_eig(A_h, A_t, damping=damping, gamma=gamma, top_k=top_k_in, rel_tol=rel_tol)
    output_side = gen_eig(G_h, G_t, damping=damping, gamma=gamma, top_k=top_k_out, rel_tol=rel_tol)
    return LayerBasis(inp=input_side, out=output_side)


def coords(dW: torch.Tensor, basis: LayerBasis) -> torch.Tensor:
    """The update in pencil coordinates, C = Yᵀ G_t′ ΔW A_t′ X (k_out × k_in), in float64.

    The pencils store G_t′Y and A_t′X; G_t′ is symmetric, so Yᵀ G_t′ = (G_t′Y)ᵀ."""
    dW = dW.to(F64)
    G_t_damped_Y = basis.out.FX.to(dW)
    A_t_damped_X = basis.inp.FX.to(dW)
    return G_t_damped_Y.T @ dW @ A_t_damped_X


def repair_delta(dW: torch.Tensor, basis: LayerBasis, mu: float) -> torch.Tensor:
    """The repaired update ΔW′ = ΔW − Y ((1 − M) ∘ C) Xᵀ with M = μ / (λ + μ), λ_ji = γ_j α_i, in float64."""
    dW = dW.to(F64)
    lam = torch.outer(basis.gamma, basis.alpha).to(F64)
    M = mu / (lam + mu)
    C = coords(dW, basis)
    Y = basis.out.X.to(dW)
    X = basis.inp.X.to(dW)
    return dW - Y @ ((1.0 - M) * C) @ X.T


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
    """Repair one layer from its factors. Returns (W0 + ΔW′ in W0's dtype, step ratio ‖ΔW′ − ΔW‖ / ‖ΔW‖)."""
    basis = layer_basis(A_h, A_t, G_h, G_t, damping=damping, gamma=gamma, top_k_in=top_k_in, top_k_out=top_k_out)
    dW = dW.to(F64)
    dW_repaired = repair_delta(dW, basis, mu)
    dW_norm = dW.norm()
    step_ratio = 0.0
    if dW_norm > 0:
        step_ratio = float((dW_repaired - dW).norm() / dW_norm)
    W_repaired = (W0.to(F64) + dW_repaired).to(W0.dtype)
    return W_repaired, step_ratio


def forecast(dW: torch.Tensor, A_h: torch.Tensor, G_h: torch.Tensor) -> float:
    """The quadratic safety drift ½ ΔᵀF_hΔ = ½ tr(G_h ΔW A_h ΔWᵀ).

    Needs only the harmful factors (VERIFICATION.md Lemma 3.2 ii). A ranking, not a calibrated number."""
    dW = dW.to(F64)
    return 0.5 * float(torch.trace(G_h.to(F64) @ dW @ A_h.to(F64) @ dW.T))


def frontier(C: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, mus) -> tuple[list[float], list[float]]:
    """Predicted (S(μ), R(μ)) for each μ in `mus`, with λ_ji = γ_j α_i:

        S(μ) = ½ Σ λ (μ/(λ+μ))² C²        R(μ) = ½ Σ (λ/(λ+μ))² C²

    R assumes θ1 is a task minimum, which fine-tunes are not: plot the measured task regret, never this R
    (CLAUDE.md)."""
    lam = torch.outer(gamma, alpha).to(F64)
    C_squared = C.to(F64) ** 2
    S, R = [], []
    for mu in mus:
        M = mu / (lam + mu)
        one_minus_M = lam / (lam + mu)
        S.append(0.5 * float((lam * M**2 * C_squared).sum()))
        R.append(0.5 * float((one_minus_M**2 * C_squared).sum()))
    return S, R


# ---------------------------------------------------------------- the whole model
@dataclass
class _ModuleCache:
    """What every μ needs for one module, computed once per deltas dict by ModelRepair.prepare."""

    C: torch.Tensor  # Yᵀ G_t′ ΔW A_t′ X, the update in pencil coordinates (kept pairs only)
    lam: torch.Tensor  # λ_ji = γ_j α_i, same shape as C
    YtY: torch.Tensor  # Yᵀ Y, for the input-weighted step norm
    dW_norm: float  # ‖ΔW‖_F
    S_before: float  # ½ tr(G_h ΔW A_h ΔWᵀ): the forecast drift of the unrepaired update
    S_kept: float  # ½ Σ λ C²: the part of S_before that lives on the kept pairs
    input_norm: float  # tr(ΔW A_t′ ΔWᵀ) = ‖ΔW A_t′^½‖²


class ModelRepair:
    """The repair over every module that has a delta and factors.

    harm_factors: harmful factors at θ0; task_factors: task factors at θ1. Either can be a
    factor_store.Factors, FactorStore or LazyFactors.

    The work is split into three cached stages, so a μ sweep is cheap and memory stays bounded at Tiny Aya
    scale. Bases are never recomputed, so build one ModelRepair per damping setting and reuse it across μ.

      basis(module)    The layer's two pencils, solved once and kept compact (eigenvalues, X and F_t′X, never
                       the n × n factors). Input pencils are cached per (input factor, row set), because
                       q/k/v/gate/up share one input factor; output pencils are cached per module. MLP-wide
                       sides keep only top_k_mlp pairs, solved by gen_eig_topk once they are at least
                       fast_topk_min wide.
      prepare(deltas)  Per module, the pencil coordinates C and the scalars every μ needs (_ModuleCache).
                       Cached for the most recent deltas dict (checked by identity).
      repair(...)      The closed form per module from the two caches above; reads no factors.

    row_set: one row set for all modules, or {leaf: row_set, "default": row_set}.
    rel_tol: the range threshold for factors that had a known null vector (the shared input A, o/down G), set
        from their measured floor.
    rel_tol_other (default rel_tol): the threshold for every other factor, including the two that the V/O
        gauge transforms (o_proj input, v_proj output). Keep it small (1e-10) so the gauge can't move
        eigenvalues across it."""

    def __init__(
        self,
        harm_factors,
        task_factors,
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
        self.harm_factors = harm_factors
        self.task_factors = task_factors
        self.row_set = row_set
        self.damping = damping
        self.gamma = gamma
        self.top_k_mlp = top_k_mlp
        self.rel_tol = rel_tol
        self.rel_tol_other = rel_tol if rel_tol_other is None else rel_tol_other
        self.device = torch.device(device)
        self.eig_device = self.device if eig_device is None else torch.device(eig_device)
        self.basis_dtype = basis_dtype
        self.eig_dtype = eig_dtype
        self.fast_topk_min = fast_topk_min

        self._input_pencils: dict[tuple[str, str], Pencil] = {}  # (input factor key, row set) -> pencil
        self._output_pencils: dict[str, Pencil] = {}  # module -> pencil
        self._prepared: dict[str, _ModuleCache] = {}  # module -> cache for the deltas in _prepared_for
        self._prepared_for = None  # the deltas dict _prepared was built from

    # -- factors and bases
    def _row_set_of(self, module: str) -> str:
        return row_set_for(self.row_set, module)

    def factors(self, module: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(A_h, A_t, G_h, G_t) for a module, in float64 on self.device."""
        row_set = self._row_set_of(module)
        A_h, G_h = self.harm_factors.pair(module, row_set)
        A_t, G_t = self.task_factors.pair(module, row_set)
        return tuple(factor.to(self.device, F64) for factor in (A_h, A_t, G_h, G_t))

    def modules(self, deltas) -> list[str]:
        """The modules of `deltas` in layer order. Raises KeyError if either side lacks factors for one."""
        missing = []
        for module in deltas:
            if module not in self.harm_factors.a_key or module not in self.task_factors.a_key:
                missing.append(module)
        if missing:
            raise KeyError(f"no factors for {len(missing)} modules, e.g. {missing[:3]}")
        return layer_order(deltas)

    def _solve_side(self, F_h: torch.Tensor, F_t: torch.Tensor, top_k: int | None, rel_tol: float) -> Pencil:
        """One side's pencil, stored as basis_dtype on self.device. A wide side with top_k set uses the fast
        top-k solver (no range restriction: MLP-wide factors have no known null vector)."""
        size = F_h.shape[0]
        use_fast_topk = top_k is not None and top_k < size and size >= self.fast_topk_min
        if use_fast_topk:
            pencil = gen_eig_topk(
                F_h, F_t, top_k, self.damping, self.gamma, eig_dtype=self.eig_dtype, device=self.eig_device
            )
        else:
            pencil = gen_eig(
                F_h.to(self.eig_device),
                F_t.to(self.eig_device),
                damping=self.damping,
                gamma=self.gamma,
                top_k=top_k,
                rel_tol=rel_tol,
            )
        return pencil.to(dtype=self.basis_dtype, device=self.device)

    def jitters(self) -> dict[str, float]:
        """The ε (× mean diagonal) each solved side needed beyond the requested jitter, for the result JSONs:
        {"<input factor>|<row set>|input" or "<module>|output": ε}, only sides with ε > 0."""
        needed = {}
        for (factor_key, row_set), pencil in self._input_pencils.items():
            if pencil.jitter > 0:
                needed[f"{factor_key}|{row_set}|input"] = pencil.jitter
        for module, pencil in self._output_pencils.items():
            if pencil.jitter > 0:
                needed[f"{module}|output"] = pencil.jitter
        return needed

    def basis(self, module: str) -> LayerBasis:
        """The module's two pencils, solving (and caching) whichever side is not cached yet."""
        input_key = (self.harm_factors.a_key[module], self._row_set_of(module))
        if input_key not in self._input_pencils or module not in self._output_pencils:
            self._solve_missing_sides(module, input_key)
        return LayerBasis(inp=self._input_pencils[input_key], out=self._output_pencils[module])

    def _solve_missing_sides(self, module: str, input_key: tuple[str, str]) -> None:
        """Read the module's factors once and solve the sides that are not cached."""
        A_h, A_t, G_h, G_t = self.factors(module)
        module_leaf = leaf(module)
        if input_key not in self._input_pencils:
            # The shared q/k/v/gate/up input factor (key != module name) has the known null vector 1/w.
            has_known_null = input_key[0] != module
            top_k = self.top_k_mlp if module_leaf in MLP_WIDE_INPUT else None
            rel_tol = self.rel_tol if has_known_null else self.rel_tol_other
            self._input_pencils[input_key] = self._solve_side(A_h, A_t, top_k, rel_tol)
        if module not in self._output_pencils:
            # o_proj and down_proj write into the residual stream: their G has the all-ones null vector.
            has_known_null = module_leaf in RESIDUAL_WRITERS
            top_k = self.top_k_mlp if module_leaf in MLP_WIDE_OUTPUT else None
            rel_tol = self.rel_tol if has_known_null else self.rel_tol_other
            self._output_pencils[module] = self._solve_side(G_h, G_t, top_k, rel_tol)

    # -- per-delta cache
    def prepare(self, deltas: dict) -> None:
        """Cache per module what every μ needs (a no-op if `deltas` is the dict already prepared).

        With the pencil complete on the range, F_h is diagonal in pencil coordinates and zero outside, and
        truncated pairs pass through unchanged, so for δ = ΔW′ − ΔW and sums over the kept pairs

            S(ΔW′) = S(ΔW) − ½ Σ λ C² + ½ Σ λ (M C)²     and     ½ tr(G_t′ δ A_t′ δᵀ) = ½ Σ ((1 − M) C)²."""
        if self._prepared_for is deltas:
            return
        self._prepared = {}
        for module in self.modules(deltas):
            self._prepared[module] = self._prepare_module(module, deltas[module])
        self._prepared_for = deltas

    def _prepare_module(self, module: str, delta) -> _ModuleCache:
        """One module's _ModuleCache entry."""
        basis = self.basis(module)
        dW = as_dense(delta, F64, self.device)
        A_h, A_t, G_h, _ = self.factors(module)
        C = coords(dW, basis)
        lam = torch.outer(basis.gamma, basis.alpha).to(F64)
        Y = basis.out.X.to(F64)
        A_t_damped = damp(A_h, A_t, self.damping, self.gamma)
        return _ModuleCache(
            C=C,
            lam=lam,
            YtY=Y.T @ Y,
            dW_norm=float(dW.norm()),
            S_before=forecast(dW, A_h, G_h),
            S_kept=0.5 * float((lam * C**2).sum()),
            input_norm=float(torch.trace(dW @ A_t_damped @ dW.T)),
        )

    def _repair_module(self, module: str, mu: float) -> tuple[torch.Tensor, dict]:
        """(δ = ΔW′ − ΔW, per-module report) for one prepared module, from the caches only."""
        cache = self._prepared[module]
        basis = self.basis(module)
        M = mu / (cache.lam + mu)
        removed = (1.0 - M) * cache.C  # (1 − M) ∘ C: the part of each coordinate the repair takes out
        Y = basis.out.X.to(F64)
        X = basis.inp.X.to(F64)
        step = -(Y @ removed @ X.T)

        # ‖δ A_t′^½‖² = tr(removedᵀ YᵀY removed), because Xᵀ A_t′ X = I
        input_step = max(float(torch.trace(removed.T @ cache.YtY @ removed)), 0.0)
        S_kept_after = 0.5 * float((cache.lam * (M * cache.C) ** 2).sum())
        report = {
            "step_ratio": _ratio(float(step.norm()), cache.dW_norm),
            # The same ratio measured on task inputs. Big Euclidean steps along directions the inputs never
            # visit are harmless and show up only in step_ratio.
            "input_step_ratio": _ratio(input_step, cache.input_norm) ** 0.5,
            "S_before_pred": cache.S_before,
            "S_after_pred": cache.S_before - cache.S_kept + S_kept_after,
            "R_pred": 0.5 * float((removed**2).sum()),
            "range_in": basis.inp.range_dim,
            "range_out": basis.out.range_dim,
            "dropped": basis.inp.dropped + basis.out.dropped,
        }
        return step, report

    def repair(self, W0: dict, deltas: dict, mu: float, modules: list[str] | None = None) -> tuple[dict, dict]:
        """Repair at one μ. Returns ({module: W0 + ΔW′ in W0's dtype and device}, report).

        Modules outside `modules` (default: all) get the unrepaired W0 + ΔW: the targeted-repair control."""
        self.prepare(deltas)
        all_modules = self.modules(deltas)
        chosen = set(all_modules) if modules is None else set(modules)

        new_weights, per_module = {}, {}
        for module in all_modules:
            dW = as_dense(deltas[module], F64, self.device)
            if module in chosen:
                step, per_module[module] = self._repair_module(module, mu)
                dW = dW + step
            W = W0[module]
            new_weights[module] = (W.to(self.device, F64) + dW).to(dtype=W.dtype, device=W.device)
        return new_weights, _summarise(mu, per_module)

    def steps(self, deltas: dict, mu: float, modules: list[str] | None = None) -> dict[str, torch.Tensor]:
        """δ = ΔW′ − ΔW (float64) for the repaired modules only (all by default)."""
        self.prepare(deltas)
        chosen = self.modules(deltas) if modules is None else layer_order(modules)
        steps = {}
        for module in chosen:
            steps[module], _ = self._repair_module(module, mu)
        return steps

    def frontier(self, deltas: dict, mus) -> dict[str, list[float]]:
        """Predicted S(μ) and R(μ) summed over modules. R is "predicted (assumes a task minimum)"."""
        self.prepare(deltas)
        S, R = [], []
        for mu in mus:
            reports = [self._repair_module(module, mu)[1] for module in self._prepared]
            S.append(sum(report["S_after_pred"] for report in reports))
            R.append(sum(report["R_pred"] for report in reports))
        return {"mu": list(mus), "S_pred": S, "R_pred_assumes_task_minimum": R}


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, or 0 when the denominator is 0 (a module with no update)."""
    if denominator > 0:
        return numerator / denominator
    return 0.0


def _summarise(mu: float, per_module: dict[str, dict]) -> dict:
    """The whole-model report: per-module reports, their summed predictions and the largest step ratios."""
    rows = per_module.values()
    report = {"mu": mu, "modules": per_module}
    for key in SUMMED_REPORT_KEYS:
        report[key] = sum(row[key] for row in rows)
    report["max_step_ratio"] = max((row["step_ratio"] for row in rows), default=0.0)
    report["max_input_step_ratio"] = max((row["input_step_ratio"] for row in rows), default=0.0)
    return report


def trust_region_repair(
    harm_factors,
    task_factors,
    W0: dict,
    deltas: dict,
    mu: float,
    *,
    gamma_c: float = 1e-3,
    gamma_e_grid=(1e-3, 1e-2, 1e-1),
    max_step_ratio: float = 1.0,
    step_ratio_key: str = "max_step_ratio",
    actual_S=None,
    repairers: dict | None = None,
    **kw,
) -> tuple[dict, dict, dict]:
    """CLAUDE.md rule 11: try covariant damping first, then Euclidean damping, smallest γ_e first, until an
    attempt passes. An attempt passes when the report's `step_ratio_key` ratio ("max_step_ratio", Euclidean,
    or "max_input_step_ratio", input-weighted) is at most `max_step_ratio` and, if `actual_S` is given, the
    measured drift falls (`actual_S(new_weights)` < `actual_S(None)`, on dev). An empty `gamma_e_grid` means
    covariant damping only.

    Pass one `repairers` dict across μ values so each setting's bases are solved once; `kw` go to ModelRepair.
    Returns (new_weights, report, setting) for the first passing attempt, or for the last attempt with
    setting["failed"] = True. setting["attempts"] records every try."""
    if repairers is None:
        repairers = {}
    S_before = None
    if actual_S is not None:
        S_before = actual_S(None)

    settings = [("covariant", gamma_c)]
    for gamma_e in gamma_e_grid:
        settings.append(("euclidean", gamma_e))

    attempts = []
    for damping, gamma in settings:
        if (damping, gamma) not in repairers:
            repairers[(damping, gamma)] = ModelRepair(harm_factors, task_factors, damping=damping, gamma=gamma, **kw)
        new_weights, report = repairers[(damping, gamma)].repair(W0, deltas, mu)
        S_after = None
        if actual_S is not None:
            S_after = actual_S(new_weights)

        steps_ok = report[step_ratio_key] <= max_step_ratio
        drift_fell = S_after is None or S_after < S_before
        passed = steps_ok and drift_fell
        attempts.append(
            {
                "damping": damping,
                "gamma": gamma,
                "max_step_ratio": report["max_step_ratio"],
                "max_input_step_ratio": report["max_input_step_ratio"],
                "S_actual": S_after,
                "S_before": S_before,
                "passed": passed,
            }
        )
        if passed:
            return new_weights, report, {"damping": damping, "gamma": gamma, "attempts": attempts}
    return new_weights, report, {"damping": damping, "gamma": gamma, "attempts": attempts, "failed": True}


def forecast_many(
    harm_factors, deltas_by_checkpoint: dict[str, dict], row_set: str | dict = "all", device="cpu"
) -> dict[str, dict[str, float]]:
    """Per-module forecasts ½ tr(G_h ΔW A_h ΔWᵀ) for several checkpoints: {checkpoint: {module: forecast}}.

    One pass over the harmful factors in layer order, so streamed factors are read once."""
    forecasts = {checkpoint: {} for checkpoint in deltas_by_checkpoint}
    all_modules = set()
    for deltas in deltas_by_checkpoint.values():
        all_modules.update(deltas)

    for module in layer_order(all_modules):
        if module not in harm_factors.a_key:
            continue
        A_h, G_h = harm_factors.pair(module, row_set_for(row_set, module))
        A_h = A_h.to(device, F64)
        G_h = G_h.to(device, F64)
        for checkpoint, deltas in deltas_by_checkpoint.items():
            if module in deltas:
                forecasts[checkpoint][module] = _forecast_delta(deltas[module], A_h, G_h, device)
    return forecasts


def _forecast_delta(delta, A_h: torch.Tensor, G_h: torch.Tensor, device) -> float:
    """½ tr(G_h ΔW A_h ΔWᵀ) for a dense ΔW or a LoRADelta.

    A LoRADelta (ΔW = B A) uses ½ tr((Bᵀ G_h B)(A A_h Aᵀ)), which never forms the d_out × d_in matrix."""
    if not isinstance(delta, LoRADelta):
        return forecast(delta.to(device), A_h, G_h)
    lora_B = delta.B.to(device, F64)
    lora_A = delta.A.to(device, F64)
    output_side = lora_B.T @ G_h @ lora_B
    input_side = lora_A @ A_h @ lora_A.T
    return 0.5 * float(torch.trace(output_side @ input_side))


def task_grad_diagnostic(model, batches, steps: dict[str, torch.Tensor], R_pred: float) -> dict[str, float]:
    """The trust diagnostic for the predicted regret: |gᵀδ| / (½ δᵀĤδ), over the modules in `steps`.

    g is the mean token cross-entropy gradient at θ1 on `batches` and ½ δᵀĤδ = R_pred (K-FAC task curvature).
    Above 0.1 the predicted R(μ) is not trustworthy (CLAUDE.md). gᵀδ is summed per batch in float64, so bf16
    gradients never accumulate across batches."""
    weights = _enable_weight_grads(model, steps)
    device = next(iter(weights.values())).device
    g_dot_delta = 0.0
    n_tokens = 0
    for batch in batches:
        nll, batch_tokens = _summed_token_nll(model, batch, device)
        nll.backward()
        for module, W in weights.items():
            g_dot_delta += float((W.grad.to(F64) * steps[module].to(device)).sum())
            W.grad = None
        n_tokens += batch_tokens
    for W in weights.values():
        W.requires_grad_(False)

    g_dot_delta /= max(n_tokens, 1)
    ratio = float("inf")
    if R_pred > 0:
        ratio = abs(g_dot_delta) / R_pred
    return {"g_dot_delta": g_dot_delta, "half_dHd": R_pred, "ratio": ratio}


def _enable_weight_grads(model, modules) -> dict[str, torch.Tensor]:
    """Freeze every parameter except the weights of `modules`, which get requires_grad and no stale grad.
    Returns {module: weight}."""
    weights = {module: model.get_submodule(module).weight for module in modules}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for W in weights.values():
        W.requires_grad_(True)
        W.grad = None
    return weights


def _summed_token_nll(model, batch: dict, device) -> tuple[torch.Tensor, int]:
    """(Σ −log p(next token) over the scored positions, number of loss-mask positions) for one batch.
    Position t is scored when loss_mask[t] is set; its target is token t + 1."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device).bool()
    scored = loss_mask[:, :-1]
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = logits[:, :-1][scored].float().log_softmax(-1)
    targets = input_ids[:, 1:][scored]
    nll = -log_probs.gather(-1, targets[:, None]).sum()
    return nll, int(loss_mask.sum())
