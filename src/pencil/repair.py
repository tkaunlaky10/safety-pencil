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

The scaled repair (13 Nov, post-freeze; tests/test_math.py::pencil_repair_scaled_np) adds two options to
ModelRepair, both off by default:
  output_nu_rel  output side G″ = G_t′ + νI, ν = output_nu_rel · mean(diag G_t′): a penalty on the input-weighted
                 step (μν/2)‖δ A_t′^½‖², with the same closed form
  scale_steps    per module δ ← t δ, t = min(1, 1/r), r = ‖δ A_t^½‖_F / ‖ΔW A_t^½‖_F (rule 11's input-weighted
                 ratio, so every scaled module passes it by construction)
Modules in exempt_leaves (v_proj: the V/O gauge transforms its output factor) get neither. Their trust-region
ratio is the curvature-norm ratio ‖δ‖ / ‖ΔW‖ in tr(G_t′ · A_t′ ·ᵀ), gauge-invariant and at most 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .delta import LoRADelta, as_dense
from .geneig import DAMPINGS, Pencil, gen_eig, gen_eig_topk
from .modules import MLP_WIDE_INPUT, MLP_WIDE_OUTPUT, RESIDUAL_WRITERS, layer_order, leaf

F64 = torch.float64
SUMMED_REPORT_KEYS = ("S_before_pred", "S_after_pred", "R_pred")  # per-module report values summed over modules
GAUGE_TOUCHED_OUTPUT = ("v_proj",)  # leaves whose output factor the V/O gauge transforms
TRUST_RATIO_TOLERANCE = 1e-9  # a scaled ratio of exactly 1 can come out a rounding error above it


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

    C: torch.Tensor  # Yᵀ G″ ΔW A_t′ X, the update in pencil coordinates (kept pairs only)
    lam: torch.Tensor  # λ_ji = γ_j α_i, same shape as C
    YtY: torch.Tensor  # Yᵀ Y (basis dtype), for the Euclidean and input-weighted step norms
    YtGY: torch.Tensor  # Yᵀ G_t′ Y (basis dtype; I without ν or jitter), for the task regret
    input_gram: torch.Tensor  # Xᵀ A_t′ X (basis dtype, shared per input pencil; I without jitter)
    task_input_gram: torch.Tensor  # Xᵀ A_t X (basis dtype, shared per input pencil)
    dW_norm: float  # ‖ΔW‖_F
    S_before: float  # ½ tr(G_h ΔW A_h ΔWᵀ): the forecast drift of the unrepaired update
    S_kept: float  # ½ Σ λ C²: the part of S_before that lives on the kept pairs
    input_norm: float  # tr(ΔW A_t′ ΔWᵀ) = ‖ΔW A_t′^½‖²
    task_input_norm: float  # tr(ΔW A_t ΔWᵀ) = ‖ΔW A_t^½‖²: the denominator of the trust-region ratio
    harm_input_norm: float  # tr(ΔW A_h ΔWᵀ) = ‖ΔW A_h^½‖²: the denominator of the harmful-input ratio
    alpha: torch.Tensor  # the input side's α: Xᵀ A_h X = diag(α), for the harmful-input step norm
    curvature_norm: float  # tr(G_t′ ΔW A_t′ ΔWᵀ): the denominator of the exempt modules' trust-region ratio
    exempt: bool  # in exempt_leaves: no ν, no scaling


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
    output_nu_rel, scale_steps, exempt_leaves: the scaled repair (module docstring).
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
        output_nu_rel: float = 0.0,
        scale_steps: bool = False,
        exempt_leaves: tuple[str, ...] = GAUGE_TOUCHED_OUTPUT,
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
        self.output_nu_rel = output_nu_rel
        self.scale_steps = scale_steps
        self.exempt_leaves = tuple(exempt_leaves)
        if damping not in DAMPINGS:
            raise ValueError(f"damping must be one of {DAMPINGS}, got {damping!r}")
        if scale_steps and damping != "covariant":
            raise ValueError("scale_steps needs covariant damping (the scaled repair's input side)")

        self._input_pencils: dict[tuple[str, str], Pencil] = {}  # (input factor key, row set) -> pencil
        self._output_pencils: dict[str, Pencil] = {}  # module -> pencil
        # (input factor key, row set) -> (Xᵀ A_t′ X, Xᵀ A_t X)
        self._input_grams: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._covariant_bounds: dict[str, float] = {}  # module -> ceiling diagnostic of a ν-damped output side
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

    def _solve_side(
        self, F_h: torch.Tensor, F_t: torch.Tensor, top_k: int | None, rel_tol: float, nu_rel: float = 0.0
    ) -> Pencil:
        """One side's pencil, stored as basis_dtype on self.device. A wide side with top_k set uses the fast
        top-k solver (no range restriction: MLP-wide factors have no known null vector)."""
        size = F_h.shape[0]
        use_fast_topk = top_k is not None and top_k < size and size >= self.fast_topk_min
        if use_fast_topk:
            pencil = gen_eig_topk(
                F_h,
                F_t,
                top_k,
                self.damping,
                self.gamma,
                eig_dtype=self.eig_dtype,
                device=self.eig_device,
                nu_rel=nu_rel,
            )
        else:
            pencil = gen_eig(
                F_h.to(self.eig_device),
                F_t.to(self.eig_device),
                damping=self.damping,
                gamma=self.gamma,
                top_k=top_k,
                rel_tol=rel_tol,
                nu_rel=nu_rel,
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
            nu_rel = 0.0 if self._exempt(module) else self.output_nu_rel
            pencil = self._solve_side(G_h, G_t, top_k, rel_tol, nu_rel)
            self._output_pencils[module] = pencil
            if pencil.nu > 0:
                self._covariant_bounds[module] = _covariant_ceiling_bound(pencil)

    def _exempt(self, module: str) -> bool:
        """True for the modules that get neither ν nor scaling (exempt_leaves)."""
        return leaf(module) in self.exempt_leaves

    def _input_grams_of(self, module: str, A_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(Xᵀ A_t′ X, Xᵀ A_t X) for the module's input pencil, cached per input pencil.

        Xᵀ A_t′ X = Xᵀ (A_t′X) is I up to jitter. Since Xᵀ A_h X = diag(α), Xᵀ A_t X is Xᵀ A_t′ X − γ_c diag(α)
        under covariant damping and Xᵀ A_t′ X without damping; Euclidean damping computes it from A_t."""
        input_key = (self.harm_factors.a_key[module], self._row_set_of(module))
        if input_key not in self._input_grams:
            pencil = self._input_pencils[input_key]
            X = pencil.X.to(F64)
            damped = _symmetric(X.T @ pencil.FX.to(F64))
            if self.damping == "covariant":
                task = damped - self.gamma * torch.diag(pencil.lam.to(F64))
            elif self.damping == "none":
                task = damped
            else:
                task = _symmetric(X.T @ (A_t @ X))
            self._input_grams[input_key] = (damped.to(self.basis_dtype), task.to(self.basis_dtype))
        return self._input_grams[input_key]

    def ceiling_report(self) -> dict:
        """Diagnostic, never used to choose anything: under covariant damping every eigenvalue is at most 1/γ_c
        when the task factor is positive semidefinite. Lists each side's largest covariant eigenvalue (α for an
        input side, γ for a ν-free output side, and for a ν-damped output side the largest covariant Rayleigh
        quotient along its solved directions, a lower bound) and flags those above (1/γ_c)(1 + 1e-3)."""
        ceiling = (1.0 / self.gamma) * (1.0 + 1e-3)
        largest = {}
        for (factor_key, row_set), pencil in self._input_pencils.items():
            largest[f"{factor_key}|{row_set}|input"] = float(pencil.lam.max())
        for module, pencil in self._output_pencils.items():
            largest[f"{module}|output"] = self._covariant_bounds.get(module, float(pencil.lam.max()))
        flagged = sorted(side for side, value in largest.items() if value > ceiling)
        return {"ceiling": ceiling, "largest": largest, "flagged": flagged}

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
        A_h, A_t, G_h, G_t = self.factors(module)
        C = coords(dW, basis)
        lam = torch.outer(basis.gamma, basis.alpha).to(F64)
        Y = basis.out.X.to(F64)
        YtY = Y.T @ Y
        YtGY = Y.T @ basis.out.FX.to(F64) - basis.out.nu * YtY  # Yᵀ G″ Y − ν YᵀY
        dW_A_h, dW_A_t = dW @ A_h, dW @ A_t
        G_h_dW = G_h @ dW
        dW_A_damped = self._damped_product(dW_A_t, dW_A_h, dW, A_t)  # ΔW A_t′
        G_damped_dW = self._damped_product(G_t @ dW, G_h_dW, dW, G_t)  # G_t′ ΔW
        input_gram, task_input_gram = self._input_grams_of(module, A_t)
        return _ModuleCache(
            C=C,
            lam=lam,
            YtY=YtY.to(self.basis_dtype),
            YtGY=_symmetric(YtGY).to(self.basis_dtype),
            input_gram=input_gram,
            task_input_gram=task_input_gram,
            dW_norm=float(dW.norm()),
            S_before=0.5 * float((G_h_dW * dW_A_h).sum()),
            S_kept=0.5 * float((lam * C**2).sum()),
            input_norm=float((dW_A_damped * dW).sum()),
            task_input_norm=float((dW_A_t * dW).sum()),
            harm_input_norm=float((dW_A_h * dW).sum()),
            alpha=basis.alpha.to(F64),
            curvature_norm=float((G_damped_dW * dW_A_damped).sum()),
            exempt=self._exempt(module),
        )

    def _damped_product(self, with_task, with_harm, dW, F_t) -> torch.Tensor:
        """ΔW F_t′ (or F_t′ ΔW) from the products with F_t and F_h, for this repair's damping."""
        if self.damping == "covariant":
            return with_task + self.gamma * with_harm
        if self.damping == "euclidean":
            return with_task + self.gamma * float(F_t.diagonal().mean()) * dW
        return with_task

    def _module_terms(self, module: str, mu: float) -> tuple[torch.Tensor, float, dict]:
        """(removed = (1 − M) ∘ C, the scale t, the module's report without its Euclidean ratio), from the caches.

        With δ = −t Y removed Xᵀ every norm is a trace over pencil coordinates:
            ‖δ A_t′^½‖² = t² Σ (YᵀY R) ∘ (R Xᵀ A_t′ X)          ‖δ A_t^½‖² the same with Xᵀ A_t X
            ‖δ A_h^½‖² = t² Σ (YᵀY R) ∘ R ∘ α                    (Xᵀ A_h X = diag(α))
            2 R(δ) = tr(G_t′ δ A_t′ δᵀ) = t² Σ (Yᵀ G_t′ Y R) ∘ (R Xᵀ A_t′ X)
        and the drift on the kept pairs is ½ Σ λ ((1 − t(1 − M)) C)²."""
        cache = self._prepared[module]
        M = _shrink(cache.lam, mu)
        removed = (1.0 - M) * cache.C
        R_input_gram = removed @ cache.input_gram.to(F64)
        YtY_R = cache.YtY.to(F64) @ removed
        input_step = max(float((YtY_R * R_input_gram).sum()), 0.0)
        task_input_step = max(float((YtY_R * (removed @ cache.task_input_gram.to(F64))).sum()), 0.0)
        curvature_step = max(float((cache.YtGY.to(F64) @ removed * R_input_gram).sum()), 0.0)
        harm_input_step = max(float((YtY_R * removed * cache.alpha).sum()), 0.0)  # Xᵀ A_h X = diag(α)

        task_ratio = _ratio_or_inf(task_input_step, cache.task_input_norm) ** 0.5
        scale = 1.0
        if self.scale_steps and not cache.exempt and task_ratio > 1.0:
            scale = 1.0 / task_ratio
        curvature_ratio = scale * _ratio(curvature_step, cache.curvature_norm) ** 0.5
        S_kept_after = 0.5 * float((cache.lam * ((1.0 - scale * (1.0 - M)) * cache.C) ** 2).sum())
        report = {
            "scale": scale,
            # rule 11's input-weighted ratio ‖δ A_t^½‖ / ‖ΔW A_t^½‖, before and after scaling
            "task_input_step_ratio_unscaled": task_ratio,
            "task_input_step_ratio": scale * task_ratio,
            # the same with A_t′ = A_t + γ_c A_h, as logged on 12 Nov
            "input_step_ratio": scale * _ratio(input_step, cache.input_norm) ** 0.5,
            "curvature_step_ratio": curvature_ratio,
            # diagnostic only, never used to accept a μ: the step on inputs from harmful prompts
            "harm_input_step_ratio": scale * _ratio(harm_input_step, cache.harm_input_norm) ** 0.5,
            "trust_ratio": curvature_ratio if cache.exempt else scale * task_ratio,
            "S_before_pred": cache.S_before,
            "S_after_pred": cache.S_before - cache.S_kept + S_kept_after,
            "R_pred": 0.5 * scale**2 * curvature_step,
        }
        return removed, scale, report

    def _repair_module(self, module: str, mu: float) -> tuple[torch.Tensor, dict]:
        """(δ = ΔW′ − ΔW after scaling, per-module report) for one prepared module, from the caches only."""
        removed, scale, report = self._module_terms(module, mu)
        basis = self.basis(module)
        Y = basis.out.X.to(F64)
        X = basis.inp.X.to(F64)
        step = -(Y @ removed @ X.T)
        unscaled_ratio = _ratio(float(step.norm()), self._prepared[module].dW_norm)
        report["step_ratio_unscaled"] = unscaled_ratio
        report["step_ratio"] = scale * unscaled_ratio  # Euclidean, the pre-registered metric
        report["range_in"] = basis.inp.range_dim
        report["range_out"] = basis.out.range_dim
        report["dropped"] = basis.inp.dropped + basis.out.dropped
        return scale * step, report

    def predicted(self, deltas: dict, mu: float) -> dict[str, float]:
        """Predicted S and R at μ summed over modules, with scaling, without forming any step (cheap)."""
        self.prepare(deltas)
        S, R = 0.0, 0.0
        for module in self._prepared:
            report = self._module_terms(module, mu)[2]
            S += report["S_after_pred"]
            R += report["R_pred"]
        return {"S_pred": S, "R_pred": R}

    def repair(
        self,
        W0: dict,
        deltas: dict,
        mu: float,
        modules: list[str] | None = None,
        task_gradient: TaskGradient | None = None,
    ) -> tuple[dict, dict]:
        """Repair at one μ. Returns ({module: W0 + ΔW′ in W0's dtype and device}, report).

        Modules outside `modules` (default: all) get the unrepaired W0 + ΔW: the targeted-repair control. With
        `task_gradient` (the task gradient g at θ1) the report adds the diagnostic |gᵀδ| / (½ δᵀĤδ), gᵀδ summed
        module by module as each step is made, so no more than one dense step is held at a time."""
        self.prepare(deltas)
        all_modules = self.modules(deltas)
        chosen = set(all_modules) if modules is None else set(modules)

        new_weights, per_module = {}, {}
        g_dot_delta = 0.0
        for module in all_modules:
            dW = as_dense(deltas[module], F64, self.device)
            if module in chosen:
                step, per_module[module] = self._repair_module(module, mu)
                if task_gradient is not None:
                    g_dot_delta += task_gradient.dot(module, step)
                dW = dW + step
            W = W0[module]
            new_weights[module] = (W.to(self.device, F64) + dW).to(dtype=W.dtype, device=W.device)
        report = _summarise(mu, per_module)
        if task_gradient is not None:
            report["task_grad_diagnostic"] = _grad_diagnostic(g_dot_delta, report["R_pred"])
        return new_weights, report

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
        S, R = [], []
        for mu in mus:
            totals = self.predicted(deltas, mu)
            S.append(totals["S_pred"])
            R.append(totals["R_pred"])
        return {"mu": list(mus), "S_pred": S, "R_pred_assumes_task_minimum": R}


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, or 0 when the denominator is 0 (a module with no update)."""
    if denominator > 0:
        return numerator / denominator
    return 0.0


def _ratio_or_inf(numerator: float, denominator: float) -> float:
    """numerator / denominator; a non-zero step against a zero denominator is infinitely large (t = 0)."""
    if denominator > 0:
        return numerator / denominator
    if numerator > 0:
        return float("inf")
    return 0.0


def _symmetric(M: torch.Tensor) -> torch.Tensor:
    """½ (M + Mᵀ)."""
    return 0.5 * (M + M.T)


def _shrink(lam: torch.Tensor, mu: float) -> torch.Tensor:
    """M = μ / (λ + μ), with its limit 1 where λ = μ = 0 (nothing to remove), so μ = 0 is allowed."""
    if mu == 0:
        return (lam == 0).to(F64)
    return mu / (lam + mu)


def _covariant_ceiling_bound(pencil: Pencil) -> float:
    """For a ν-damped output pencil (Yᵀ G″ Y = I, Yᵀ G_h Y = diag γ″): the largest covariant Rayleigh quotient
    yᵀG_h y / yᵀG_t′y over its solved directions, γ″_j / (Yᵀ G_t′ Y)_jj, a lower bound on the covariant top
    eigenvalue. A non-positive denominator means G_t′ is indefinite there (infinite)."""
    Y = pencil.X.to(F64)
    task_part = (Y * (pencil.FX.to(F64) - pencil.nu * Y)).sum(0)  # diag(Yᵀ G_t′ Y)
    gamma = pencil.lam.to(F64)
    bounds = torch.where(task_part > 0, gamma / task_part.clamp_min(1e-300), torch.full_like(gamma, float("inf")))
    bounds = torch.where(gamma > 0, bounds, torch.zeros_like(gamma))
    return float(bounds.max())


def _summarise(mu: float, per_module: dict[str, dict]) -> dict:
    """The whole-model report: per-module reports, their summed predictions and the largest step ratios."""
    rows = per_module.values()
    report = {"mu": mu, "modules": per_module}
    for key in SUMMED_REPORT_KEYS:
        report[key] = sum(row[key] for row in rows)
    for key in ("step_ratio", "input_step_ratio", "task_input_step_ratio", "trust_ratio", "harm_input_step_ratio"):
        report[f"max_{key}"] = max((row[key] for row in rows), default=0.0)
    report["n_scaled"] = sum(row["scale"] < 1.0 for row in rows)
    return report


def scaled_trust_region_repair(
    repairer: ModelRepair, W0: dict, deltas: dict, mu: float, actual_S=None, task_gradient: TaskGradient | None = None
):
    """Rule 11 for the scaled repair: one attempt, which passes when every module's trust ratio is at most 1
    (input-weighted after scaling; curvature norm for the exempt modules) and, if `actual_S` is given, the
    measured dev drift falls. Returns (new_weights, report, setting) like trust_region_repair."""
    S_before = None
    if actual_S is not None:
        S_before = actual_S(None)
    new_weights, report = repairer.repair(W0, deltas, mu, task_gradient=task_gradient)
    S_after = None
    if actual_S is not None:
        S_after = actual_S(new_weights)
    steps_ok = report["max_trust_ratio"] <= 1.0 + TRUST_RATIO_TOLERANCE
    drift_fell = S_after is None or S_after < S_before
    attempt = {
        "damping": "scaled",
        "gamma": repairer.gamma,
        "output_nu_rel": repairer.output_nu_rel,
        "max_step_ratio": report["max_step_ratio"],
        "max_input_step_ratio": report["max_input_step_ratio"],
        "max_task_input_step_ratio": report["max_task_input_step_ratio"],
        "max_trust_ratio": report["max_trust_ratio"],
        "S_actual": S_after,
        "S_before": S_before,
        "passed": steps_ok and drift_fell,
    }
    setting = {"damping": "scaled", "gamma": repairer.gamma, "attempts": [attempt]}
    if not attempt["passed"]:
        setting["failed"] = True
    return new_weights, report, setting


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


@dataclass
class TaskGradient:
    """The mean token cross-entropy gradient g at θ1, per module weight: sums[module] / n_tokens."""

    sums: dict[str, torch.Tensor]  # Σ over batches of the summed-token NLL gradient (float32, model's device)
    n_tokens: int

    def dot(self, module: str, step: torch.Tensor) -> float:
        """gᵀδ for one module's step δ (float64)."""
        grad_sum = self.sums[module]
        return float((grad_sum.to(F64) * step.to(grad_sum.device, F64)).sum()) / max(self.n_tokens, 1)


def task_gradient(model, batches, modules) -> TaskGradient:
    """g at the model's current weights over `batches`, for the weights of `modules`. Each batch's bf16 gradient
    is added into a float32 sum on the model's device, so bf16 never accumulates across batches. Leaves every
    parameter with requires_grad False."""
    weights = _enable_weight_grads(model, modules)
    sums = {module: torch.zeros_like(W, dtype=torch.float32) for module, W in weights.items()}
    device = next(iter(weights.values())).device
    n_tokens = 0
    for batch in batches:
        nll, batch_tokens = _summed_token_nll(model, batch, device)
        nll.backward()
        for module, W in weights.items():
            sums[module] += W.grad.float()
            W.grad = None
        n_tokens += batch_tokens
    for W in weights.values():
        W.requires_grad_(False)
    return TaskGradient(sums=sums, n_tokens=n_tokens)


def _grad_diagnostic(g_dot_delta: float, R_pred: float) -> dict[str, float]:
    """{"g_dot_delta", "half_dHd", "ratio" = |gᵀδ| / (½ δᵀĤδ)}; above 0.1 the predicted R is not trustworthy."""
    ratio = float("inf")
    if R_pred > 0:
        ratio = abs(g_dot_delta) / R_pred
    return {"g_dot_delta": g_dot_delta, "half_dHd": R_pred, "ratio": ratio}


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
    return _grad_diagnostic(g_dot_delta, R_pred)


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
