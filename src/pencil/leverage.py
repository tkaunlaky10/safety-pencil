"""Leverage: where a unit of task-curvature-normalised change moves safety most.

A module with pencil eigenvalues α_i (input side) and γ_j (output side) has leverage values λ_ji = γ_j α_i,
the safety curvature per unit of task curvature along (Y_j, X_i). We report per module the maximum and the
top-10 sum, plus per-head values for o_proj (input side restricted to one head's columns) and per-KV-group
values for v_proj (output side restricted to one group's rows). Those come from a principal block of the
pencil, so by interlacing they are lower bounds on the module's top value; say so in any figure
(tests/test_math.py::test_per_head_leverage_is_a_lower_bound).
"""

from __future__ import annotations

import torch

from .geneig import gen_eig
from .modules import layer_order, parse_name
from .repair import LayerBasis


def top_products(alpha: torch.Tensor, gamma: torch.Tensor, k: int = 10) -> torch.Tensor:
    """The k largest λ_ji = γ_j α_i, descending, without forming the whole outer product (only the top k of
    each side can contribute)."""
    top_alpha = alpha.sort(descending=True).values[:k]
    top_gamma = gamma.sort(descending=True).values[:k]
    products = torch.outer(top_gamma, top_alpha).flatten()
    return products.sort(descending=True).values[:k]


def block_leverage(F_h, F_t, n_blocks: int, block: int, other_side_max: float, **gen_eig_kw) -> list[float]:
    """For each diagonal block of one side's factors: its top pencil eigenvalue × the other side's top
    eigenvalue (0 for a block with an empty range). `gen_eig_kw` go to gen_eig."""
    bounds = []
    for i in range(n_blocks):
        rows = slice(i * block, (i + 1) * block)
        lam = gen_eig(F_h[rows, rows], F_t[rows, rows], **gen_eig_kw).lam
        bound = 0.0
        if lam.numel():
            bound = float(lam.max()) * other_side_max
        bounds.append(bound)
    return bounds


def leverage_table(
    repairer, modules, *, n_heads: int | None = None, n_kv: int | None = None, head_dim: int | None = None, k: int = 10
) -> dict[str, dict]:
    """Per module: max and top-k sum of λ, the two sides' top eigenvalues and range sizes, using a
    ModelRepair's bases. With head shapes given, adds o_proj per-head and v_proj per-KV-group lower bounds."""
    table = {}
    for module in layer_order(modules):
        row = _module_row(repairer.basis(module), module, k)
        if head_dim is not None and row["module"] in ("o_proj", "v_proj"):
            _add_head_bounds(row, repairer, module, n_heads, n_kv, head_dim)
        table[module] = row
    return table


def _module_row(basis: LayerBasis, module: str, k: int) -> dict:
    """The leverage summary of one module's basis."""
    top = top_products(basis.alpha, basis.gamma, k)
    _, layer, module_leaf = parse_name(module)
    largest = 0.0
    if top.numel():
        largest = float(top[0])
    return {
        "layer": layer,
        "module": module_leaf,
        "max": largest,
        f"top{k}_sum": float(top.sum()),
        "alpha_max": float(basis.alpha.max()),
        "gamma_max": float(basis.gamma.max()),
        "range_in": basis.inp.range_dim,
        "range_out": basis.out.range_dim,
        "jitter_in": basis.inp.jitter,  # ε × mean diagonal the fast top-k solver needed (0 elsewhere)
        "jitter_out": basis.out.jitter,
    }


def _add_head_bounds(row: dict, repairer, module: str, n_heads, n_kv, head_dim: int) -> None:
    """Add o_proj per-head (input-side blocks) or v_proj per-KV-group (output-side blocks) lower bounds."""
    A_h, A_t, G_h, G_t = repairer.factors(module)
    # These two sides have no known null vector, so they use the bases' rel_tol_other.
    pencil_options = {"damping": repairer.damping, "gamma": repairer.gamma, "rel_tol": repairer.rel_tol_other}
    if row["module"] == "o_proj" and n_heads:
        row["per_head_lower_bound"] = block_leverage(A_h, A_t, n_heads, head_dim, row["gamma_max"], **pencil_options)
    if row["module"] == "v_proj" and n_kv:
        row["per_kv_group_lower_bound"] = block_leverage(G_h, G_t, n_kv, head_dim, row["alpha_max"], **pencil_options)


def top_output_direction(basis: LayerBasis) -> torch.Tensor:
    """The output-space direction (column of Y) of the module's largest γ, unit norm. For o_proj and down_proj
    it lives in the residual stream, like the refusal direction."""
    y = basis.out.X[:, int(basis.gamma.argmax())]
    return y / y.norm()


def refusal_cosine(y: torch.Tensor, r: torch.Tensor, n_null: int = 1000, seed: int = 0) -> dict[str, float]:
    """|cos(y, r)| and its tail probability against n_null random unit vectors (the P5 null distribution)."""
    y = y.double() / y.double().norm()
    r = r.double().to(y.device) / r.double().norm()
    abs_cos = float((y @ r).abs())
    generator = torch.Generator().manual_seed(seed)
    random_vectors = torch.randn(n_null, y.numel(), generator=generator, dtype=torch.float64)
    random_units = random_vectors / random_vectors.norm(dim=1, keepdim=True)
    null_abs_cos = (random_units.to(y.device) @ r).abs()
    return {
        "abs_cos": abs_cos,
        "p_value": float(((null_abs_cos >= abs_cos).sum() + 1) / (n_null + 1)),
        "null_mean": float(null_abs_cos.mean()),
        "null_q95": float(null_abs_cos.quantile(0.95)),
    }


def top_fraction(table: dict[str, dict], frac: float = 0.1, key: str = "max") -> list[str]:
    """The top `frac` of modules by leverage (at least one)."""
    ranked = sorted(table, key=lambda module: table[module][key], reverse=True)
    n_keep = max(1, round(frac * len(ranked)))
    return ranked[:n_keep]


def random_fraction(table: dict[str, dict], frac: float = 0.1, seed: int = 0) -> list[str]:
    """A random `frac` of modules (at least one): the control for the targeted repair (P5)."""
    names = sorted(table)
    n_keep = max(1, round(frac * len(names)))
    permutation = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed))
    return [names[i] for i in permutation[:n_keep].tolist()]
