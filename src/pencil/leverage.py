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
    """The k largest entries of outer(gamma, alpha), without forming the whole outer product."""
    a = alpha.sort(descending=True).values[:k]
    g = gamma.sort(descending=True).values[:k]
    return torch.outer(g, a).flatten().sort(descending=True).values[:k]


def block_leverage(F_h, F_t, n_blocks: int, block: int, other_side_max: float, **gen_eig_kw) -> list[float]:
    """For each diagonal block of one side's factors: its top eigenvalue × the other side's top eigenvalue."""
    out = []
    for i in range(n_blocks):
        s = slice(i * block, (i + 1) * block)
        lam = gen_eig(F_h[s, s], F_t[s, s], **gen_eig_kw).lam
        out.append(float(lam.max()) * other_side_max if lam.numel() else 0.0)
    return out


def leverage_table(
    repairer, modules, *, n_heads: int | None = None, n_kv: int | None = None, head_dim: int | None = None, k: int = 10
) -> dict[str, dict]:
    """Per module: max and top-k sum of λ, the two sides' top eigenvalues and range sizes, using a
    ModelRepair's bases. With head shapes given, adds o_proj per-head and v_proj per-KV-group lower bounds."""
    table = {}
    for n in layer_order(modules):
        b = repairer.basis(n)
        top = top_products(b.alpha, b.gamma, k)
        _, layer, leaf = parse_name(n)
        row = {
            "layer": layer,
            "module": leaf,
            "max": float(top[0]) if top.numel() else 0.0,
            f"top{k}_sum": float(top.sum()),
            "alpha_max": float(b.alpha.max()),
            "gamma_max": float(b.gamma.max()),
            "range_in": b.inp.range_dim,
            "range_out": b.out.range_dim,
        }
        if head_dim is not None and leaf in ("o_proj", "v_proj"):
            A_h, A_t, G_h, G_t = repairer.factors(n)
            # these two sides have no known null vector, so they use the bases' rel_tol_other
            kw = {"damping": repairer.damping, "gamma": repairer.gamma, "rel_tol": repairer.rel_tol_other}
            if leaf == "o_proj" and n_heads:
                row["per_head_lower_bound"] = block_leverage(A_h, A_t, n_heads, head_dim, row["gamma_max"], **kw)
            if leaf == "v_proj" and n_kv:
                row["per_kv_group_lower_bound"] = block_leverage(G_h, G_t, n_kv, head_dim, row["alpha_max"], **kw)
        table[n] = row
    return table


def top_output_direction(b: LayerBasis) -> torch.Tensor:
    """The output-space direction (column of Y) of the module's largest γ, unit norm. For o_proj and down_proj
    it lives in the residual stream, like the refusal direction."""
    y = b.out.X[:, int(b.gamma.argmax())]
    return y / y.norm()


def refusal_cosine(y: torch.Tensor, r: torch.Tensor, n_null: int = 1000, seed: int = 0) -> dict[str, float]:
    """|cos(y, r)| and its tail probability against random unit vectors (the P5 null distribution)."""
    y = y.double() / y.double().norm()
    r = r.double().to(y.device) / r.double().norm()
    c = float((y @ r).abs())
    z = torch.randn(n_null, y.numel(), generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    null = ((z / z.norm(dim=1, keepdim=True)).to(y.device) @ r).abs()
    return {
        "abs_cos": c,
        "p_value": float(((null >= c).sum() + 1) / (n_null + 1)),
        "null_mean": float(null.mean()),
        "null_q95": float(null.quantile(0.95)),
    }


def top_fraction(table: dict[str, dict], frac: float = 0.1, key: str = "max") -> list[str]:
    """The top `frac` of modules by leverage."""
    ranked = sorted(table, key=lambda n: table[n][key], reverse=True)
    return ranked[: max(1, round(frac * len(ranked)))]


def random_fraction(table: dict[str, dict], frac: float = 0.1, seed: int = 0) -> list[str]:
    """A random `frac` of modules: the control for the targeted repair (P5)."""
    names = sorted(table)
    idx = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed))[: max(1, round(frac * len(names)))]
    return [names[i] for i in idx.tolist()]
