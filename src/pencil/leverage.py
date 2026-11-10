"""Leverage: where a unit of task-curvature-normalised change moves safety most.

For a module with pencil eigenvalues α_i (input side) and γ_j (output side), the leverage values are
λ_ji = γ_j α_i: the safety curvature per unit of task curvature along the (Y_j, X_i) direction. We report
per module the maximum and the sum of the top 10, per o_proj head (input side restricted to that head's
columns) and per KV group for v_proj (output side restricted to that group's rows).

Per-head and per-group values come from a principal block of the factor pencil, so by Courant–Fischer
(interlacing) they are lower bounds on the module's top value (test_math.test_per_head_leverage_is_a_lower_bound).
Say so in any figure.
"""
from __future__ import annotations

import torch

from .factors import parse_name
from .geneig import gen_eig
from .repair import LayerBasis


def top_products(alpha: torch.Tensor, gamma: torch.Tensor, k: int = 10) -> torch.Tensor:
    """The k largest values of outer(gamma, alpha), without forming the full outer product."""
    a = alpha.sort(descending=True).values[:k]
    g = gamma.sort(descending=True).values[:k]
    return torch.outer(g, a).flatten().sort(descending=True).values[:k]


def module_leverage(b: LayerBasis, k: int = 10) -> dict[str, float]:
    tp = top_products(b.alpha, b.gamma, k)
    return {"max": float(tp[0]) if tp.numel() else 0.0, f"top{k}_sum": float(tp.sum()),
            "alpha_max": float(b.alpha.max()), "gamma_max": float(b.gamma.max()),
            "range_in": b.inp.range_dim, "range_out": b.out.range_dim}


def head_leverage_o(A_h, A_t, gamma_max: float, n_heads: int, head_dim: int, *, damping="covariant",
                    gamma=1e-3, rel_tol=1e-10) -> list[float]:
    """Per query head of o_proj: top α on that head's input block × the module's top γ (a lower bound)."""
    out = []
    for h in range(n_heads):
        s = slice(h * head_dim, (h + 1) * head_dim)
        p = gen_eig(A_h[s, s], A_t[s, s], damping, gamma, None, rel_tol)
        out.append(float(p.lam.max()) * gamma_max if p.lam.numel() else 0.0)
    return out


def group_leverage_v(G_h, G_t, alpha_max: float, n_kv: int, head_dim: int, *, damping="covariant",
                     gamma=1e-3, rel_tol=1e-10) -> list[float]:
    """Per KV group of v_proj: top γ on that group's output block × the module's top α (a lower bound)."""
    out = []
    for g in range(n_kv):
        s = slice(g * head_dim, (g + 1) * head_dim)
        p = gen_eig(G_h[s, s], G_t[s, s], damping, gamma, None, rel_tol)
        out.append(float(p.lam.max()) * alpha_max if p.lam.numel() else 0.0)
    return out


def leverage_table(repairer, modules, *, n_heads: int | None = None, n_kv: int | None = None,
                   head_dim: int | None = None, k: int = 10) -> dict[str, dict]:
    """Leverage for every module, using a ModelRepair's cached bases and settings. With head shapes given,
    adds per-head values for o_proj and per-KV-group values for v_proj."""
    out = {}
    for n in repairer.order(modules):
        b = repairer.basis(n)
        row = module_leverage(b, k)
        _, layer, leaf = parse_name(n)
        row.update(layer=layer, module=leaf)
        if head_dim is not None:
            A_h, A_t, G_h, G_t = repairer.factors(n)
            # o_proj input and v_proj output have no known null vector: use the same tolerance as their bases
            kw = dict(damping=repairer.damping, gamma=repairer.gamma, rel_tol=repairer.rel_tol_other)
            if leaf == "o_proj" and n_heads:
                row["per_head_lower_bound"] = head_leverage_o(A_h, A_t, row["gamma_max"], n_heads, head_dim, **kw)
            if leaf == "v_proj" and n_kv:
                row["per_kv_group_lower_bound"] = group_leverage_v(G_h, G_t, row["alpha_max"], n_kv, head_dim, **kw)
        out[n] = row
    return out


def top_output_direction(b: LayerBasis) -> torch.Tensor:
    """Output-space direction of the module's top pair: the column of Y with the largest γ, unit norm.
    For o_proj and down_proj this lives in the residual stream, where the refusal direction lives."""
    y = b.out.X[:, int(b.gamma.argmax())]
    return y / y.norm()


def refusal_cosine(y: torch.Tensor, r: torch.Tensor, n_null: int = 1000, seed: int = 0) -> dict[str, float]:
    """|cos(y, r)| and its tail probability under random unit vectors (the P5 null distribution)."""
    y = y.double() / y.double().norm()
    r = r.double().to(y.device) / r.double().norm()
    c = float((y @ r).abs())
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_null, y.numel(), generator=g, dtype=torch.float64).to(y.device)
    null = ((z / z.norm(dim=1, keepdim=True)) @ r).abs()
    return {"abs_cos": c, "p_value": float(((null >= c).sum() + 1) / (n_null + 1)),
            "null_mean": float(null.mean()), "null_q95": float(null.quantile(0.95))}


def rank_modules(table: dict[str, dict], key: str = "max") -> list[str]:
    return sorted(table, key=lambda n: table[n][key], reverse=True)


def top_fraction(table: dict[str, dict], frac: float = 0.1, key: str = "max") -> list[str]:
    ranked = rank_modules(table, key)
    return ranked[:max(1, round(frac * len(ranked)))]


def random_fraction(table: dict[str, dict], frac: float = 0.1, seed: int = 0) -> list[str]:
    """The random-module control for the targeted repair (P5)."""
    names = sorted(table)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(names), generator=g)[:max(1, round(frac * len(names)))]
    return [names[i] for i in idx.tolist()]
