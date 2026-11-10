"""The V/O gauge: a reparameterisation of attention that changes the weights but not the function.

For each layer and KV head g, pick an invertible head_dim × head_dim matrix M_g and set

    v_proj rows of KV head g       ← M_g W_v        (and the V bias, if any)
    o_proj columns of query head h ← W_o M_g⁻¹      for every h with h // n_rep == g

Values are never rotated by RoPE, so the logits are unchanged. Writing M̃ = blockdiag(M_g) over KV heads
and M̄ = blockdiag(M_{h // n_rep}) over query heads, everything else transforms consistently:

    v_proj:  W′ = M̃ W,    ΔW′ = M̃ ΔW,    G′ = M̃⁻ᵀ G M̃⁻¹      (A unchanged)
    o_proj:  W′ = W M̄⁻¹,  ΔW′ = ΔW M̄⁻¹,  A′ = M̄ A M̄ᵀ         (G unchanged)

All other factors are functions of activations the gauge leaves alone. A gauge-invariant repair gives the
same function in both coordinates: compare outputs, never raw weights (CLAUDE.md).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .delta import LoRADelta, model_weights, set_weights
from .modules import decoder, lm_logits, parse_name, target_modules

F64 = torch.float64
Gauges = dict[int, list[torch.Tensor]]  # layer -> [M_g for each KV head]


@dataclass
class AttnShape:
    n_layers: int
    n_heads: int
    n_kv: int
    head_dim: int

    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv

    @classmethod
    def from_config(cls, cfg) -> AttnShape:
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cls(cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, head_dim)


def random_gauge(head_dim: int, cond: float, generator: torch.Generator) -> torch.Tensor:
    """M = Q1 diag(s) Q2 with s log-uniform on [1, cond] (both ends included), so cond(M) = cond."""
    q1, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=generator, dtype=F64))
    q2, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=generator, dtype=F64))
    u = torch.rand(head_dim, generator=generator, dtype=F64)
    u[0], u[-1] = 0.0, 1.0
    return q1 @ torch.diag(cond**u) @ q2


def random_gauges(shape: AttnShape, cond: float = 10.0, seed: int = 0) -> Gauges:
    g = torch.Generator().manual_seed(seed)
    return {
        layer: [random_gauge(shape.head_dim, cond, g) for _ in range(shape.n_kv)] for layer in range(shape.n_layers)
    }


def _kv_block(Ms: list[torch.Tensor]) -> torch.Tensor:
    return torch.block_diag(*Ms)  # M̃


def _head_block(Ms: list[torch.Tensor], n_rep: int) -> torch.Tensor:
    return torch.block_diag(*[Ms[h // n_rep] for h in range(len(Ms) * n_rep)])  # M̄


def gauge_weights(tensors: dict, gauges: Gauges, shape: AttnShape) -> dict:
    """Apply the gauge to per-module tensors that transform like weights (weights or deltas, dense or
    LoRADelta): v_proj ← M̃ X, o_proj ← X M̄⁻¹; every other module unchanged."""
    out = dict(tensors)
    for name, x in tensors.items():
        _, layer, leaf = parse_name(name)
        if layer not in gauges or leaf not in ("v_proj", "o_proj"):
            continue
        if leaf == "v_proj":
            left = _kv_block(gauges[layer])
            out[name] = (
                LoRADelta((left @ x.B.to(F64)).to(x.B.dtype), x.A)
                if isinstance(x, LoRADelta)
                else (left.to(x.device) @ x.to(F64)).to(x.dtype)
            )
        else:
            right = torch.linalg.inv(_head_block(gauges[layer], shape.n_rep))
            out[name] = (
                LoRADelta(x.B, (x.A.to(F64) @ right).to(x.A.dtype))
                if isinstance(x, LoRADelta)
                else (x.to(F64) @ right.to(x.device)).to(x.dtype)
            )
    return out


def reparam_vo(model: nn.Module, gauges: Gauges, shape: AttnShape | None = None) -> nn.Module:
    """Apply the V/O gauge to a model in place (float64 arithmetic). Returns the model."""
    shape = shape or AttnShape.from_config(model.config)
    vo = [n for n in target_modules(model, gauges) if parse_name(n)[2] in ("v_proj", "o_proj")]
    set_weights(model, gauge_weights(model_weights(model, vo), gauges, shape))
    with torch.no_grad():
        for layer, Ms in gauges.items():
            bias = decoder(model).layers[layer].self_attn.v_proj.bias
            if bias is not None:
                bias.copy_((_kv_block(Ms).to(bias.device) @ bias.to(F64)).to(bias.dtype))
    return model


def reparam_copy(model: nn.Module, gauges: Gauges, shape: AttnShape | None = None) -> nn.Module:
    return reparam_vo(copy.deepcopy(model), gauges, shape)


def gauge_transform(gauges: Gauges, shape: AttnShape):
    """A LazyFactors transform: A′ = M̄ A M̄ᵀ on o_proj inputs, G′ = M̃⁻ᵀ G M̃⁻¹ on v_proj outputs."""

    def transform(kind, name, F):
        if kind not in ("A", "G") or not name.endswith(("o_proj", "v_proj")):
            return F
        _, layer, leaf = parse_name(name)
        if layer not in gauges:
            return F
        if kind == "A" and leaf == "o_proj":
            M = _head_block(gauges[layer], shape.n_rep).to(F)
            return M @ F @ M.T
        if kind == "G" and leaf == "v_proj":
            M_inv = torch.linalg.inv(_kv_block(gauges[layer])).to(F)
            return M_inv.T @ F @ M_inv
        return F

    return transform


# ---------------------------------------------------------------- comparing outputs
def _masked_logprobs(model, batch, mask_key):
    dev = next(model.parameters()).device
    ids, att = batch["input_ids"].to(dev), batch["attention_mask"].to(dev)
    mask = batch[mask_key].to(dev).bool() & att.bool()
    hidden = decoder(model)(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    return lm_logits(model, hidden[mask]).double().log_softmax(-1)  # KL can be ~1e-7: float64


def _kl_summary(pairs) -> dict[str, float]:
    kls, max_diff = [], 0.0
    for lp_a, lp_b in pairs:
        kls.append((lp_a.exp() * (lp_a - lp_b)).sum(-1))
        max_diff = max(max_diff, float((lp_a - lp_b).abs().max()))
    k = torch.cat(kls)
    return {
        "kl_mean": float(k.mean()),
        "kl_max": float(k.max()),
        "max_abs_logprob_diff": max_diff,
        "n_positions": int(k.numel()),
    }


@torch.no_grad()
def output_kl(model_a: nn.Module, model_b: nn.Module, batches, mask_key: str = "rows_all") -> dict[str, float]:
    """Per-position KL(a ‖ b) of next-token distributions (mean and max) and the max |Δ log-prob|, over the
    positions in batch[mask_key]. Both models must fit on the device together."""
    return _kl_summary(
        (_masked_logprobs(model_a, b, mask_key), _masked_logprobs(model_b, b, mask_key)) for b in batches
    )


@torch.no_grad()
def logprob_table(model: nn.Module, batches, mask_key: str = "rows_all") -> list[torch.Tensor]:
    """Log-probs at the masked positions of every batch, on the CPU. With kl_to_table this compares two
    weight settings of one model object in turn, so only one copy is on the GPU."""
    return [_masked_logprobs(model, b, mask_key).cpu() for b in batches]


@torch.no_grad()
def kl_to_table(table: list[torch.Tensor], model: nn.Module, batches, mask_key: str = "rows_all") -> dict[str, float]:
    """output_kl between a stored table (as model a) and `model` on the same batches."""
    return _kl_summary((lp_a, _masked_logprobs(model, b, mask_key).cpu()) for lp_a, b in zip(table, batches))
