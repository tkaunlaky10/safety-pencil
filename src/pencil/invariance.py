"""The V/O gauge: an exact reparameterisation of attention that changes the weights but not the function.

For each layer and KV head g, pick an invertible head_dim × head_dim matrix M_g and set

    v_proj rows of group g        ← M_g @ W_v            (and the V bias, if any)
    o_proj columns of query head h ← W_o @ M_g⁻¹         for every h with h // n_rep == g

Values are never rotated by RoPE, so this commutes with attention and the model's logits are unchanged.
The factors and the fine-tune's update transform with it:

    o_proj:  A′ = M̄ A M̄ᵀ, ΔW′ = ΔW M̄⁻¹      (M̄ = block-diag of M_{h // n_rep} over query heads)
    v_proj:  G′ = M̃⁻ᵀ G M̃⁻¹, ΔW′ = M̃ ΔW      (M̃ = block-diag of M_g over KV heads)

Every other factor is a function of activations the gauge leaves unchanged. A gauge-invariant repair
gives the same *function* in both coordinates (CLAUDE.md: compare logits or KL, never raw weights;
range restriction means weights may differ along exactly-null directions).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass

import torch
from torch import nn

from .factors import Factors, _decoder, _logits, parse_name


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
    def from_config(cls, cfg) -> "AttnShape":
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cls(cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, hd)


Gauges = dict[int, list[torch.Tensor]]          # layer -> [M_g for each KV head]


def random_gauge(head_dim: int, cond: float, generator: torch.Generator) -> torch.Tensor:
    """M = Q1 diag(s) Q2 with s log-uniform on [1, cond] (both ends included), so cond(M) = cond."""
    q1, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=generator, dtype=torch.float64))
    q2, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=generator, dtype=torch.float64))
    u = torch.rand(head_dim, generator=generator, dtype=torch.float64)
    u[0], u[-1] = 0.0, 1.0
    s = cond ** u
    return q1 @ torch.diag(s) @ q2


def random_gauges(shape: AttnShape, cond: float = 10.0, seed: int = 0, layers=None) -> Gauges:
    g = torch.Generator().manual_seed(seed)
    layers = range(shape.n_layers) if layers is None else layers
    return {l: [random_gauge(shape.head_dim, cond, g) for _ in range(shape.n_kv)] for l in layers}


def _block_diag_kv(Ms: list[torch.Tensor]) -> torch.Tensor:
    return torch.block_diag(*Ms)


def _block_diag_heads(Ms: list[torch.Tensor], n_rep: int) -> torch.Tensor:
    return torch.block_diag(*[Ms[h // n_rep] for h in range(len(Ms) * n_rep)])


def reparam_vo(model: nn.Module, gauges: Gauges, shape: AttnShape | None = None) -> nn.Module:
    """Apply the V/O gauge in place (arithmetic in float64, cast back). Returns the model."""
    shape = shape or AttnShape.from_config(model.config)
    hd, n_rep = shape.head_dim, shape.n_rep
    layers = _decoder_layers(model)
    with torch.no_grad():
        for l, Ms in gauges.items():
            attn = layers[l].self_attn
            v, o = attn.v_proj, attn.o_proj
            Wv, Wo = v.weight.double(), o.weight.double()
            for g, M in enumerate(Ms):
                M = M.to(Wv.device)
                Mi = torch.linalg.inv(M)
                rows = slice(g * hd, (g + 1) * hd)
                Wv[rows] = M @ Wv[rows]
                if v.bias is not None:
                    vb = v.bias.double()
                    vb[rows] = M @ vb[rows]
                    v.bias.copy_(vb.to(v.bias.dtype))
                for h in range(g * n_rep, (g + 1) * n_rep):
                    cols = slice(h * hd, (h + 1) * hd)
                    Wo[:, cols] = Wo[:, cols] @ Mi
            v.weight.copy_(Wv.to(v.weight.dtype))
            o.weight.copy_(Wo.to(o.weight.dtype))
    return model


def reparam_copy(model: nn.Module, gauges: Gauges, shape: AttnShape | None = None) -> nn.Module:
    return reparam_vo(copy.deepcopy(model), gauges, shape)


def transform_delta(deltas: dict[str, torch.Tensor], gauges: Gauges, shape: AttnShape) -> dict[str, torch.Tensor]:
    """ΔW_v ← M̃ ΔW_v and ΔW_o ← ΔW_o M̄⁻¹; every other module unchanged."""
    from .delta import LoRADelta
    out = dict(deltas)
    for name, d in deltas.items():
        _, l, leaf = parse_name(name)
        if l not in gauges or leaf not in ("v_proj", "o_proj"):
            continue
        if leaf == "v_proj":
            Mt = _block_diag_kv(gauges[l]).to(d.device)
            if isinstance(d, LoRADelta):
                out[name] = LoRADelta((Mt @ d.B.double()).to(d.dtype), d.A)
            else:
                out[name] = (Mt @ d.double()).to(d.dtype)
        else:
            Mbi = torch.linalg.inv(_block_diag_heads(gauges[l], shape.n_rep)).to(d.device)
            if isinstance(d, LoRADelta):
                out[name] = LoRADelta(d.B, (d.A.double() @ Mbi).to(d.dtype))
            else:
                out[name] = (d.double() @ Mbi).to(d.dtype)
    return out


def transform_factors(f: Factors, gauges: Gauges, shape: AttnShape) -> Factors:
    """Analytic factors of the reparameterised model: A_o′ = M̄ A_o M̄ᵀ, G_v′ = M̃⁻ᵀ G_v M̃⁻¹."""
    A = {k: dict(v) for k, v in f.A.items()}
    G = {k: dict(v) for k, v in f.G.items()}
    for name in f.modules:
        _, l, leaf = parse_name(name)
        if l not in gauges:
            continue
        if leaf == "o_proj":
            key = f.a_key[name]
            Mbar = _block_diag_heads(gauges[l], shape.n_rep)
            A[key] = {rs: Mbar.to(t) @ t @ Mbar.to(t).T for rs, t in f.A[key].items()}
        elif leaf == "v_proj":
            Mi = torch.linalg.inv(_block_diag_kv(gauges[l]))
            G[name] = {rs: Mi.to(t).T @ t @ Mi.to(t) for rs, t in f.G[name].items()}
    return Factors(A=A, G=G, a_key=dict(f.a_key), n_rows=dict(f.n_rows), n_loss=f.n_loss, n_seqs=f.n_seqs,
                   D={}, meta={**f.meta, "transformed": "vo_gauge"})


def _decoder_layers(model: nn.Module):
    dec = getattr(model, "model", None)
    return (dec if dec is not None else model.get_decoder()).layers


# ---------------------------------------------------------------- functional comparisons
@torch.no_grad()
def output_kl(model_a: nn.Module, model_b: nn.Module, batches, mask_key: str = "rows_all") -> dict[str, float]:
    """Mean and max per-position KL(a ‖ b) of next-token distributions, and the max |Δlogit|, over the
    positions in batch[mask_key] (the user prompt + response by default)."""
    dev = next(model_a.parameters()).device
    kls, max_logit = [], 0.0
    for b in batches:
        ids, att = b["input_ids"].to(dev), b["attention_mask"].to(dev)
        m = b[mask_key].to(dev).bool() & att.bool()
        la = _masked_logits(model_a, ids, att, m).double()      # KL can be ~1e-7: fp32 log-probs aren't enough
        lb = _masked_logits(model_b, ids, att, m).double()
        pa = la.log_softmax(-1)
        kls.append((pa.exp() * (pa - lb.log_softmax(-1))).sum(-1))
        max_logit = max(max_logit, float((la - lb).abs().max()))
    k = torch.cat(kls)
    return {"kl_mean": float(k.mean()), "kl_max": float(k.max()), "max_abs_logit_diff": max_logit,
            "n_positions": int(k.numel())}


def _masked_logits(model, ids, att, mask):
    """Logits at the masked positions only (the LM head on 262k tokens is the memory hog)."""
    h = _decoder(model)(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    return _logits(model, h[mask])


def gauge_transform(gauges: Gauges, shape: AttnShape):
    """A LazyFactors transform applying the analytic gauge to factors on demand:
    A_o′ = M̄ A_o M̄ᵀ (o_proj input), G_v′ = M̃⁻ᵀ G_v M̃⁻¹ (v_proj output); everything else unchanged."""
    def fn(kind, key, F):
        if kind == "D" or not (key.endswith("o_proj") or key.endswith("v_proj")):
            return F
        _, l, leaf = parse_name(key)
        if l not in gauges:
            return F
        if kind == "A" and leaf == "o_proj":
            Mbar = _block_diag_heads(gauges[l], shape.n_rep).to(F)
            return Mbar @ F @ Mbar.T
        if kind == "G" and leaf == "v_proj":
            Mi = torch.linalg.inv(_block_diag_kv(gauges[l])).to(F)
            return Mi.T @ F @ Mi
        return F
    return fn


# transform_delta applies to any per-module tensor that transforms like a weight (v rows, o columns),
# so it also maps θ0's (or the base model's) linear weights into the gauge coordinates
reparam_weights = transform_delta


@torch.no_grad()
def logprob_table(model: nn.Module, batches, mask_key: str = "rows_all") -> list[torch.Tensor]:
    """Next-token log-probabilities at the masked positions of every batch, as float64 CPU tensors. With
    `kl_to_table` this compares two weight settings of one model object one after the other, so only one
    model is on the GPU (Tiny Aya in fp32 is 13 GB)."""
    dev = next(model.parameters()).device
    out = []
    for b in batches:
        ids, att = b["input_ids"].to(dev), b["attention_mask"].to(dev)
        m = b[mask_key].to(dev).bool() & att.bool()
        out.append(_masked_logits(model, ids, att, m).double().log_softmax(-1).cpu())
    return out


@torch.no_grad()
def kl_to_table(table: list[torch.Tensor], model: nn.Module, batches, mask_key: str = "rows_all") -> dict[str, float]:
    """KL(table ‖ model) per position and the max |Δ log-prob|, on the same batches as the table."""
    dev = next(model.parameters()).device
    kls, max_diff = [], 0.0
    for lp_a, b in zip(table, batches):
        ids, att = b["input_ids"].to(dev), b["attention_mask"].to(dev)
        m = b[mask_key].to(dev).bool() & att.bool()
        lp_b = _masked_logits(model, ids, att, m).double().log_softmax(-1).cpu()
        kls.append((lp_a.exp() * (lp_a - lp_b)).sum(-1))
        max_diff = max(max_diff, float((lp_a - lp_b).abs().max()))
    k = torch.cat(kls)
    return {"kl_mean": float(k.mean()), "kl_max": float(k.max()), "max_abs_logprob_diff": max_diff,
            "n_positions": int(k.numel())}
