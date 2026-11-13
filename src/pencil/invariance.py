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
GAUGED_LEAVES = ("v_proj", "o_proj")


@dataclass
class AttnShape:
    """The attention dimensions the gauge needs."""

    n_layers: int
    n_heads: int  # query heads
    n_kv: int  # KV heads
    head_dim: int

    @property
    def n_rep(self) -> int:
        """Query heads per KV head: query head h reads KV head h // n_rep."""
        return self.n_heads // self.n_kv

    @classmethod
    def from_config(cls, cfg) -> AttnShape:
        """The shape from a Hugging Face config (head_dim defaults to hidden_size / num_attention_heads)."""
        head_dim = getattr(cfg, "head_dim", None)
        if not head_dim:
            head_dim = cfg.hidden_size // cfg.num_attention_heads
        return cls(cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, head_dim)


def random_gauge(head_dim: int, cond: float, generator: torch.Generator) -> torch.Tensor:
    """A random M = Q1 diag(s) Q2 with orthogonal Q1, Q2 and s log-uniform on [1, cond] (both ends included),
    so the condition number of M is exactly cond."""
    q1, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=generator, dtype=F64))
    q2, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=generator, dtype=F64))
    exponents = torch.rand(head_dim, generator=generator, dtype=F64)
    exponents[0] = 0.0
    exponents[-1] = 1.0
    return q1 @ torch.diag(cond**exponents) @ q2


def random_gauges(shape: AttnShape, cond: float = 10.0, seed: int = 0) -> Gauges:
    """One random_gauge per KV head in every layer, from one seeded generator."""
    generator = torch.Generator().manual_seed(seed)
    gauges = {}
    for layer in range(shape.n_layers):
        gauges[layer] = [random_gauge(shape.head_dim, cond, generator) for _ in range(shape.n_kv)]
    return gauges


def _kv_block(Ms: list[torch.Tensor]) -> torch.Tensor:
    """M̃ = blockdiag(M_g) over KV heads (the v_proj output side)."""
    return torch.block_diag(*Ms)


def _head_block(Ms: list[torch.Tensor], n_rep: int) -> torch.Tensor:
    """M̄ = blockdiag(M_{h // n_rep}) over query heads (the o_proj input side)."""
    n_query_heads = len(Ms) * n_rep
    per_query_head = [Ms[h // n_rep] for h in range(n_query_heads)]
    return torch.block_diag(*per_query_head)


def gauge_weights(tensors: dict, gauges: Gauges, shape: AttnShape) -> dict:
    """Apply the gauge to per-module tensors that transform like weights (weights or deltas, dense or
    LoRADelta): v_proj ← M̃ W, o_proj ← W M̄⁻¹; every other module unchanged."""
    gauged = dict(tensors)
    for name, tensor in tensors.items():
        _, layer, module_leaf = parse_name(name)
        if layer not in gauges or module_leaf not in GAUGED_LEAVES:
            continue
        if module_leaf == "v_proj":
            gauged[name] = _multiply_left(_kv_block(gauges[layer]), tensor)
        else:
            right = torch.linalg.inv(_head_block(gauges[layer], shape.n_rep))
            gauged[name] = _multiply_right(tensor, right)
    return gauged


def _multiply_left(left: torch.Tensor, tensor):
    """left @ W in float64, kept in W's dtype. For a LoRADelta (W = B A) only B changes."""
    if isinstance(tensor, LoRADelta):
        return LoRADelta((left @ tensor.B.to(F64)).to(tensor.B.dtype), tensor.A)
    return (left.to(tensor.device) @ tensor.to(F64)).to(tensor.dtype)


def _multiply_right(tensor, right: torch.Tensor):
    """W @ right in float64, kept in W's dtype. For a LoRADelta (W = B A) only A changes."""
    if isinstance(tensor, LoRADelta):
        return LoRADelta(tensor.B, (tensor.A.to(F64) @ right).to(tensor.A.dtype))
    return (tensor.to(F64) @ right.to(tensor.device)).to(tensor.dtype)


def reparam_vo(model: nn.Module, gauges: Gauges, shape: AttnShape | None = None) -> nn.Module:
    """Apply the V/O gauge to a model in place (float64 arithmetic). Returns the model."""
    if shape is None:
        shape = AttnShape.from_config(model.config)
    gauged_modules = []
    for name in target_modules(model, layers=gauges):
        if parse_name(name)[2] in GAUGED_LEAVES:
            gauged_modules.append(name)
    set_weights(model, gauge_weights(model_weights(model, gauged_modules), gauges, shape))
    with torch.no_grad():
        for layer, Ms in gauges.items():
            bias = decoder(model).layers[layer].self_attn.v_proj.bias
            if bias is not None:
                bias.copy_((_kv_block(Ms).to(bias.device) @ bias.to(F64)).to(bias.dtype))
    return model


def reparam_copy(model: nn.Module, gauges: Gauges, shape: AttnShape | None = None) -> nn.Module:
    """A gauged deep copy of the model (the original is untouched)."""
    return reparam_vo(copy.deepcopy(model), gauges, shape)


def gauge_transform(gauges: Gauges, shape: AttnShape):
    """A LazyFactors transform giving the factors of the gauged model: A′ = M̄ A M̄ᵀ on o_proj inputs and
    G′ = M̃⁻ᵀ G M̃⁻¹ on v_proj outputs; every other factor unchanged."""

    def transform(kind, name, F):
        # Checked before parse_name, which rejects input factor keys such as "model.layers.3.input".
        if kind not in ("A", "G") or not name.endswith(GAUGED_LEAVES):
            return F
        _, layer, module_leaf = parse_name(name)
        if layer not in gauges:
            return F
        if kind == "A" and module_leaf == "o_proj":
            M = _head_block(gauges[layer], shape.n_rep).to(F)
            return M @ F @ M.T
        if kind == "G" and module_leaf == "v_proj":
            M_inv = torch.linalg.inv(_kv_block(gauges[layer])).to(F)
            return M_inv.T @ F @ M_inv
        return F

    return transform


# ---------------------------------------------------------------- comparing outputs
def _masked_logprobs(model: nn.Module, batch: dict, mask_key: str) -> torch.Tensor:
    """Next-token log-probs (positions × vocab, float64) at the positions in batch[mask_key]."""
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    mask = batch[mask_key].to(device).bool() & attention_mask.bool()
    hidden = decoder(model)(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    return lm_logits(model, hidden[mask]).double().log_softmax(-1)  # float64: the KL can be ~1e-7


def _kl_summary(logprob_pairs, per_batch: bool = False) -> dict:
    """Mean and max per-position KL(a ‖ b) and the max |log p_a − log p_b| over (log p_a, log p_b) pairs.
    per_batch adds each batch's summed KL and position count ("kl_sum_per_batch", "n_per_batch"): with one
    sequence per batch, what a paired bootstrap over prompts needs.

    The pairs are consumed one at a time, so a generator keeps only one batch of vocab-sized tensors."""
    kls = []
    max_diff = 0.0
    for logprobs_a, logprobs_b in logprob_pairs:
        kls.append((logprobs_a.exp() * (logprobs_a - logprobs_b)).sum(-1))
        max_diff = max(max_diff, float((logprobs_a - logprobs_b).abs().max()))
    all_kls = torch.cat(kls)
    summary = {
        "kl_mean": float(all_kls.mean()),
        "kl_max": float(all_kls.max()),
        "max_abs_logprob_diff": max_diff,
        "n_positions": int(all_kls.numel()),
    }
    if per_batch:
        summary["kl_sum_per_batch"] = [float(batch_kls.sum()) for batch_kls in kls]
        summary["n_per_batch"] = [int(batch_kls.numel()) for batch_kls in kls]
    return summary


def _paired_logprobs(model_a: nn.Module, model_b: nn.Module, batches, mask_key: str):
    """Yield (log p_a, log p_b) batch by batch."""
    for batch in batches:
        yield _masked_logprobs(model_a, batch, mask_key), _masked_logprobs(model_b, batch, mask_key)


def _table_and_model_logprobs(table: list[torch.Tensor], model: nn.Module, batches, mask_key: str):
    """Yield (stored log p, the model's log p on the CPU) batch by batch."""
    for logprobs_a, batch in zip(table, batches):
        yield logprobs_a, _masked_logprobs(model, batch, mask_key).cpu()


@torch.no_grad()
def output_kl(
    model_a: nn.Module, model_b: nn.Module, batches, mask_key: str = "rows_all", per_batch: bool = False
) -> dict:
    """Per-position KL(a ‖ b) of next-token distributions (mean and max) and the max |Δ log-prob|, over the
    positions in batch[mask_key] (per batch too with per_batch). Both models must fit on the device together."""
    return _kl_summary(_paired_logprobs(model_a, model_b, batches, mask_key), per_batch)


@torch.no_grad()
def logprob_table(model: nn.Module, batches, mask_key: str = "rows_all") -> list[torch.Tensor]:
    """Log-probs at the masked positions of every batch, on the CPU. With kl_to_table this compares two
    weight settings of one model object in turn, so only one copy is on the GPU."""
    return [_masked_logprobs(model, batch, mask_key).cpu() for batch in batches]


@torch.no_grad()
def kl_to_table(table: list[torch.Tensor], model: nn.Module, batches, mask_key: str = "rows_all") -> dict[str, float]:
    """output_kl between a stored table (as model a) and `model` on the same batches."""
    return _kl_summary(_table_and_model_logprobs(table, model, batches, mask_key))
