"""Record the architecture facts the method depends on in configs/model_facts.json (Prompt 0 step 3).

Sizes and settings come from the model's config. The structural facts are measured, not read from source
code: the script builds a tiny randomly initialised copy of the model (same class and config flags, small
sizes), runs one random sequence through it, and watches what the layers receive.

  shared input    q, k, v, gate and up all receive the same tensor, so they share one K-FAC input factor
                  (a parallel block: attention and MLP read the same input_layernorm output)
  null vector     that shared input is orthogonal to 1/w, w = the input_layernorm weight (the known null
                  vector the repair projects out)
  plain V         doubling v_proj's weight doubles the o_proj input: V reaches o_proj linearly, with no bias
                  and no norm, which the V/O gauge demo needs
  KV grouping     changing one KV head's value rows moves exactly the query heads that read it
                  (expected: query head h reads KV head h // n_rep)

    python scripts/00_model_facts.py            # needs HF_TOKEN in .env (gated model)
    python scripts/00_model_facts.py --dry-run  # tiny offline model, writes outputs/dryrun/model_facts.json
"""

from __future__ import annotations

import argparse
import copy
from functools import partial

import torch

from pencil.config import REPO_ROOT
from pencil.modules import target_modules
from pencil.runtime import common_args, setup, write_json

# The five linear layers that read the decoder layer's input in a parallel block.
SHARED_INPUT_MODULES = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
}

# The tiny copy: small sizes, the same query-to-KV head ratio, everything else from the real config.
TINY_KV_HEADS = 2
TINY_HEAD_DIM = 16
TINY_INTERMEDIATE = 96
TINY_LAYERS = 2
TINY_VOCAB = 128
SEQ_LEN = 12
SEED = 0

# Relative errors below this count as "equal" for float32 on the tiny model.
TOLERANCE = 1e-5


# ---------------------------------------------------------------- the tiny copy


def tiny_copy(config):
    """The same config class and flags with small sizes, keeping the query-to-KV head ratio."""
    n_rep = config.num_attention_heads // config.num_key_value_heads
    tiny = copy.deepcopy(config)
    tiny.num_key_value_heads = TINY_KV_HEADS
    tiny.num_attention_heads = TINY_KV_HEADS * n_rep
    tiny.head_dim = TINY_HEAD_DIM
    tiny.hidden_size = tiny.num_attention_heads * TINY_HEAD_DIM
    tiny.intermediate_size = TINY_INTERMEDIATE
    tiny.num_hidden_layers = TINY_LAYERS
    tiny.vocab_size = TINY_VOCAB
    tiny.pad_token_id = 0  # must be a valid row of the smaller embedding
    if getattr(config, "layer_types", None):
        tiny.layer_types = list(config.layer_types[:TINY_LAYERS])
    return tiny


def build_tiny_model(config):
    """A randomly initialised float32 model of the config's class, with a non-trivial LayerNorm weight."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(tiny_copy(config), attn_implementation="eager").float().eval()
    first_layer = model.model.layers[0]
    with torch.no_grad():
        # with the default all-ones weight, 1/w would be the trivial all-ones vector
        first_layer.input_layernorm.weight.uniform_(0.5, 1.5)
    return model


def random_input(model) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(1, model.config.vocab_size, (1, SEQ_LEN), generator=generator)


# ---------------------------------------------------------------- watching layer inputs


def _store_input(store: dict, name: str, module, args) -> None:
    """Forward pre-hook: keep a copy of the tensor `module` receives."""
    store[name] = args[0].detach().clone()


def layer_inputs(model, input_ids, module_paths: dict[str, str]) -> dict[str, torch.Tensor]:
    """Run `input_ids` once and return the input each named submodule of the first layer received."""
    first_layer = model.model.layers[0]
    received = {}
    handles = []
    for name, path in module_paths.items():
        hook = partial(_store_input, received, name)
        handles.append(first_layer.get_submodule(path).register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return received


def o_proj_input(model, input_ids) -> torch.Tensor:
    """The first layer's o_proj input: the attention heads' outputs side by side, (1, seq, n_heads · head_dim)."""
    return layer_inputs(model, input_ids, {"o_proj": "self_attn.o_proj"})["o_proj"]


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).norm() / expected.norm()).item()


# ---------------------------------------------------------------- the four checks


def check_shared_input(model, input_ids) -> dict:
    """Do q, k, v, gate and up receive the same tensor? Returns the largest relative difference from q_proj's."""
    received = layer_inputs(model, input_ids, SHARED_INPUT_MODULES)
    reference = received["q_proj"]
    largest = max(relative_error(tensor, reference) for tensor in received.values())
    return {"shared": largest < TOLERANCE, "max_rel_difference": largest}


def check_null_vector(model, input_ids) -> dict:
    """Is the shared input orthogonal to 1/w? Returns the largest |cos(a, 1/w)| over positions."""
    first_layer = model.model.layers[0]
    a = layer_inputs(model, input_ids, {"q_proj": "self_attn.q_proj"})["q_proj"][0]  # (seq, hidden)
    u = 1.0 / first_layer.input_layernorm.weight.detach()
    cosines = (a @ u).abs() / (a.norm(dim=-1) * u.norm())
    largest = cosines.max().item()
    return {"orthogonal": largest < TOLERANCE, "max_abs_cos": largest}


def check_plain_v(model, input_ids) -> dict:
    """Does V reach o_proj linearly? Doubles v_proj's weight (bias set to zero if there is one) and compares
    the o_proj input with twice the original. A norm or other non-linearity on V makes the error large."""
    v_proj = model.model.layers[0].self_attn.v_proj
    saved_weight = v_proj.weight.detach().clone()
    saved_bias = None if v_proj.bias is None else v_proj.bias.detach().clone()
    try:
        with torch.no_grad():
            if v_proj.bias is not None:
                v_proj.bias.zero_()
            before = o_proj_input(model, input_ids)
            v_proj.weight.mul_(2.0)
            after = o_proj_input(model, input_ids)
    finally:
        with torch.no_grad():
            v_proj.weight.copy_(saved_weight)
            if saved_bias is not None:
                v_proj.bias.copy_(saved_bias)
    error = relative_error(after, 2.0 * before)
    return {"linear": error < TOLERANCE, "rel_error": error}


def measure_kv_grouping(model, input_ids) -> list[int | None]:
    """For each query head, the KV head it reads: perturb one KV head's value rows at a time and see which query
    heads' slices of the o_proj input move. None marks a query head that no KV head moved."""
    config = model.config
    head_dim = config.head_dim
    v_proj = model.model.layers[0].self_attn.v_proj
    baseline = o_proj_input(model, input_ids)
    reads: list[int | None] = [None] * config.num_attention_heads
    generator = torch.Generator().manual_seed(SEED)
    for kv_head in range(config.num_key_value_heads):
        rows = slice(kv_head * head_dim, (kv_head + 1) * head_dim)
        saved_rows = v_proj.weight[rows].detach().clone()
        with torch.no_grad():
            v_proj.weight[rows] += torch.randn(saved_rows.shape, generator=generator)
        perturbed = o_proj_input(model, input_ids)
        with torch.no_grad():
            v_proj.weight[rows] = saved_rows
        for query_head in range(config.num_attention_heads):
            cols = slice(query_head * head_dim, (query_head + 1) * head_dim)
            if not torch.allclose(perturbed[..., cols], baseline[..., cols]):
                reads[query_head] = kv_head
    return reads


def describe_grouping(reads: list[int | None], n_rep: int) -> str:
    expected = [query_head // n_rep for query_head in range(len(reads))]
    if reads == expected:
        return f"query head h reads KV head h // {n_rep}"
    return f"measured: query head h reads KV head {reads}[h]"


# ---------------------------------------------------------------- the facts file


def config_facts(config) -> dict:
    """Sizes and settings straight from the real config."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return {
        "model_type": config.model_type,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": head_dim,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "attention_bias": getattr(config, "attention_bias", None),
        "use_qk_norm": getattr(config, "use_qk_norm", None),
        "layer_types": getattr(config, "layer_types", None),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
        "logit_scale": getattr(config, "logit_scale", None),
        "n_rep": config.num_attention_heads // config.num_key_value_heads,
    }


def facts(spec: str) -> dict:
    """The architecture facts of model `spec`: its config, plus the four checks run on a tiny copy."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(spec)
    model = build_tiny_model(config)
    input_ids = random_input(model)
    first_layer = model.model.layers[0]

    shared = check_shared_input(model, input_ids)
    null_vector = check_null_vector(model, input_ids)
    plain_v = check_plain_v(model, input_ids)
    reads = measure_kv_grouping(model, input_ids)
    n_rep = config.num_attention_heads // config.num_key_value_heads
    linear_modules = sorted({name.split("layers.0.", 1)[1] for name in target_modules(model, [0])})

    return {
        "model": spec,
        **config_facts(config),
        "parallel_block": shared["shared"],
        "linear_modules": linear_modules,
        "v_proj_bias": first_layer.self_attn.v_proj.bias is not None,
        "v_norm": not plain_v["linear"],
        "kv_grouping": describe_grouping(reads, n_rep),
        "layernorm_bias": getattr(first_layer.input_layernorm, "bias", None) is not None,
        "shared_input_null_vector": null_vector["orthogonal"],
        "checks": {
            "how": "measured on a tiny random copy of the model (same class and flags), first decoder layer",
            "shared_input_max_rel_difference": shared["max_rel_difference"],
            "null_vector_max_abs_cos": null_vector["max_abs_cos"],
            "v_linearity_rel_error": plain_v["rel_error"],
            "kv_head_read_by_each_query_head": reads,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(parser)
    args = parser.parse_args(argv)
    ctx = setup(args)

    model_facts = facts(ctx.aligned)
    out_path = ctx.out_root / "model_facts.json" if ctx.dry else REPO_ROOT / "configs" / "model_facts.json"
    write_json(out_path, model_facts)

    print(
        f"parallel_block={model_facts['parallel_block']} v_proj_bias={model_facts['v_proj_bias']} "
        f"v_norm={model_facts['v_norm']} null_vector={model_facts['shared_input_null_vector']}; "
        f"{model_facts['kv_grouping']}; intermediate_size={model_facts['intermediate_size']} -> {out_path}"
    )
    method_assumptions_hold = (
        model_facts["parallel_block"]
        and model_facts["shared_input_null_vector"]
        and not model_facts["v_proj_bias"]
        and not model_facts["v_norm"]
    )
    if not method_assumptions_hold:
        print("WARNING: the shared input factor, its null vector or the V/O invariance demo do not hold for this model")


if __name__ == "__main__":
    main()
