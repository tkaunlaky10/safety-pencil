"""Record the architecture facts the method depends on in configs/model_facts.json (Prompt 0 step 3).

Reads the config, builds the model skeleton on the meta device (no weights downloaded beyond config.json)
and inspects the installed transformers modeling code:
  (a) parallel block: attention and MLP read the same input_layernorm output;
  (b) linear module names;
  (c) whether v_proj has a bias, or a norm is applied to V;
  (d) the KV grouping rule (query head h reads KV head h // n_rep).
Relevant source lines are quoted in the JSON.

    python scripts/00_model_facts.py            # needs HF_TOKEN in .env (gated model)
    python scripts/00_model_facts.py --dry-run  # tiny offline model, writes outputs/dryrun/model_facts.json
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys

import torch

from pencil.config import REPO_ROOT
from pencil.factors import target_modules
from pencil.runtime import common_args, setup, write_json


def facts(spec: str) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(spec)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    mod = sys.modules[type(model).__module__]
    layer_cls = type(model.model.layers[0])
    attn_cls = type(model.model.layers[0].self_attn)
    fwd = inspect.getsource(layer_cls.forward)
    attn_src = inspect.getsource(attn_cls)
    rkv = inspect.getsource(mod.repeat_kv) if hasattr(mod, "repeat_kv") else ""
    lines = [l.strip() for l in fwd.splitlines()]
    ln_line = next((l for l in lines if "input_layernorm(" in l), "")
    mlp_line = next((l for l in lines if "self.mlp(" in l), "")
    sum_line = next((l for l in lines if re.search(r"residual \+ .*att.*\+ .*mlp", l)), "")
    parallel = (not hasattr(model.model.layers[0], "post_attention_layernorm")
                and "self.mlp(hidden_states)" in mlp_line and bool(sum_line))
    v = model.model.layers[0].self_attn.v_proj
    names = sorted({n.split("layers.0.", 1)[1] for n in target_modules(model, [0])})
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    return {
        "model": spec,
        "model_type": cfg.model_type,
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "attention_bias": getattr(cfg, "attention_bias", None),
        "use_qk_norm": getattr(cfg, "use_qk_norm", None),
        "layer_types": getattr(cfg, "layer_types", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "logit_scale": getattr(cfg, "logit_scale", None),
        "parallel_block": parallel,
        "linear_modules": names,
        "v_proj_bias": v.bias is not None,
        "v_norm": bool(re.search(r"v_norm|value_norm", attn_src)),
        "n_rep": n_rep,
        "kv_grouping": f"query head h reads KV head h // {n_rep}" if "repeat_interleave" in rkv or "expand" in rkv else "unknown",
        "layernorm_bias": getattr(model.model.layers[0].input_layernorm, "bias", None) is not None,
        "comment": {
            "decoder_forward": [l for l in (ln_line, mlp_line, sum_line) if l],
            "repeat_kv": [l.strip() for l in rkv.splitlines() if "expand" in l or "reshape" in l],
            "modeling_file": mod.__file__,
        },
    }


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    args = p.parse_args(argv)
    ctx = setup(args)
    f = facts(ctx.aligned)
    out = ctx.out_root / "model_facts.json" if ctx.dry else REPO_ROOT / "configs" / "model_facts.json"
    write_json(out, f)
    print(f"parallel_block={f['parallel_block']} v_proj_bias={f['v_proj_bias']} {f['kv_grouping']}; "
          f"intermediate_size={f['intermediate_size']} -> {out}")
    if not f["parallel_block"] or f["v_proj_bias"] or f["v_norm"]:
        print("WARNING: the shared-input factor or the V/O invariance demo assumes a parallel block and a plain v_proj")


if __name__ == "__main__":
    main()
