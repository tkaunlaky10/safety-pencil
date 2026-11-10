"""Plumbing shared by scripts/: config and devices, loading models, data and factors, and the dev probe.

Every script takes --dry-run, which swaps in the tiny offline model and data from pencil.dryrun and writes
to outputs/dryrun/ instead of outputs/.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import chat
from .config import REPO_ROOT, load_config, load_env


@dataclass
class Ctx:
    """What every script needs: the config, where to read and write, and the device."""

    cfg: dict
    dry: bool
    device: str
    dtype: torch.dtype
    data_dir: Path
    out_root: Path
    t0: float = field(default_factory=time.time)

    @property
    def aligned(self) -> str:
        return self.cfg["models"]["aligned"]

    @property
    def base(self) -> str:
        return self.cfg["models"]["base"]

    def out(self, *parts: str) -> Path:
        p = self.out_root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def elapsed(self) -> float:
        return time.time() - self.t0


def common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", default=None, help="experiment.yaml (default configs/experiment.yaml)")
    p.add_argument("--dry-run", action="store_true", help="tiny offline model and placeholder data, CPU")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--data-dir", default=None, help="processed data (default data/processed)")
    return p


def setup(args) -> Ctx:
    load_env()
    cfg = load_config(args.config)
    if args.dry_run:
        from .dryrun import DRY, make_dryrun

        paths = make_dryrun()
        cfg = copy.deepcopy(cfg)
        cfg["models"].update(aligned=str(paths["aligned"]), base=str(paths["base"]))
        cfg["factors"].update(n_prompts=4, layers_per_pass=2, max_len=256, selection_layers=[1, 3])
        cfg["pencil"]["top_k_mlp"] = 32
        cfg["lora"].update(epochs=1, batch_size=2, grad_accum=1, max_len=256, r=4, alpha=8)
        cfg["generation"].update(max_new_tokens=8, batch_size=4)
        cfg["evaluation"].update(probe_per_lang=3, mmlu_per_lang=3)
        return Ctx(cfg, True, args.device or "cpu", torch.float32, paths["data"], DRY)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    data_dir = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data" / "processed"
    return Ctx(cfg, False, device, dtype, data_dir, REPO_ROOT / "outputs")


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


# ---------------------------------------------------------------- models
def load_tokenizer(ctx: Ctx):
    """The pinned tokenizer (θ0's), used with every model so the chat template never changes."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ctx.aligned)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(spec: str, ctx: Ctx, dtype: torch.dtype | None = None):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(spec), dtype=dtype or ctx.dtype).to(ctx.device).eval()


def model_shape(model) -> dict:
    c = model.config
    return {
        "n_layers": c.num_hidden_layers,
        "hidden": c.hidden_size,
        "inter": c.intermediate_size,
        "n_heads": c.num_attention_heads,
        "n_kv": c.num_key_value_heads,
        "head_dim": getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads,
    }


def adapter_dir(ctx: Ctx, run: str, frac: int = 100) -> Path:
    """outputs/ckpt/{run}/frac{F}: the LoRA adapter saved at F% of training."""
    return ctx.out_root / "ckpt" / run / f"frac{frac}"


# ---------------------------------------------------------------- data
def harmful_rows(ctx: Ctx, lang: str, split: str, n: int | None = None) -> list[dict]:
    from .datasets import load_split

    return load_split(ctx.data_dir, "harmful", split, lang)[:n]


def task_rows(ctx: Ctx, name: str, n: int | None = None) -> list[dict]:
    from .datasets import load_split

    return load_split(ctx.data_dir, "task_t1", name)[:n]


def task_dev_rows(ctx: Ctx) -> list[dict]:
    """The first probe_per_lang held-out task rows: task dev for μ choices (and the probe tier's task rows)."""
    return [
        {k: r[k] for k in ("id", "prompt", "answer")}
        for r in task_rows(ctx, "heldout", ctx.cfg["evaluation"]["probe_per_lang"])
    ]


def encode_pairs(tok, pairs: list[tuple[str, str]], max_len: int | None) -> list[chat.Encoded]:
    """Template (prompt, response) pairs; pairs whose response alone exceeds max_len are skipped."""
    offset, suffix = chat.user_content_offset(tok), chat.prompt_suffix_len(tok)
    out = []
    for prompt, response in pairs:
        try:
            out.append(chat.encode_chat(tok, prompt, response, max_len=max_len, user_offset=offset, suffix_len=suffix))
        except ValueError:
            continue
    return out


def make_batches(encoded: list[chat.Encoded], pad_id: int, size: int) -> list[dict]:
    return [chat.collate(encoded[i : i + size], pad_id) for i in range(0, len(encoded), size)]


def theta0_answers(ctx: Ctx, model0, tok, lang: str, split: str, n: int | None, batch_size: int = 1) -> list[dict]:
    """Harmful prompts with θ0's own greedy answers, teacher-forced, as batches. θ0 is the aligned model, so
    these are the responses we want to keep (no new harmful completions). Answers are cached in
    outputs/generations/ (gitignored)."""
    from .generate import generate_cached

    gen = ctx.cfg["generation"]
    rows = harmful_rows(ctx, lang, split, n)
    key = f"{split}_harmful_{lang}"
    answers = generate_cached(
        model0,
        tok,
        {key: rows},
        checkpoint=ctx.aligned,
        max_new_tokens=gen["max_new_tokens"],
        batch_size=gen["batch_size"],
        root=ctx.out("generations"),
    )[key]
    encoded = encode_pairs(tok, [(r["prompt"], answers[r["id"]]) for r in rows], ctx.cfg["factors"]["max_len"])
    return make_batches(encoded, tok.pad_token_id, batch_size)


# ---------------------------------------------------------------- the dev probe
def dev_harm_batches(ctx: Ctx, model0, tok, langs) -> dict[str, list[dict]]:
    """Per language, θ0's answers to the first probe_per_lang dev harmful prompts."""
    n = ctx.cfg["evaluation"]["probe_per_lang"]
    return {lang: theta0_answers(ctx, model0, tok, lang, "dev", n) for lang in langs}


def measure_S(model0, model, batches_by_lang) -> dict[str, float]:
    """The measured safety drift per language: mean per-token KL(θ0 ‖ θ) on θ0's own dev answers."""
    from .invariance import output_kl

    return {lang: output_kl(model0, model, b, mask_key="loss_mask")["kl_mean"] for lang, b in batches_by_lang.items()}


class DevProbe:
    """The cheap dev measurement used to compare repairs and baselines: drift S per language and the task
    NLL on the task dev rows. `measure(model)` returns {"S": {lang: S}, "S_mean": ..., "task_nll": ...}."""

    def __init__(self, ctx: Ctx, model0, tok, langs):
        self.model0, self.tok = model0, tok
        self.harm = dev_harm_batches(ctx, model0, tok, langs)
        self.task = task_dev_rows(ctx)

    def task_nll(self, model) -> float:
        from .evaluate import task_nll

        return task_nll(model, self.tok, self.task)["nll_per_token"]

    def measure(self, model) -> dict:
        S = measure_S(self.model0, model, self.harm)
        return {"S": S, "S_mean": sum(S.values()) / len(S), "task_nll": self.task_nll(model)}


# ---------------------------------------------------------------- factors
def factor_dir(ctx: Ctx, side: str, name: str, selection: bool = False) -> Path:
    """outputs/factors/{side}/{name}, or outputs/factors_selection/... for the row-set selection pass."""
    return ctx.out_root / ("factors_selection" if selection else "factors") / side / name


def load_factors(ctx: Ctx, side: str, name: str, selection: bool = False):
    """One pass's factor files, indexed; tensors are read on demand."""
    from .factor_store import FactorStore

    return FactorStore.from_dir(factor_dir(ctx, side, name, selection))


def available_langs(ctx: Ctx, selection: bool = False) -> list[str]:
    return [lang for lang in ctx.cfg["languages"] if factor_dir(ctx, "harm", lang, selection).exists()]


def harm_factors(ctx: Ctx, langs, weights: str = "equal", selection: bool = False):
    """Harmful factors pooled over languages with equal weights (default) or by loss-token count."""
    from .factor_store import LazyFactors

    stores = [load_factors(ctx, "harm", lang, selection) for lang in langs]
    return LazyFactors(
        stores,
        [s.n_loss for s in stores] if weights == "tokens" else None,
        meta={"langs": list(langs), "weights": weights},
    )


def task_factors(ctx: Ctx, run: str, frac: int = 100, selection: bool = False):
    from .factor_store import LazyFactors

    name = f"run{run}" + ("" if frac == 100 else f"_frac{frac}")
    return LazyFactors([load_factors(ctx, "task", name, selection)], meta={"run": run, "frac": frac})


def prepare_factors(model, fac_h, fac_t, pc: dict):
    """factor_store.prepare_for_repair with the pencil config's settings. Returns (fh, ft, rel_tol, floors)."""
    from .factor_store import as_lazy, prepare_for_repair

    if not pc.get("project_known_nulls", True):
        return as_lazy(fac_h), as_lazy(fac_t), pc.get("rel_tol", 1e-10), {}
    return prepare_for_repair(
        model, fac_h, fac_t, base_rel_tol=pc.get("rel_tol", 1e-10), floor_factor=pc.get("rel_tol_floor_factor", 10.0)
    )


def single_row_set(pc: dict) -> str:
    """pencil.repair_row_set as one name, for code that uses the same rows for every module (the baselines,
    the invariance demo, the diagonal Fisher)."""
    row_set = pc.get("repair_row_set", "all")
    return row_set if isinstance(row_set, str) else row_set.get("default", "all")


def repair_kwargs(pc: dict, rel_tol: float, **overrides) -> dict:
    """ModelRepair keyword arguments from the pencil config. Bases are solved on the GPU when there is one and
    kept in float32 (about 12 GB of RAM for all 36 layers)."""
    kw = {
        "row_set": pc.get("repair_row_set", "all"),
        "damping": pc["damping"],
        "gamma": pc["gamma_c"],
        "top_k_mlp": pc.get("top_k_mlp"),
        "rel_tol": rel_tol,
        "rel_tol_other": pc.get("rel_tol", 1e-10),
        "eig_device": "cuda" if torch.cuda.is_available() else "cpu",
        "basis_dtype": getattr(torch, pc.get("basis_dtype", "float32")),
    }
    return kw | overrides


@dataclass
class RepairInputs:
    """Everything the repair, invariance, baseline and localisation scripts start from."""

    tok: object
    model0: torch.nn.Module
    langs: list[str]
    deltas: dict  # module -> LoRADelta, only modules that have factors, in layer order
    fh: object  # LazyFactors, harmful side at θ0, known nulls projected out
    ft: object  # LazyFactors, task side at θ1
    rel_tol: float
    floors: dict


def load_repair_inputs(
    ctx: Ctx,
    run: str = "A",
    frac: int = 100,
    *,
    langs=None,
    pool: str = "equal",
    selection: bool = False,
    dtype: torch.dtype | None = None,
) -> RepairInputs:
    from .delta import lora_delta
    from .modules import layer_order

    tok = load_tokenizer(ctx)
    model0 = load_model(ctx.aligned, ctx, dtype)
    langs = langs or available_langs(ctx, selection)
    fh, ft, rel_tol, floors = prepare_factors(
        model0, harm_factors(ctx, langs, pool, selection), task_factors(ctx, run, frac, selection), ctx.cfg["pencil"]
    )
    deltas = lora_delta(adapter_dir(ctx, run, frac), lowrank=True)
    deltas = {n: deltas[n] for n in layer_order(deltas) if n in fh.a_key}
    return RepairInputs(tok, model0, langs, deltas, fh, ft, rel_tol, floors)
