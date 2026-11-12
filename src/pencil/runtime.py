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

# Config settings that shrink every stage so the dry run finishes on CPU in seconds.
DRY_RUN_OVERRIDES = {
    "factors": {"n_prompts": 4, "layers_per_pass": 2, "max_len": 256, "selection_layers": [1, 3]},
    "pencil": {"top_k_mlp": 32},
    "lora": {"epochs": 1, "batch_size": 2, "grad_accum": 1, "max_len": 256, "r": 4, "alpha": 8},
    "generation": {"max_new_tokens": 8, "batch_size": 4},
    "evaluation": {"probe_per_lang": 3, "mmlu_per_lang": 3},
}

# Eigenvalue cut-off (relative to the largest) for factors without a known null vector.
DEFAULT_REL_TOL = 1e-10
# Factors with a known null vector are cut at this multiple of their measured noise floor.
DEFAULT_REL_TOL_FLOOR_FACTOR = 10.0


@dataclass
class Ctx:
    """What every script needs: the config, where to read and write, and the device."""

    cfg: dict  # configs/experiment.yaml, with DRY_RUN_OVERRIDES applied under --dry-run
    dry: bool  # True under --dry-run
    device: str
    dtype: torch.dtype  # model dtype: bfloat16 on CUDA, float32 on CPU and in the dry run
    data_dir: Path  # processed data
    out_root: Path  # outputs/, or outputs/dryrun/ under --dry-run
    t0: float = field(default_factory=time.time)  # creation time, for elapsed()

    @property
    def aligned(self) -> str:
        """θ0, the aligned model (Hugging Face id or local path)."""
        return self.cfg["models"]["aligned"]

    @property
    def base(self) -> str:
        """The pre-alignment base model (Hugging Face id or local path)."""
        return self.cfg["models"]["base"]

    def out(self, *parts: str) -> Path:
        """The output folder out_root/parts..., created if it does not exist."""
        path = self.out_root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def elapsed(self) -> float:
        """Seconds since the context was created."""
        return time.time() - self.t0


def common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the flags every script shares (--config, --dry-run, --device, --data-dir) and return the parser."""
    parser.add_argument("--config", default=None, help="experiment.yaml (default configs/experiment.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="tiny offline model and placeholder data, CPU")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    parser.add_argument("--data-dir", default=None, help="processed data (default data/processed)")
    return parser


def setup(args) -> Ctx:
    """Load .env and the config, and build the context for a real run or for the offline dry run."""
    load_env()
    cfg = load_config(args.config)
    if args.dry_run:
        return _dry_run_context(cfg, args.device)
    return _full_run_context(cfg, args.device, args.data_dir)


def _dry_run_context(cfg: dict, requested_device: str | None) -> Ctx:
    """Context for --dry-run: the tiny offline models and data, DRY_RUN_OVERRIDES, float32, outputs/dryrun/."""
    from .dryrun import DRY, make_dryrun

    paths = make_dryrun()
    dry_cfg = copy.deepcopy(cfg)
    dry_cfg["models"].update(aligned=str(paths["aligned"]), base=str(paths["base"]))
    for section, settings in DRY_RUN_OVERRIDES.items():
        dry_cfg[section].update(copy.deepcopy(settings))
    device = requested_device or "cpu"
    return Ctx(cfg=dry_cfg, dry=True, device=device, dtype=torch.float32, data_dir=paths["data"], out_root=DRY)


def _full_run_context(cfg: dict, requested_device: str | None, data_dir: str | None) -> Ctx:
    """Context for a real run: CUDA when available (in bfloat16), data/processed and outputs/ unless overridden."""
    if requested_device:
        device = requested_device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    data_path = Path(data_dir) if data_dir else REPO_ROOT / "data" / "processed"
    return Ctx(cfg=cfg, dry=False, device=device, dtype=dtype, data_dir=data_path, out_root=REPO_ROOT / "outputs")


def write_json(path: Path, obj) -> Path:
    """Write `obj` as indented JSON (non-JSON values as their str()), creating parent folders. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


# ---------------------------------------------------------------- models
def load_tokenizer(ctx: Ctx):
    """The pinned tokenizer (θ0's), used with every model so the chat template never changes."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ctx.aligned)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(spec: str, ctx: Ctx, dtype: torch.dtype | None = None, device: str | None = None):
    """A causal LM from a Hugging Face id or local path, in eval mode, on `device` (default ctx.device) in `dtype`
    (default ctx.dtype)."""
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype = ctx.dtype
    if device is None:
        device = ctx.device
    model = AutoModelForCausalLM.from_pretrained(str(spec), dtype=dtype)
    return model.to(device).eval()


def model_shape(model) -> dict:
    """The sizes the factor and leverage code needs: layers, hidden and MLP width, query and KV heads, head size."""
    config = model.config
    head_dim = getattr(config, "head_dim", None)
    if not head_dim:
        head_dim = config.hidden_size // config.num_attention_heads
    return {
        "n_layers": config.num_hidden_layers,
        "hidden": config.hidden_size,
        "inter": config.intermediate_size,
        "n_heads": config.num_attention_heads,
        "n_kv": config.num_key_value_heads,
        "head_dim": head_dim,
    }


def adapter_dir(ctx: Ctx, run: str, frac: int = 100) -> Path:
    """outputs/ckpt/{run}/frac{F}: the LoRA adapter saved at F% of training."""
    return ctx.out_root / "ckpt" / run / f"frac{frac}"


# ---------------------------------------------------------------- data
def harmful_rows(ctx: Ctx, lang: str, split: str, n: int | None = None) -> list[dict]:
    """The first n harmful-prompt rows of one language and split (all of them when n is None)."""
    from .datasets import load_split

    return load_split(ctx.data_dir, "harmful", split, lang)[:n]


def task_rows(ctx: Ctx, name: str, n: int | None = None) -> list[dict]:
    """The first n rows of a task T1 file such as 'train_4000' or 'heldout' (all of them when n is None)."""
    from .datasets import load_split

    return load_split(ctx.data_dir, "task_t1", name)[:n]


def task_dev_rows(ctx: Ctx) -> list[dict]:
    """The first probe_per_lang held-out task rows: task dev for μ choices (and the probe tier's task rows)."""
    n_rows = ctx.cfg["evaluation"]["probe_per_lang"]
    dev_rows = []
    for row in task_rows(ctx, "heldout", n_rows):
        dev_rows.append({"id": row["id"], "prompt": row["prompt"], "answer": row["answer"]})
    return dev_rows


def encode_pairs(tokenizer, pairs: list[tuple[str, str]], max_len: int | None) -> list[chat.Encoded]:
    """Template (prompt, response) pairs; pairs whose response alone exceeds max_len are skipped."""
    user_offset = chat.user_content_offset(tokenizer)
    suffix_len = chat.prompt_suffix_len(tokenizer)
    encoded = []
    for prompt, response in pairs:
        try:
            pair = chat.encode_chat(
                tokenizer, prompt, response, max_len=max_len, user_offset=user_offset, suffix_len=suffix_len
            )
        except ValueError:
            continue
        encoded.append(pair)
    return encoded


def make_batches(encoded: list[chat.Encoded], pad_id: int, size: int) -> list[dict]:
    """Collate consecutive groups of `size` encoded sequences into padded batches (the last may be smaller)."""
    batches = []
    for start in range(0, len(encoded), size):
        batches.append(chat.collate(encoded[start : start + size], pad_id))
    return batches


def theta0_answers(
    ctx: Ctx, model0, tokenizer, lang: str, split: str, n: int | None, batch_size: int = 1
) -> list[dict]:
    """Harmful prompts with θ0's own greedy answers, encoded for teacher forcing and batched.

    θ0 is the aligned model, so these are the responses we want to keep (no new harmful completions).
    Answers are cached in outputs/generations/ (gitignored)."""
    from .generate import generate_cached

    generation_cfg = ctx.cfg["generation"]
    rows = harmful_rows(ctx, lang, split, n)
    cache_key = f"{split}_harmful_{lang}"
    cached = generate_cached(
        model0,
        tokenizer,
        {cache_key: rows},
        checkpoint=ctx.aligned,
        max_new_tokens=generation_cfg["max_new_tokens"],
        batch_size=generation_cfg["batch_size"],
        root=ctx.out("generations"),
    )
    answers = cached[cache_key]
    pairs = [(row["prompt"], answers[row["id"]]) for row in rows]
    encoded = encode_pairs(tokenizer, pairs, ctx.cfg["factors"]["max_len"])
    return make_batches(encoded, tokenizer.pad_token_id, batch_size)


# ---------------------------------------------------------------- the dev probe
def dev_harm_batches(ctx: Ctx, model0, tokenizer, langs) -> dict[str, list[dict]]:
    """Per language, θ0's answers to the first probe_per_lang dev harmful prompts."""
    n_prompts = ctx.cfg["evaluation"]["probe_per_lang"]
    batches_by_lang = {}
    for lang in langs:
        batches_by_lang[lang] = theta0_answers(ctx, model0, tokenizer, lang, "dev", n_prompts)
    return batches_by_lang


def measure_S(model0, model, batches_by_lang) -> dict[str, float]:
    """The measured safety drift per language: mean per-token KL(θ0 ‖ θ) on θ0's own dev answers."""
    from .invariance import output_kl

    drift = {}
    for lang, batches in batches_by_lang.items():
        drift[lang] = output_kl(model0, model, batches, mask_key="loss_mask")["kl_mean"]
    return drift


PROBE_TASK_BATCH = 1  # DevProbe.task_nll: sequences scored at once


class DevProbe:
    """The cheap dev measurement used to compare repairs and baselines: drift S per language and the task
    NLL on the task dev rows. `measure(model)` returns {"S": {lang: S}, "S_mean": ..., "task_nll": ...}."""

    def __init__(self, ctx: Ctx, model0, tokenizer, langs):
        self.model0 = model0
        self.tokenizer = tokenizer
        self.harm = dev_harm_batches(ctx, model0, tokenizer, langs)
        self.task = task_dev_rows(ctx)

    def task_nll(self, model) -> float:
        """Mean task NLL per answer token on the task dev rows, one sequence at a time: the probe runs with θ0
        and a working copy both on the GPU, and a batch of long answers' float32 log-softmax over the 262k-token
        vocabulary does not fit next to them on an L4 (the 12 Nov drift job ran out of memory here)."""
        from .evaluate import task_nll

        return task_nll(model, self.tokenizer, self.task, batch_size=PROBE_TASK_BATCH)["nll_per_token"]

    def measure(self, model) -> dict:
        """Drift S per language, its mean over languages, and the task dev NLL."""
        S = measure_S(self.model0, model, self.harm)
        S_mean = sum(S.values()) / len(S)
        return {"S": S, "S_mean": S_mean, "task_nll": self.task_nll(model)}


# ---------------------------------------------------------------- factors
def factor_dir(ctx: Ctx, side: str, name: str, selection: bool = False) -> Path:
    """outputs/factors/{side}/{name}, or outputs/factors_selection/... for the row-set selection pass."""
    folder = "factors_selection" if selection else "factors"
    return ctx.out_root / folder / side / name


def task_pass_name(run: str, frac: int = 100) -> str:
    """Folder name of a run's task-side factor pass: 'runA' for the final adapter, 'runA_frac50' otherwise."""
    if frac == 100:
        return f"run{run}"
    return f"run{run}_frac{frac}"


def load_factors(ctx: Ctx, side: str, name: str, selection: bool = False):
    """One pass's factor files, indexed; tensors are read on demand."""
    from .factor_store import FactorStore

    return FactorStore.from_dir(factor_dir(ctx, side, name, selection))


def available_langs(ctx: Ctx, selection: bool = False) -> list[str]:
    """The configured languages that have harmful-side factors on disk, in config order."""
    langs = []
    for lang in ctx.cfg["languages"]:
        if factor_dir(ctx, "harm", lang, selection).exists():
            langs.append(lang)
    return langs


def harm_factors(ctx: Ctx, langs, weights: str = "equal", selection: bool = False):
    """Harmful factors pooled over languages with equal weights (default) or by loss-token count."""
    from .factor_store import LazyFactors

    stores = [load_factors(ctx, "harm", lang, selection) for lang in langs]
    pool_weights = [store.n_loss for store in stores] if weights == "tokens" else None
    return LazyFactors(stores, pool_weights, meta={"langs": list(langs), "weights": weights})


def task_factors(ctx: Ctx, run: str, frac: int = 100, selection: bool = False):
    """Task factors at θ1 for one run's adapter at `frac`% of training."""
    from .factor_store import LazyFactors

    store = load_factors(ctx, "task", task_pass_name(run, frac), selection)
    return LazyFactors([store], meta={"run": run, "frac": frac})


def prepare_factors(model, harm, task, pencil_cfg: dict):
    """factor_store.prepare_for_repair with the pencil config's settings. Returns (harm, task, rel_tol, floors).

    With pencil.project_known_nulls off, the factors are only wrapped as LazyFactors and get no floors."""
    from .factor_store import as_lazy, prepare_for_repair

    rel_tol = pencil_cfg.get("rel_tol", DEFAULT_REL_TOL)
    if not pencil_cfg.get("project_known_nulls", True):
        return as_lazy(harm), as_lazy(task), rel_tol, {}
    floor_factor = pencil_cfg.get("rel_tol_floor_factor", DEFAULT_REL_TOL_FLOOR_FACTOR)
    return prepare_for_repair(model, harm, task, base_rel_tol=rel_tol, floor_factor=floor_factor)


def single_row_set(pencil_cfg: dict) -> str:
    """pencil.repair_row_set as one name, for code that uses the same rows for every module (the baselines,
    the invariance demo, the diagonal Fisher)."""
    row_set = pencil_cfg.get("repair_row_set", "all")
    if isinstance(row_set, str):
        return row_set
    return row_set.get("default", "all")


def repair_kwargs(pencil_cfg: dict, rel_tol: float, **overrides) -> dict:
    """ModelRepair keyword arguments from the pencil config, with `overrides` taking precedence.

    Bases are solved on the GPU when there is one and kept in float32 (about 12 GB of RAM for all 36 layers)."""
    eig_device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {
        "row_set": pencil_cfg.get("repair_row_set", "all"),
        "damping": pencil_cfg["damping"],
        "gamma": pencil_cfg["gamma_c"],
        "top_k_mlp": pencil_cfg.get("top_k_mlp"),
        "rel_tol": rel_tol,
        "rel_tol_other": pencil_cfg.get("rel_tol", DEFAULT_REL_TOL),
        "eig_device": eig_device,
        "basis_dtype": getattr(torch, pencil_cfg.get("basis_dtype", "float32")),
    }
    kwargs.update(overrides)
    return kwargs


@dataclass
class RepairInputs:
    """Everything the repair, invariance, baseline and localisation scripts start from."""

    tokenizer: object  # the pinned tokenizer
    model0: torch.nn.Module  # θ0
    langs: list[str]
    deltas: dict  # module name -> LoRADelta; only modules that have factors, in layer order
    harm_factors: object  # LazyFactors, harmful side at θ0, known nulls projected out
    task_factors: object  # LazyFactors, task side at θ1
    rel_tol: float
    floors: dict  # measured noise floors of the factors with known null vectors


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
    """Load θ0, its tokenizer, the prepared harmful and task factors and the run's low-rank ΔW.

    `langs` defaults to every language with harmful factors on disk; `pool` is how they are weighted."""
    tokenizer = load_tokenizer(ctx)
    model0 = load_model(ctx.aligned, ctx, dtype)
    if not langs:
        langs = available_langs(ctx, selection)
    harm = harm_factors(ctx, langs, pool, selection)
    task = task_factors(ctx, run, frac, selection)
    harm, task, rel_tol, floors = prepare_factors(model0, harm, task, ctx.cfg["pencil"])
    deltas = _deltas_with_factors(adapter_dir(ctx, run, frac), harm)
    return RepairInputs(tokenizer, model0, langs, deltas, harm, task, rel_tol, floors)


def _deltas_with_factors(adapter: Path, harm) -> dict:
    """The adapter's low-rank ΔW for the modules that have harmful factors, in layer order."""
    from .delta import lora_delta
    from .modules import layer_order

    all_deltas = lora_delta(adapter, lowrank=True)
    deltas = {}
    for module in layer_order(all_deltas):
        if module in harm.a_key:
            deltas[module] = all_deltas[module]
    return deltas
