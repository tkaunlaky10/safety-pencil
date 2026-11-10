"""Shared plumbing for scripts/: config, model and data loading, and the offline dry run.

Every script takes `--dry-run`. It builds (once) a tiny random Cohere2 "aligned" and "base" model with a
Tiny-Aya-style chat template under outputs/dryrun/models/, and a small placeholder dataset in the processed
layout under outputs/dryrun/data/ (no harmful text: the "harmful" prompts are placeholders), then runs the
real code path end to end on CPU in seconds. Outputs go to outputs/dryrun/ instead of outputs/.
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

DRY = REPO_ROOT / "outputs" / "dryrun"


@dataclass
class Ctx:
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
        paths = make_dryrun()
        cfg = copy.deepcopy(cfg)
        cfg["models"].update(aligned=str(paths["aligned"]), base=str(paths["base"]))
        cfg["factors"].update(n_prompts=4, layers_per_pass=2, max_len=256, selection_layers=[1, 3])
        cfg["pencil"]["top_k_mlp"] = 32
        cfg["lora"].update(epochs=1, batch_size=2, grad_accum=1, max_len=256, r=4, alpha=8)
        cfg["generation"].update(max_new_tokens=8, batch_size=4)
        cfg["evaluation"].update(probe_per_lang=3, mmlu_per_lang=3)
        device = args.device or "cpu"
        return Ctx(cfg, True, device, torch.float32, paths["data"], DRY)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    data_dir = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data" / "processed"
    return Ctx(cfg, False, device, dtype, data_dir, REPO_ROOT / "outputs")


def load_tokenizer(ctx: Ctx):
    """The pinned tokenizer (θ0's), used for every model so the chat template never changes."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ctx.aligned)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(spec: str, ctx: Ctx, dtype: torch.dtype | None = None):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(spec), dtype=dtype or ctx.dtype)
    return m.to(ctx.device).eval()


def model_shape(model) -> dict:
    c = model.config
    return {"n_layers": c.num_hidden_layers, "hidden": c.hidden_size, "inter": c.intermediate_size,
            "n_heads": c.num_attention_heads, "n_kv": c.num_key_value_heads,
            "head_dim": getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads}


# ---------------------------------------------------------------- data
def harmful_rows(ctx: Ctx, lang: str, split: str, n: int | None = None) -> list[dict]:
    from .datasets import load_split
    rows = load_split(ctx.data_dir, "harmful", split, lang)
    return rows[:n] if n else rows


def task_rows(ctx: Ctx, name: str, n: int | None = None) -> list[dict]:
    from .datasets import load_split
    rows = load_split(ctx.data_dir, "task_t1", name)
    return rows[:n] if n else rows


def encode_pairs(tok, pairs: list[tuple[str, str]], max_len: int | None) -> list[chat.Encoded]:
    off, suf = chat.user_content_offset(tok), chat.prompt_suffix_len(tok)
    out = []
    for p, r in pairs:
        try:
            out.append(chat.encode_chat(tok, p, r, max_len=max_len, user_offset=off, suffix_len=suf))
        except ValueError:
            continue                              # response alone longer than max_len: skip, count below
    return out


def make_batches(encoded: list[chat.Encoded], pad_id: int, size: int) -> list[dict]:
    return [chat.collate(encoded[i:i + size], pad_id) for i in range(0, len(encoded), size)]


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


# ---------------------------------------------------------------- the offline dry run
def make_dryrun(root: Path = DRY) -> dict[str, Path]:
    """Tiny aligned/base models + tokenizer + placeholder data in the processed layout. Idempotent."""
    from .datasets import write_jsonl
    from .testing import tiny_cohere2, tiny_tokenizer

    models = root / "models"
    aligned, base, data = models / "tiny-aya-global", models / "tiny-aya-base", root / "data"
    if not (aligned / "config.json").exists():
        tok = tiny_tokenizer()
        m = tiny_cohere2(seed=0)
        m.save_pretrained(aligned)
        tok.save_pretrained(aligned)
        b = tiny_cohere2(seed=0)
        with torch.no_grad():                     # "base": the aligned model minus a small perturbation
            g = torch.Generator().manual_seed(5)
            for p in b.parameters():
                p.add_(0.02 * torch.randn(p.shape, generator=g) * p.abs().mean())
        b.save_pretrained(base)
        tok.save_pretrained(base)
    if not (data / "task_t1" / "heldout.jsonl").exists():
        langs = {"en": "placeholder request {i} in English",
                 "hi": "प्लेसहोल्डर अनुरोध {i}",
                 "hinglish": "placeholder request {i} Hinglish mein"}
        for lang, tmpl in langs.items():
            for split, n in (("factor", 6), ("dev", 4), ("test", 4)):
                write_jsonl(data / "harmful" / lang / f"{split}.jsonl",
                            [{"id": f"{lang}-{split}-{i}", "prompt": tmpl.format(i=f"{split} {i}"),
                              "metadata": {"lang": lang, "split": split, "source": "dryrun"}} for i in range(n)])
            for split, n in (("dev", 3), ("test", 3)):
                write_jsonl(data / "xstest" / lang / f"{split}.jsonl",
                            [{"id": f"x{lang}-{split}-{i}", "prompt": f"benign question {i} ({lang})",
                              "metadata": {"lang": lang, "split": split, "source": "dryrun"}} for i in range(n)])
        task = [{"id": f"t-{i}", "prompt": f"सवाल {i}: दो और {i} कितने हैं?", "answer": f"{2 + i}"} for i in range(24)]
        write_jsonl(data / "task_t1" / "train_4000.jsonl", task[:12])
        write_jsonl(data / "task_t1" / "train_1000.jsonl", task[:8])
        write_jsonl(data / "task_t1" / "heldout.jsonl", task[12:])
        write_jsonl(data / "contamination" / "beavertails_20.jsonl",
                    [{"id": f"c-{i}", "prompt": f"placeholder contaminated prompt {i}", "answer": "placeholder answer",
                      "metadata": {"source": "dryrun"}} for i in range(4)])
        for lang in langs:
            write_jsonl(data / "mmlu" / f"{lang}.jsonl",
                        [{"id": f"m{lang}{i}", "question": f"Question {i}?", "choices": ["a", "b", "c", "d"],
                          "answer": i % 4} for i in range(4)])
    return {"aligned": aligned, "base": base, "data": data}


# ---------------------------------------------------------------- factor files
def factor_dir(ctx: Ctx, side: str, name: str, selection: bool = False) -> Path:
    """outputs/factors/{side}/{name}, or outputs/factors_selection/... for the row-set selection pass."""
    return ctx.out_root / ("factors_selection" if selection else "factors") / side / name


def load_factors(ctx: Ctx, side: str, name: str, selection: bool = False):
    """All layer-chunk files of one pass, indexed but not loaded (FactorStore reads tensors on demand)."""
    from .factors import FactorStore
    d = factor_dir(ctx, side, name, selection)
    files = sorted(d.glob("layers*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no factor files in {d}; run scripts/04_factors.py first")
    return FactorStore(files)


def harm_factors(ctx: Ctx, langs, weights: str = "equal", selection: bool = False):
    """Harmful factors pooled over languages, computed one layer at a time: equal weights (the default) or
    by loss-token count."""
    from .factors import LazyFactors
    stores = [load_factors(ctx, "harm", l, selection) for l in langs]
    w = [s.n_loss for s in stores] if weights == "tokens" else None
    return LazyFactors(stores, w, meta={"langs": list(langs), "weights": weights})


def task_factors(ctx: Ctx, run: str, frac: int = 100, selection: bool = False):
    from .factors import LazyFactors
    name = f"run{run}" + ("" if frac == 100 else f"_frac{frac}")
    return LazyFactors([load_factors(ctx, "task", name, selection)], meta={"run": run, "frac": frac})


def available_langs(ctx: Ctx, selection: bool = False) -> list[str]:
    return [l for l in ctx.cfg["languages"] if factor_dir(ctx, "harm", l, selection).exists()]


def prepare_factors(model, fh, ft, pc: dict):
    """Project the known null vectors out (if configured) and pick rel_tol from their measured floors.
    Works on Factors or LazyFactors (the projection is then applied lazily, layer by layer).
    Returns (fh, ft, rel_tol, floors). rel_tol applies to the factors that had a known null vector; use
    repair_kwargs(), which also passes rel_tol_other = pc["rel_tol"] for the rest."""
    from .factors import LazyFactors, known_null_vectors, null_floors, project_out
    from .geneig import auto_rel_tol
    if not pc.get("project_known_nulls", True):
        return fh, ft, pc.get("rel_tol", 1e-10), {}
    nulls = known_null_vectors(model, fh)
    floors = {**{f"harm::{k}": v for k, v in null_floors(fh, nulls).items()},
              **{f"task::{k}": v for k, v in null_floors(ft, nulls).items()}}
    tol = auto_rel_tol(floors.values(), pc.get("rel_tol", 1e-10), pc.get("rel_tol_floor_factor", 10.0))
    proj = lambda f: f.with_nulls(nulls) if isinstance(f, LazyFactors) else project_out(f, nulls)
    return proj(fh), proj(ft), tol, floors


def repair_kwargs(pc: dict, tol: float, **over) -> dict:
    """ModelRepair keyword arguments from the pencil config and the measured tolerance. Bases are solved on
    the GPU when there is one and kept in float32 (12 GB of RAM for all 36 layers instead of 24)."""
    kw = dict(row_set=pc.get("repair_row_set", "all"), damping=pc["damping"], gamma=pc["gamma_c"],
              top_k_mlp=pc.get("top_k_mlp"), rel_tol=tol, rel_tol_other=pc.get("rel_tol", 1e-10),
              eig_device="cuda" if torch.cuda.is_available() else "cpu",
              basis_dtype=getattr(torch, pc.get("basis_dtype", "float32")))
    kw.update(over)
    return kw


# ---------------------------------------------------------------- the cheap dev probe (S and task NLL)
def dev_harm_batches(ctx: Ctx, model0, tok, langs, n: int) -> dict[str, list[dict]]:
    """θ0's greedy responses to dev harmful prompts, teacher-forced: the probe for the measured drift S."""
    from .generate import generate_cached
    gen = ctx.cfg["generation"]
    out = {}
    for l in langs:
        rows = harmful_rows(ctx, l, "dev", n)
        resp = generate_cached(model0, tok, {f"dev_harmful_{l}": rows}, checkpoint=ctx.aligned,
                               max_new_tokens=gen["max_new_tokens"], batch_size=gen["batch_size"],
                               root=ctx.out("generations"))[f"dev_harmful_{l}"]
        enc = encode_pairs(tok, [(r["prompt"], resp[r["id"]]) for r in rows], ctx.cfg["factors"]["max_len"])
        out[l] = make_batches(enc, tok.pad_token_id, 1)
    return out


def measure_S(model0, model, batches_by_lang) -> dict[str, float]:
    """Per language: mean per-token KL(θ0 ‖ θ) at response positions of θ0's own dev answers."""
    from .invariance import output_kl
    return {l: output_kl(model0, model, b, mask_key="loss_mask")["kl_mean"] for l, b in batches_by_lang.items()}


def task_dev_rows(ctx: Ctx) -> list[dict]:
    """Task dev rows: the first probe_per_lang held-out rows (the probe tier's task rows too)."""
    n = ctx.cfg["evaluation"]["probe_per_lang"]
    return [{"id": r["id"], "prompt": r["prompt"], "answer": r["answer"]} for r in task_rows(ctx, "heldout", n)]
