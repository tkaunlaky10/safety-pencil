#!/usr/bin/env python
"""Evaluate one checkpoint and write one result JSON to outputs/results/{out-name}.json.

    python scripts/02_evaluate.py --checkpoint outputs/checkpoints/A/step_100 --tier probe --out-name A_100_probe
    python scripts/02_evaluate.py --checkpoint CohereLabs/tiny-aya-global --tier full --out-name theta0_full \
        --langs en hi --reference CohereLabs/tiny-aya-global

pencil.evaluate.evaluate generates on harmful + benign prompts, computes teacher-forced metrics, then runs
Llama Guard 3 and the Aya Expanse refusal judge, with one model on the GPU at a time.

--tier probe reads the dev split and keeps the first evaluation.probe_per_lang items of every list; --tier full
reads the test split (CLAUDE.md rule 5). Inputs are the files built by scripts/01_*.py under --data-dir
(layout: pencil.datasets):

    harmful/{lang}/{split}.jsonl   {"id", "prompt", "metadata"}   aya_redteaming (+ Hinglish)
    xstest/{lang}/{split}.jsonl    {"id", "prompt", "metadata"}   XSTest safe prompts (+ hi, Hinglish)
    task_t1/heldout.jsonl          {"id", "prompt", "answer"}     held-out T1 pairs, both tiers
    mmlu/{lang}.jsonl              {"id", "question", "choices", "answer"}   Global-MMLU (01_build_mmlu.py)

Missing files are skipped with a warning (e.g. no Hinglish yet).

Raw generations are cached in outputs/generations/{checkpoint}/ (gitignored; --no-cache skips this) for reuse
and for scripts/02_label_sample.py. They contain harmful completions: never commit or upload them.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import signal
import sys
import tempfile
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHELL_XDG = os.environ.get("XDG_CACHE_HOME")  # before .env is loaded


# ------------------------------------------------------------------------------------------ data


def load_eval_data(data_dir: Path, split: str, langs: list[str]) -> dict:
    """The plain dict pencil.evaluate.evaluate() takes, read from the files in the module docstring."""
    from pencil.datasets import read_jsonl, split_path

    def rows(path: Path) -> list[dict]:
        if not path.exists():
            warnings.warn(f"{path} not found; skipped", stacklevel=2)
            return []
        print(f"  {path.relative_to(data_dir)}")
        return read_jsonl(path)

    data: dict = {"harmful": {}, "xstest": {}, "task": [], "mmlu": {}}
    for lang in langs:
        for kind in ("harmful", "xstest"):
            if found := rows(split_path(data_dir, kind, split, lang)):
                data[kind][lang] = [{"id": r["id"], "prompt": r["prompt"]} for r in found]
        if found := rows(data_dir / "mmlu" / f"{lang}.jsonl"):
            data["mmlu"][lang] = [
                {
                    "id": r.get("id", f"mmlu-{lang}-{i}"),
                    "question": r["question"],
                    "choices": r["choices"],
                    "answer": r["answer"],
                }
                for i, r in enumerate(found)
            ]
    data["task"] = [
        {"id": r["id"], "prompt": r["prompt"], "answer": r["answer"]}
        for r in rows(split_path(data_dir, "task_t1", "heldout"))
    ]
    return data


# ------------------------------------------------------------------------------------------ main


def isolate_caches() -> str:
    """Force AUDITKIT_DISABLE_RUN_CACHE=1 and point XDG_CACHE_HOME at a temporary folder wiped at exit, since
    AuditKIT keeps raw completions under $XDG_CACHE_HOME/auditkit. HF_HOME is pinned first: huggingface_hub
    derives its cache from XDG_CACHE_HOME and would otherwise re-download every model into the temp folder."""
    os.environ["AUDITKIT_DISABLE_RUN_CACHE"] = "1"
    os.environ.setdefault("HF_HOME", str(Path(SHELL_XDG or Path.home() / ".cache").expanduser() / "huggingface"))
    tmp = tempfile.mkdtemp(prefix="auditkit-xdg-")
    os.environ["XDG_CACHE_HOME"] = tmp
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so atexit also runs when killed politely
    return tmp


def _pct(r: dict | None) -> str:
    return "     -" if not r or r.get("p") is None else f"{100 * r['p']:6.1f}"


def print_summary(res: dict) -> None:
    print(f"\n{'lang':9}{'refusal%':>9}{'unsafe%':>9}{'over-ref%':>10}{'mmlu%':>7}{'KL/tok':>9}")
    for lang, m in res["metrics"].items():
        h, x = m.get("harmful", {}), m.get("xstest", {})
        kl = m.get("kl_to_reference", {}).get("kl_per_token")
        print(
            f"{lang:9}{_pct(h.get('refusal')):>9}{_pct(h.get('unsafe')):>9}{_pct(x.get('over_refusal')):>10}"
            f"{_pct(m.get('mmlu', {}).get('accuracy')):>7}{'-' if kl is None else f'{kl:.4f}':>9}"
        )
    if res.get("task") and res["task"]["nll_per_token"] is not None:
        print(f"task NLL/token {res['task']['nll_per_token']:.4f} over {res['task']['n_tokens']} tokens")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="local path, hub id or PEFT adapter folder")
    ap.add_argument("--tier", required=True, choices=["probe", "full"])
    ap.add_argument("--data-dir", default=str(REPO / "data" / "processed"))
    ap.add_argument("--out-name", required=True, help="writes outputs/results/{out-name}.json")
    ap.add_argument("--langs", nargs="+", help="default: `languages` in the config")
    ap.add_argument("--reference", help="model for output KL on benign prompts (e.g. θ0); off by default")
    ap.add_argument("--tokenizer", help="pinned tokenizer; default: models.aligned (θ0)")
    ap.add_argument("--harm-judge", help="AuditKIT spec; default hf:<models.harm_judge>")
    ap.add_argument("--refusal-judge", help="AuditKIT spec; default hf:<models.refusal_judge>")
    ap.add_argument("--config", help="default: configs/experiment.yaml")
    ap.add_argument("--no-cache", action="store_true", help="do not keep generations in outputs/generations/")
    args = ap.parse_args(argv)

    from pencil.config import load_config, load_env

    load_env()
    tmp = isolate_caches()
    from transformers import AutoTokenizer  # after the cache variables are set

    from pencil.evaluate import TIER_SPLIT, evaluate

    cfg = load_config(args.config)
    langs = args.langs or list(cfg["languages"])
    split = TIER_SPLIT[args.tier]
    print(
        f"[02_evaluate] {args.checkpoint} | tier {args.tier} (split {split}) | langs {langs} | "
        f"temporary XDG_CACHE_HOME {tmp}"
    )
    data = load_eval_data(Path(args.data_dir), split, langs)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or cfg["models"]["aligned"])
    res = evaluate(
        args.checkpoint,
        args.tier,
        cfg=cfg,
        data=data,
        judges={"harm": args.harm_judge, "refusal": args.refusal_judge},
        out_name=args.out_name,
        tokenizer=tok,
        reference=args.reference,
        cache=not args.no_cache,
    )
    print_summary(res)
    print(f"wrote outputs/results/{args.out_name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
