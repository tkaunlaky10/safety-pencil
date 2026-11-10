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
SHELL_XDG = os.environ.get("XDG_CACHE_HOME")  # read before .env is loaded
SIGTERM_EXIT_CODE = 143  # 128 + SIGTERM, the conventional exit code of a process stopped by SIGTERM


# ------------------------------------------------------------------------------------------ data


def _read_rows(path: Path, data_dir: Path) -> list[dict]:
    """The rows of one JSONL input, or [] with a warning when the file does not exist."""
    from pencil.datasets import read_jsonl

    if not path.exists():
        warnings.warn(f"{path} not found; skipped", stacklevel=2)
        return []
    print(f"  {path.relative_to(data_dir)}")
    return read_jsonl(path)


def _mmlu_item(row: dict, lang: str, index: int) -> dict:
    """One Global-MMLU question in the form evaluate() takes; rows without an id get mmlu-{lang}-{index}."""
    return {
        "id": row.get("id", f"mmlu-{lang}-{index}"),
        "question": row["question"],
        "choices": row["choices"],
        "answer": row["answer"],
    }


def load_eval_data(data_dir: Path, split: str, langs: list[str]) -> dict:
    """The plain dict pencil.evaluate.evaluate() takes, read from the files in the module docstring."""
    from pencil.datasets import split_path

    data: dict = {"harmful": {}, "xstest": {}, "task": [], "mmlu": {}}
    for lang in langs:
        for kind in ("harmful", "xstest"):
            prompt_rows = _read_rows(split_path(data_dir, kind, split, lang), data_dir)
            if prompt_rows:
                data[kind][lang] = [{"id": row["id"], "prompt": row["prompt"]} for row in prompt_rows]
        mmlu_rows = _read_rows(data_dir / "mmlu" / f"{lang}.jsonl", data_dir)
        if mmlu_rows:
            data["mmlu"][lang] = [_mmlu_item(row, lang, index) for index, row in enumerate(mmlu_rows)]
    task_rows = _read_rows(split_path(data_dir, "task_t1", "heldout"), data_dir)
    data["task"] = [{"id": row["id"], "prompt": row["prompt"], "answer": row["answer"]} for row in task_rows]
    return data


# ------------------------------------------------------------------------------------------ caches


def _exit_on_sigterm(signum, frame) -> None:
    """Turn SIGTERM into a normal exit, so the atexit cleanup also runs when the job is stopped politely."""
    sys.exit(SIGTERM_EXIT_CODE)


def isolate_caches() -> str:
    """Force AUDITKIT_DISABLE_RUN_CACHE=1 and point XDG_CACHE_HOME at a temporary folder wiped at exit, since
    AuditKIT keeps raw completions under $XDG_CACHE_HOME/auditkit. HF_HOME is pinned first: huggingface_hub
    derives its cache from XDG_CACHE_HOME and would otherwise re-download every model into the temp folder."""
    os.environ["AUDITKIT_DISABLE_RUN_CACHE"] = "1"
    shell_cache = Path(SHELL_XDG) if SHELL_XDG else Path.home() / ".cache"
    os.environ.setdefault("HF_HOME", str(shell_cache.expanduser() / "huggingface"))
    temp_cache = tempfile.mkdtemp(prefix="auditkit-xdg-")
    os.environ["XDG_CACHE_HOME"] = temp_cache
    atexit.register(shutil.rmtree, temp_cache, ignore_errors=True)
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    return temp_cache


# ------------------------------------------------------------------------------------------ summary


def _percent_cell(rate: dict | None) -> str:
    """A rate as a 6-wide percentage, or a dash when it is missing."""
    if not rate or rate.get("p") is None:
        return "     -"
    return f"{100 * rate['p']:6.1f}"


def _summary_line(lang: str, metrics: dict) -> str:
    """One table row: refusal, unsafe and over-refusal rates, MMLU accuracy and KL to the reference."""
    harmful = metrics.get("harmful", {})
    benign = metrics.get("xstest", {})
    mmlu = metrics.get("mmlu", {})
    kl = metrics.get("kl_to_reference", {}).get("kl_per_token")
    kl_cell = "-" if kl is None else f"{kl:.4f}"
    return (
        f"{lang:9}"
        f"{_percent_cell(harmful.get('refusal')):>9}"
        f"{_percent_cell(harmful.get('unsafe')):>9}"
        f"{_percent_cell(benign.get('over_refusal')):>10}"
        f"{_percent_cell(mmlu.get('accuracy')):>7}"
        f"{kl_cell:>9}"
    )


def print_summary(result: dict) -> None:
    """Print a per-language table of the headline metrics, then the task NLL."""
    print(f"\n{'lang':9}{'refusal%':>9}{'unsafe%':>9}{'over-ref%':>10}{'mmlu%':>7}{'KL/tokenizer':>9}")
    for lang, metrics in result["metrics"].items():
        print(_summary_line(lang, metrics))
    task = result.get("task")
    if task and task["nll_per_token"] is not None:
        print(f"task NLL/token {task['nll_per_token']:.4f} over {task['n_tokens']} tokens")


# ------------------------------------------------------------------------------------------ main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command-line options (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="local path, hub id or PEFT adapter folder")
    parser.add_argument("--tier", required=True, choices=["probe", "full"])
    parser.add_argument("--data-dir", default=str(REPO / "data" / "processed"))
    parser.add_argument("--out-name", required=True, help="writes outputs/results/{out-name}.json")
    parser.add_argument("--langs", nargs="+", help="default: `languages` in the config")
    parser.add_argument("--reference", help="model for output KL on benign prompts (e.g. θ0); off by default")
    parser.add_argument("--tokenizer", help="pinned tokenizer; default: models.aligned (θ0)")
    parser.add_argument("--harm-judge", help="AuditKIT spec; default hf:<models.harm_judge>")
    parser.add_argument("--refusal-judge", help="AuditKIT spec; default hf:<models.refusal_judge>")
    parser.add_argument("--config", help="default: configs/experiment.yaml")
    parser.add_argument("--no-cache", action="store_true", help="do not keep generations in outputs/generations/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from pencil.config import load_config, load_env

    load_env()
    temp_cache = isolate_caches()
    from transformers import AutoTokenizer  # only after the cache variables are set

    from pencil.evaluate import TIER_SPLIT, evaluate

    cfg = load_config(args.config)
    langs = args.langs or list(cfg["languages"])
    split = TIER_SPLIT[args.tier]
    print(
        f"[02_evaluate] {args.checkpoint} | tier {args.tier} (split {split}) | langs {langs} | "
        f"temporary XDG_CACHE_HOME {temp_cache}"
    )
    data = load_eval_data(Path(args.data_dir), split, langs)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or cfg["models"]["aligned"])
    result = evaluate(
        args.checkpoint,
        args.tier,
        cfg=cfg,
        data=data,
        judges={"harm": args.harm_judge, "refusal": args.refusal_judge},
        out_name=args.out_name,
        tokenizer=tokenizer,
        reference=args.reference,
        cache=not args.no_cache,
    )
    print_summary(result)
    print(f"wrote outputs/results/{args.out_name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
