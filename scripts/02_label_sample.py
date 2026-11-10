#!/usr/bin/env python
"""Sample cached responses from two checkpoints into a CSV for human labelling (judge validation, Cohen's κ).

    python scripts/02_label_sample.py --checkpoints CohereLabs/tiny-aya-global outputs/checkpoints/A/step_100 \
        [--n 100] [--kind harmful] [--split test] [--langs en hi hinglish] [--seed 0]

Reads outputs/generations/{checkpoint}/{split}_{kind}_{lang}.jsonl (written by scripts/02_evaluate.py; pass the
same --checkpoint strings) and writes outputs/labels/label_sample_{kind}_{split}.csv with the columns
id, lang, checkpoint, prompt, response, label_refusal, label_unsafe. The two label columns are left blank for
the labeller: label_refusal is REFUSE / PARTIAL / COMPLY (configs/refusal_rubric.txt), label_unsafe is 1 / 0.
Per language, n rows are drawn without replacement, half from each checkpoint, then shuffled.

WARNING: the CSV holds raw completions of harmful prompts. It is local-only: it must stay under outputs/
(gitignored) and never be committed, uploaded or shared outside the team (CLAUDE.md rule 7).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
COLUMNS = ["id", "lang", "checkpoint", "prompt", "response", "label_refusal", "label_unsafe"]
WARNING = ("WARNING: this CSV contains raw model completions of harmful prompts. Keep it under outputs/ "
           "(gitignored); never commit, upload or publish it.")


def sample_rows(checkpoints: list[str], *, kind: str, split: str, langs: list[str], n: int, seed: int,
                root: str | Path | None = None) -> list[dict]:
    from pencil.generate import generations_path, load_generations

    rng = random.Random(seed)
    out: list[dict] = []
    for lang in langs:
        rows: list[dict] = []
        for j, ckpt in enumerate(checkpoints):
            quota = n // len(checkpoints) + (j < n % len(checkpoints))
            path = generations_path(ckpt, f"{split}_{kind}_{lang}", root)
            cached = sorted(load_generations(path).values(), key=lambda r: r["id"])
            if len(cached) < quota:
                warnings.warn(f"{path}: {len(cached)} cached generations, wanted {quota}")
            for r in rng.sample(cached, min(quota, len(cached))):
                rows.append({"id": r["id"], "lang": lang, "checkpoint": ckpt, "prompt": r["prompt"],
                             "response": r["completion"], "label_refusal": "", "label_unsafe": ""})
        rng.shuffle(rows)
        out += rows
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs=2, required=True, metavar="CKPT")
    ap.add_argument("--n", type=int, default=100, help="rows per language, split between the checkpoints")
    ap.add_argument("--kind", default="harmful", choices=["harmful", "xstest"])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--langs", nargs="+", default=["en", "hi", "hinglish"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-root", help="default: outputs/generations")
    ap.add_argument("--out", help="CSV path; must be under outputs/")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO / "src"))
    out = Path(args.out) if args.out else OUTPUTS / "labels" / f"label_sample_{args.kind}_{args.split}.csv"
    if not out.resolve().is_relative_to(OUTPUTS.resolve()):
        sys.exit(f"refusing to write raw completions outside {OUTPUTS} (gitignored): {out}")
    rows = sample_rows(args.checkpoints, kind=args.kind, split=args.split, langs=args.langs, n=args.n,
                       seed=args.seed, root=args.gen_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:         # BOM: Excel then shows Devanagari
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {lang: sum(r["lang"] == lang for r in rows) for lang in args.langs}
    print(f"wrote {len(rows)} rows {counts} to {out}")
    print(WARNING, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
