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
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
COLUMNS = ["id", "lang", "checkpoint", "prompt", "response", "label_refusal", "label_unsafe"]
WARNING = (
    "WARNING: this CSV contains raw model completions of harmful prompts. Keep it under outputs/ "
    "(gitignored); never commit, upload or publish it."
)


def _quota(n: int, n_checkpoints: int, j: int) -> int:
    """Rows for checkpoint j when n rows are split as evenly as possible; the first n % n_checkpoints get one
    extra."""
    base, extra = divmod(n, n_checkpoints)
    if j < extra:
        return base + 1
    return base


def _label_row(cached: dict, lang: str, checkpoint: str) -> dict:
    """One CSV row: the prompt and response, with the two label columns blank for the labeller."""
    return {
        "id": cached["id"],
        "lang": lang,
        "checkpoint": checkpoint,
        "prompt": cached["prompt"],
        "response": cached["completion"],
        "label_refusal": "",
        "label_unsafe": "",
    }


def sample_rows(
    checkpoints: list[str],
    *,
    kind: str,
    split: str,
    langs: list[str],
    n: int,
    seed: int,
    root: str | Path | None = None,
) -> list[dict]:
    """Per language, a seeded sample of n cached responses split between the checkpoints, then shuffled."""
    from pencil.generate import generations_path, load_generations

    rng = random.Random(seed)
    sampled: list[dict] = []
    for lang in langs:
        lang_rows: list[dict] = []
        for j, checkpoint in enumerate(checkpoints):
            quota = _quota(n, len(checkpoints), j)
            path = generations_path(checkpoint, f"{split}_{kind}_{lang}", root)
            cached = sorted(load_generations(path).values(), key=lambda row: row["id"])
            if len(cached) < quota:
                warnings.warn(f"{path}: {len(cached)} cached generations, wanted {quota}", stacklevel=2)
            for row in rng.sample(cached, min(quota, len(cached))):
                lang_rows.append(_label_row(row, lang, checkpoint))
        rng.shuffle(lang_rows)
        sampled += lang_rows
    return sampled


def _output_path(args: argparse.Namespace) -> Path:
    """The CSV path (default outputs/labels/label_sample_{kind}_{split}.csv); exits if it is outside outputs/."""
    default_out = OUTPUTS / "labels" / f"label_sample_{args.kind}_{args.split}.csv"
    out = Path(args.out) if args.out else default_out
    if not out.resolve().is_relative_to(OUTPUTS.resolve()):
        sys.exit(f"refusing to write raw completions outside {OUTPUTS} (gitignored): {out}")
    return out


def write_label_csv(path: Path, rows: list[dict]) -> None:
    """Write the rows with a UTF-8 BOM, so Excel shows Devanagari correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command-line options (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs=2, required=True, metavar="CKPT")
    parser.add_argument("--n", type=int, default=100, help="rows per language, split between the checkpoints")
    parser.add_argument("--kind", default="harmful", choices=["harmful", "xstest"])
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--langs", nargs="+", default=["en", "hi", "hinglish"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gen-root", help="default: outputs/generations")
    parser.add_argument("--out", help="CSV path; must be under outputs/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = _output_path(args)
    rows = sample_rows(
        args.checkpoints,
        kind=args.kind,
        split=args.split,
        langs=args.langs,
        n=args.n,
        seed=args.seed,
        root=args.gen_root,
    )
    write_label_csv(out, rows)
    rows_per_lang = Counter(row["lang"] for row in rows)
    counts = {lang: rows_per_lang[lang] for lang in args.langs}
    print(f"wrote {len(rows)} rows {counts} to {out}")
    print(WARNING, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
