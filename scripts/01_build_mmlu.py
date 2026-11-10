"""Global-MMLU subsets for the general-capability check: data/processed/mmlu/{lang}.jsonl.

Rows {"id", "question", "choices": [4 strings], "answer": "A".."D"}, 200 per language by default, drawn
with a fixed seed from the `test` split of CohereLabs/Global-MMLU (configs `en`, `hi`). Global-MMLU has
no Hinglish config, so Hinglish is not built. Expected columns: question, option_a..option_d, answer;
the script stops with the columns it found if they differ.

    python scripts/01_build_mmlu.py
    python scripts/01_build_mmlu.py --input-jsonl hi=/path/to/local.jsonl     # offline / tests
"""

from __future__ import annotations

import argparse
import random
import sys

from pencil.config import load_config
from pencil.datasets import read_jsonl, resolve_data_dir, write_jsonl

COLS = ["question", "option_a", "option_b", "option_c", "option_d", "answer"]


def convert(rows, lang: str, n: int, seed: int) -> list[dict]:
    """n seeded random rows as evaluation items; rows whose answer is not A-D are dropped (not replaced)."""
    rows = list(rows)
    if rows and not set(COLS) <= set(rows[0]):
        raise SystemExit(f"Global-MMLU {lang}: expected columns {COLS}, found {sorted(rows[0])}")
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        r = rows[i]
        ans = str(r["answer"]).strip().upper()
        if ans not in ("A", "B", "C", "D"):
            continue
        out.append(
            {
                "id": f"mmlu-{lang}-{r.get('sample_id', i)}",
                "question": r["question"],
                "choices": [r[c] for c in COLS[1:5]],
                "answer": ans,
                "subject": r.get("subject"),
                "cultural_sensitivity": r.get("cultural_sensitivity_label"),
            }
        )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--langs", nargs="+", default=["en", "hi"])
    p.add_argument("--n", type=int, default=None, help="default evaluation.mmlu_per_lang")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input-jsonl", nargs="+", default=[], help="lang=path, instead of the Hub")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    n = args.n or cfg["evaluation"]["mmlu_per_lang"]
    local = dict(x.split("=", 1) for x in args.input_jsonl)
    out_dir = resolve_data_dir(args.data_dir) / "mmlu"
    for lang in args.langs:
        if lang in local:
            rows = read_jsonl(local[lang])
        else:
            from datasets import load_dataset

            rows = load_dataset(cfg["datasets"]["global_mmlu"], lang, split="test")
        items = convert(rows, lang, n, args.seed)
        path = write_jsonl(out_dir / f"{lang}.jsonl", items)
        print(f"{lang}: {len(items)} questions -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
