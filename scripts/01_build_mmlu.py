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

OPTION_COLUMNS = ["option_a", "option_b", "option_c", "option_d"]
COLS = ["question", *OPTION_COLUMNS, "answer"]  # the columns every Global-MMLU row must have
ANSWER_LETTERS = ("A", "B", "C", "D")


def convert(rows, lang: str, n: int, seed: int) -> list[dict]:
    """n seeded random rows as evaluation items; rows whose answer is not A-D are dropped (not replaced)."""
    rows = list(rows)
    if rows and not set(COLS) <= set(rows[0]):
        raise SystemExit(f"Global-MMLU {lang}: expected columns {COLS}, found {sorted(rows[0])}")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    items = []
    for index in order[:n]:
        row = rows[index]
        answer = str(row["answer"]).strip().upper()
        if answer not in ANSWER_LETTERS:
            continue
        items.append(
            {
                "id": f"mmlu-{lang}-{row.get('sample_id', index)}",
                "question": row["question"],
                "choices": [row[column] for column in OPTION_COLUMNS],
                "answer": answer,
                "subject": row.get("subject"),
                "cultural_sensitivity": row.get("cultural_sensitivity_label"),
            }
        )
    return items


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--langs", nargs="+", default=["en", "hi"])
    parser.add_argument("--n", type=int, default=None, help="default evaluation.mmlu_per_lang")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-jsonl", nargs="+", default=[], help="lang=path, instead of the Hub")
    return parser.parse_args(argv)


def load_rows(lang: str, local: dict[str, str], config: dict):
    """One language's rows: the local JSONL given for it, else the Hub test split."""
    if lang in local:
        return read_jsonl(local[lang])
    from datasets import load_dataset

    return load_dataset(config["datasets"]["global_mmlu"], lang, split="test")


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    # --n 0 (like no value) means the config's size
    n_questions = args.n if args.n else config["evaluation"]["mmlu_per_lang"]
    local = dict(value.split("=", 1) for value in args.input_jsonl)  # {lang: path}
    out_dir = resolve_data_dir(args.data_dir) / "mmlu"
    for lang in args.langs:
        items = convert(load_rows(lang, local, config), lang, n_questions, args.seed)
        path = write_jsonl(out_dir / f"{lang}.jsonl", items)
        print(f"{lang}: {len(items)} questions -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
