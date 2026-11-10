"""The offline dry run: a tiny random "aligned" and "base" Cohere2 model with a Tiny-Aya-style chat template,
and a small placeholder dataset in the processed layout (the "harmful" prompts are placeholders, no
harmful text). Every script's --dry-run points its config here and runs the real code on CPU in seconds.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import REPO_ROOT

DRY = REPO_ROOT / "outputs" / "dryrun"

# The "base" model is the aligned one with every parameter nudged by this fraction of its mean magnitude.
BASE_PERTURBATION = 0.02
BASE_PERTURBATION_SEED = 5

# Placeholder "harmful" prompt per language; {i} becomes e.g. "dev 3".
PLACEHOLDER_PROMPTS = {
    "en": "placeholder request {i} in English",
    "hi": "प्लेसहोल्डर अनुरोध {i}",
    "hinglish": "placeholder request {i} Hinglish mein",
}
HARMFUL_SPLIT_SIZES = (("factor", 6), ("dev", 4), ("test", 4))
XSTEST_SPLIT_SIZES = (("dev", 3), ("test", 3))
N_TASK_ROWS = 24
N_CONTAMINATION_ROWS = 4
N_MMLU_QUESTIONS = 4


def make_dryrun(root: Path = DRY) -> dict[str, Path]:
    """Build the tiny models and the placeholder data once. Returns their paths."""
    models = root / "models"
    aligned = models / "tiny-aya-global"
    base = models / "tiny-aya-base"
    data = root / "data"
    if not (aligned / "config.json").exists():
        _write_tiny_models(aligned, base)
    if not (data / "task_t1" / "heldout.jsonl").exists():
        _write_placeholder_data(data)
    return {"aligned": aligned, "base": base, "data": data}


def _write_tiny_models(aligned: Path, base: Path) -> None:
    """Save the tiny aligned model and a slightly perturbed copy as the base model, each with the tokenizer."""
    from .testing import tiny_cohere2, tiny_tokenizer

    tokenizer = tiny_tokenizer()
    aligned_model = tiny_cohere2(seed=0)
    aligned_model.save_pretrained(aligned)
    tokenizer.save_pretrained(aligned)

    base_model = tiny_cohere2(seed=0)
    generator = torch.Generator().manual_seed(BASE_PERTURBATION_SEED)
    with torch.no_grad():
        for param in base_model.parameters():
            noise = torch.randn(param.shape, generator=generator)
            param.add_(BASE_PERTURBATION * noise * param.abs().mean())
    base_model.save_pretrained(base)
    tokenizer.save_pretrained(base)


def _write_placeholder_data(data: Path) -> None:
    """Write placeholder harmful, XSTest, task, contamination and MMLU files in the processed layout."""
    from .datasets import write_jsonl

    for lang, template in PLACEHOLDER_PROMPTS.items():
        for split, n_rows in HARMFUL_SPLIT_SIZES:
            write_jsonl(data / "harmful" / lang / f"{split}.jsonl", _harmful_rows(lang, template, split, n_rows))
        for split, n_rows in XSTEST_SPLIT_SIZES:
            write_jsonl(data / "xstest" / lang / f"{split}.jsonl", _xstest_rows(lang, split, n_rows))

    task = _task_rows()
    write_jsonl(data / "task_t1" / "train_4000.jsonl", task[:12])
    write_jsonl(data / "task_t1" / "train_1000.jsonl", task[:8])
    write_jsonl(data / "task_t1" / "heldout.jsonl", task[12:])
    write_jsonl(data / "contamination" / "beavertails_20.jsonl", _contamination_rows())
    for lang in PLACEHOLDER_PROMPTS:
        write_jsonl(data / "mmlu" / f"{lang}.jsonl", _mmlu_rows(lang))


def _harmful_rows(lang: str, template: str, split: str, n_rows: int) -> list[dict]:
    """Placeholder rows standing in for one language and split of the harmful prompts."""
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "id": f"{lang}-{split}-{i}",
                "prompt": template.format(i=f"{split} {i}"),
                "metadata": {"lang": lang, "split": split, "source": "dryrun"},
            }
        )
    return rows


def _xstest_rows(lang: str, split: str, n_rows: int) -> list[dict]:
    """Benign placeholder rows standing in for one language and split of XSTest."""
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "id": f"x{lang}-{split}-{i}",
                "prompt": f"benign question {i} ({lang})",
                "metadata": {"lang": lang, "split": split, "source": "dryrun"},
            }
        )
    return rows


def _task_rows() -> list[dict]:
    """Small Hindi arithmetic questions ("question i: two plus i is how much?") with their answers."""
    rows = []
    for i in range(N_TASK_ROWS):
        rows.append({"id": f"t-{i}", "prompt": f"सवाल {i}: दो और {i} कितने हैं?", "answer": f"{2 + i}"})
    return rows


def _contamination_rows() -> list[dict]:
    """Placeholder rows standing in for the BeaverTails contamination pairs."""
    rows = []
    for i in range(N_CONTAMINATION_ROWS):
        rows.append(
            {
                "id": f"c-{i}",
                "prompt": f"placeholder contaminated prompt {i}",
                "answer": "placeholder answer",
                "metadata": {"source": "dryrun"},
            }
        )
    return rows


def _mmlu_rows(lang: str) -> list[dict]:
    """Four-choice placeholder questions standing in for Global-MMLU in one language."""
    rows = []
    for i in range(N_MMLU_QUESTIONS):
        rows.append(
            {"id": f"m{lang}{i}", "question": f"Question {i}?", "choices": ["a", "b", "c", "d"], "answer": i % 4}
        )
    return rows
