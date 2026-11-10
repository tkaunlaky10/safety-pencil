"""The offline dry run: a tiny random "aligned" and "base" Cohere2 model with a Tiny-Aya-style chat template,
and a small placeholder dataset in the processed layout (the "harmful" prompts are placeholders, no
harmful text). Every script's --dry-run points its config here and runs the real code on CPU in seconds.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import REPO_ROOT

DRY = REPO_ROOT / "outputs" / "dryrun"


def make_dryrun(root: Path = DRY) -> dict[str, Path]:
    """Build the tiny models and the placeholder data once. Returns their paths."""
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
        with torch.no_grad():  # "base": the aligned model minus a small perturbation
            g = torch.Generator().manual_seed(5)
            for p in b.parameters():
                p.add_(0.02 * torch.randn(p.shape, generator=g) * p.abs().mean())
        b.save_pretrained(base)
        tok.save_pretrained(base)
    if not (data / "task_t1" / "heldout.jsonl").exists():
        langs = {
            "en": "placeholder request {i} in English",
            "hi": "प्लेसहोल्डर अनुरोध {i}",
            "hinglish": "placeholder request {i} Hinglish mein",
        }
        for lang, tmpl in langs.items():
            for split, n in (("factor", 6), ("dev", 4), ("test", 4)):
                write_jsonl(
                    data / "harmful" / lang / f"{split}.jsonl",
                    [
                        {
                            "id": f"{lang}-{split}-{i}",
                            "prompt": tmpl.format(i=f"{split} {i}"),
                            "metadata": {"lang": lang, "split": split, "source": "dryrun"},
                        }
                        for i in range(n)
                    ],
                )
            for split, n in (("dev", 3), ("test", 3)):
                write_jsonl(
                    data / "xstest" / lang / f"{split}.jsonl",
                    [
                        {
                            "id": f"x{lang}-{split}-{i}",
                            "prompt": f"benign question {i} ({lang})",
                            "metadata": {"lang": lang, "split": split, "source": "dryrun"},
                        }
                        for i in range(n)
                    ],
                )
        task = [{"id": f"t-{i}", "prompt": f"सवाल {i}: दो और {i} कितने हैं?", "answer": f"{2 + i}"} for i in range(24)]
        write_jsonl(data / "task_t1" / "train_4000.jsonl", task[:12])
        write_jsonl(data / "task_t1" / "train_1000.jsonl", task[:8])
        write_jsonl(data / "task_t1" / "heldout.jsonl", task[12:])
        write_jsonl(
            data / "contamination" / "beavertails_20.jsonl",
            [
                {
                    "id": f"c-{i}",
                    "prompt": f"placeholder contaminated prompt {i}",
                    "answer": "placeholder answer",
                    "metadata": {"source": "dryrun"},
                }
                for i in range(4)
            ],
        )
        for lang in langs:
            write_jsonl(
                data / "mmlu" / f"{lang}.jsonl",
                [
                    {
                        "id": f"m{lang}{i}",
                        "question": f"Question {i}?",
                        "choices": ["a", "b", "c", "d"],
                        "answer": i % 4,
                    }
                    for i in range(4)
                ],
            )
    return {"aligned": aligned, "base": base, "data": data}
