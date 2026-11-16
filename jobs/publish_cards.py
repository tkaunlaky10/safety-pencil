# ruff: noqa: E501  (Markdown tables in the cards cannot be wrapped)
"""The README cards for the public Hugging Face repos made by jobs/publish.py (SUBMISSION_TODO.md steps 5–6)."""

from __future__ import annotations

GITHUB = "https://github.com/tkaunlaky10/safety-pencil"
BANNER = "research artifact: unrepaired, do not deploy"

CONTENT_WARNING = (
    "> **Content warning.** This repository contains harmful prompts (requests for dangerous, hateful or illegal "
    "content) collected for red-teaming. They are prompts only: no model answers to them are included."
)

DATA_CARD = f"""---
license: other
license_name: mixed-see-card
language: [en, hi]
pretty_name: "Safety Pencil: data"
tags: [safety, red-teaming, multilingual, hinglish, over-refusal]
size_categories: [1K<n<10K]
---

# Safety Pencil: data

The processed data behind [Safety Pencil]({GITHUB}) (Lexsi AI Safety and Alignment Hackathon, Track 2b): how
fine-tuning Tiny Aya Global on a harmless Hindi task changes its safety behaviour in English, Hindi and Hinglish,
and a closed-form curvature repair. See the repo's `WRITEUP.md` §4 for how each file was built.

{CONTENT_WARNING}

## Files

| Folder | What it is | Rows |
| --- | --- | --- |
| `harmful/{{en,hi,hinglish}}/{{factor,dev,test}}.jsonl` | Harmful prompts, split 40 / 10 / 50% per language with a fixed seed, stratified by harm category | en 388 / 97 / 485, hi 362 / 91 / 453, Hinglish 510 / 134 / 635 |
| `xstest/{{en,hi,hinglish}}/{{dev,test}}.jsonl` | XSTest safe prompts (harmless but alarming-sounding), for over-refusal | en and hi 50 / 200, Hinglish 48 / 198 |
| `task_t1/` | The benign Hindi fine-tuning task: `train_4000` (run A), `train_1000` (runs C and D), `heldout` (300 human-written rows, for task NLL) | 4000 / 1000 / 300 |
| `mmlu/{{en,hi}}.jsonl` | Global-MMLU subsets, 200 questions each, the capability-retention check | 200 / 200 |
| `*/_curatorkit/`, `*/manifest.json`, `*/conversion.json` | CuratorKIT provenance, checksums and conversion settings | |

The leak check (`data_contamination_check.json` in `tkaunlaky10/safety-pencil-results`) found 0 exact and 0
within-set duplicates between test and fit splits.

## Sources and licences

| Data | Source | Licence |
| --- | --- | --- |
| Harmful prompts, en and hi | `CohereLabs/aya_redteaming` (splits english, hindi) | Apache-2.0 |
| Harmful prompts, Hinglish | Converted from the en and hi prompts within each split by `CohereLabs/aya-expanse-32b` through CuratorKIT | Derived from Apache-2.0 data; the converter model is CC-BY-NC 4.0 |
| Over-refusal prompts | XSTest safe prompts (`walledai/XSTest`); hi and Hinglish translated by Aya Expanse 32B through CuratorKIT | CC-BY-4.0 |
| Task T1 | `CohereLabs/aya_dataset` Hindi rows, topped up from `CohereLabs/aya_collection` | Apache-2.0 |
| Capability check | `CohereLabs/Global-MMLU` (en, hi) | Apache-2.0 |

Use each part under its source's licence. Run D's fine-tune also used 20 BeaverTails rows (via SafeTune,
CC-BY-NC 4.0). They are harmful question–answer pairs and are **not included** here; get BeaverTails from its
original source if you need it.

## Intended use

Research on multilingual safety evaluation and repair. Not for training models to produce harmful content.
"""

RESULTS_CARD = f"""---
license: mit
language: [en, hi]
pretty_name: "Safety Pencil: results and logs"
tags: [safety, evaluation, logs]
---

# Safety Pencil: results and logs

Every measured result of [Safety Pencil]({GITHUB}) (Lexsi AI Safety and Alignment Hackathon, Track 2b), the job
records and logs that produced them, and the generated summary. Each number in the repo's `WRITEUP.md` comes from
a file here.

| Path | What it is |
| --- | --- |
| `outputs/results/*.json` | One JSON per result, with its config, seed, git hash, checkpoint and timestamp. Per-prompt rows hold only ids, the SHA-256 of the model's answer, and the judges' labels (Llama Guard 3 unsafe and categories; Aya Expanse refusal label) |
| `outputs/figures/*.png` | The figures |
| `runs/<run>/run.json`, `runs/<run>/log.txt` | One record per Hugging Face Job (stage, code version, hardware) and its log, with tokens masked |
| `summary.md` | Tables generated from the result JSONs by `jobs/report.py` |

**No model generations are included,** only hashes of them, because many are answers to harmful prompts. The data
these results were measured on is in `tkaunlaky10/safety-pencil-data`.

Licence: MIT, like the code. The judges are `meta-llama/Llama-Guard-3-8B` (Llama 3.1 Community Licence) and
`CohereLabs/aya-expanse-8b` (CC-BY-NC 4.0).
"""

RUNS = {
    "A": "4000 rows of `task_t1/train_4000`, seed 0. The main run",
    "C": "1000 rows of `task_t1/train_1000`, seed 0",
    "D": "1000 rows of `task_t1/train_1000` plus 20 BeaverTails harmful question–answer rows (2%, via SafeTune), "
    "seed 0. The contaminated variant",
}


def model_card(run: str) -> str:
    """The model card for the run's LoRA adapters (frac25 … frac100)."""
    return f"""---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
pipeline_tag: text-generation
language: [hi, en]
tags: [lora, safety, research-artifact, do-not-deploy]
---

# Safety Pencil: LoRA run {run} ({BANNER})

> **{BANNER}.** These adapters were trained to study safety drift. Fine-tuning on harmless Hindi data weakened the
> base model's safety behaviour: on the main run, English harmful compliance rose from 7% to 22%. Do not deploy
> them or build products on them.

LoRA adapters on `CohereLabs/tiny-aya-global` from [Safety Pencil]({GITHUB}) (Lexsi AI Safety and Alignment
Hackathon, Track 2b). Run {run}: {RUNS[run]}. The subfolders `frac25`, `frac50`, `frac75` and `frac100` are the
adapter at 25, 50, 75 and 100% of training.

## Training

- Base model: `CohereLabs/tiny-aya-global`, bf16.
- LoRA: r 16, α 32, dropout 0.05, on all seven projections (q, k, v, o, gate, up, down).
- Learning rate 2e-4, 2 epochs, max length 1024, batch 4 × gradient accumulation 4.
- Data: the Hindi task T1 in `tkaunlaky10/safety-pencil-data`, built from `CohereLabs/aya_dataset` and
  `CohereLabs/aya_collection` (Apache-2.0).
- Trained with our own PEFT loop using SafeTune's `lora_wrap` and `tokenize_dataset`. Its measured effects (refusal, harmful compliance,
  over-refusal, MMLU, KL drift) are in `tkaunlaky10/safety-pencil-results` and the repo's `WRITEUP.md`.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "tkaunlaky10/safety-pencil-lora-run{run}", subfolder="frac100")
tokenizer = AutoTokenizer.from_pretrained("tkaunlaky10/safety-pencil-lora-run{run}", subfolder="frac100")
```

## Licence

CC-BY-NC 4.0 plus Cohere Labs' acceptable-use policy, as derivatives of Tiny Aya Global. SafeTune's licence (LSAL)
forbids deploying drifted checkpoints.
"""


LANG_NAMES = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
# (key in the numbers, row label, judge or source)
WISEFT_ROWS = (
    ("unsafe", "Harmful compliance, %", "Llama Guard 3 8B: unsafe answers to harmful prompts"),
    ("over_refusal", "Over-refusal, %", "Aya Expanse 8B: refusals of XSTest safe prompts"),
    ("refusal", "Refusal on harmful prompts, %", "Aya Expanse 8B"),
    ("youden_j", "Refusal Youden J, points (post hoc)", "refusal − over-refusal"),
)


def _wiseft_table(numbers: dict) -> str:
    """Markdown rows: each metric × language, then Global-MMLU and task NLL, for θ0, θ1 and this model."""
    columns = ("theta0", "theta1", "wiseft")
    lines = ["| Metric | Language | θ0 aligned | θ1 fine-tuned | This model |", "| --- | --- | --- | --- | --- |"]
    for key, label, _ in WISEFT_ROWS:
        for lang, lang_name in LANG_NAMES.items():
            values = " | ".join(f"{numbers[c][lang][key]:.1f}" for c in columns)
            lines.append(f"| {label} | {lang_name} | {values} |")
    for lang in ("en", "hi"):
        values = " | ".join(f"{numbers[c][lang]['mmlu']:.1f}" for c in columns)
        lines.append(f"| Global-MMLU accuracy, % | {LANG_NAMES[lang]} | {values} |")
    values = " | ".join(f"{numbers[c]['task_nll']:.3f}" for c in columns)
    lines.append(f"| Hindi task NLL, nats/token (held-out, lower is better) | Hindi | {values} |")
    return "\n".join(lines)


def wiseft_card(run: str, alpha: float, repo: str, numbers: dict) -> str:
    """The model card for run `run`'s adapter with every lora_B scaled by (1 − alpha)."""
    keep = 1 - alpha
    judges = "\n".join(f"- **{label}**: {how}." for _, label, how in WISEFT_ROWS)
    sources = ", ".join(
        f"{name} = `{numbers[c]['source']}`"
        for c, name in (("theta0", "θ0"), ("theta1", "θ1"), ("wiseft", "this model"))
    )
    return f"""---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
pipeline_tag: text-generation
language: [en, hi]
tags: [safety, lora, wise-ft, model-merging, multilingual, research-artifact]
---

# Safety Pencil: run {run} adapter interpolated at α = {alpha:g} (WiSE-FT)

> **Evaluated research artifact, not certified for deployment: red-team it in your own languages first.**

## What it is

Run {run}'s Hindi instruction-tuning LoRA adapter from [Safety Pencil]({GITHUB}) (4,000 rows; LoRA r 16, LoRA alpha
32, 2 epochs, on `CohereLabs/tiny-aya-global`), with every `lora_B` multiplied by {keep:g}. Because the LoRA update
is ΔW = (lora_alpha / r) · B · A, this gives exactly θ = θ0 + {keep:g}·ΔW: WiSE-FT with α = {alpha:g}. The original adapter is
[`tkaunlaky10/safety-pencil-lora-run{run}`](https://huggingface.co/tkaunlaky10/safety-pencil-lora-run{run}) (`frac100`).

Our evaluation used the merged checkpoint W0 + {keep:g}·ΔW from `scripts/11_wiseft_checkpoint.py`, which rounds the
merged weights to bf16. This adapter applies the same update unmerged, so the two agree up to bf16 rounding.

## Why

A benign Hindi fine-tune tripled English harmful compliance and quadrupled English over-refusal. Halving the update
restores both with no loss on the Hindi task. See `WRITEUP.md` §6.7 in the [GitHub repo]({GITHUB}).

## Results (test split, greedy decoding)

{_wiseft_table(numbers)}

Sources, in [`tkaunlaky10/safety-pencil-results`](https://huggingface.co/datasets/tkaunlaky10/safety-pencil-results)
under `outputs/results/`: {sources}.

{judges}
- **Global-MMLU**: 200 questions per language. **Task NLL**: 300 held-out human-written Hindi rows.

Test sets: 485 / 453 / 635 harmful prompts (`CohereLabs/aya_redteaming`, Hinglish converted) and 200 / 200 / 198
XSTest safe prompts, English / Hindi / Hinglish.

## Limitations

- Hinglish harmful compliance is about {numbers["wiseft"]["hinglish"]["unsafe"]:.0f}%, as for the aligned model ({numbers["theta0"]["hinglish"]["unsafe"]:.0f}%). Interpolating toward θ0 cannot fix what θ0 lacks.
- The judges were not validated against human labels.
- One seed.
- Evaluated with greedy decoding and the base model's default chat template; keep both.
- Evaluated research artifact, not certified for deployment: red-team it in your own languages first.

## Usage

Accept the terms of [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) on Hugging
Face first.

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "{repo}")

messages = [{{"role": "user", "content": "भारत की राजधानी क्या है?"}}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True))
```

## Licence

CC-BY-NC-4.0 as a derivative of Tiny Aya Global, plus the Cohere Labs acceptable-use policy.

## Links

- Code and write-up: [GitHub]({GITHUB}), [`WRITEUP.md`]({GITHUB}/blob/main/WRITEUP.md)
- Data: [`tkaunlaky10/safety-pencil-data`](https://huggingface.co/datasets/tkaunlaky10/safety-pencil-data)
- Results and logs: [`tkaunlaky10/safety-pencil-results`](https://huggingface.co/datasets/tkaunlaky10/safety-pencil-results)
"""
