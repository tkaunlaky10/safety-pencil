#!/usr/bin/env bash
# Calibration (L4, about 40 min). Checks everything the fine-tunes depend on before money goes into them:
# the test suite, access to the gated models, the datasets that need no LLM, the model facts, and a 20-step
# timing run of fine-tune A. Pushes the datasets and results. When it succeeds, the entrypoint starts hinglish
# and fine-tunes A, C and D.
set -euo pipefail

python -m pytest -q
python jobs/check_access.py

python scripts/01_build_harmful.py --stratify
python scripts/01_build_task_t1.py
python scripts/01_build_mmlu.py
python scripts/01_build_xstest.py --skip-llm
python scripts/01_check_contamination.py
python jobs/sync.py push data/processed outputs/results

python scripts/00_model_facts.py
python scripts/03_finetune.py --run A --max-steps 20
python jobs/sync.py push configs/model_facts.json outputs/results
