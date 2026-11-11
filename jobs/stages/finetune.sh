#!/usr/bin/env bash
# Fine-tune one run ($STAGE_ARG: A, C or D) on an L4, push its four adapters (25/50/75/100% of training), then
# probe-evaluate them in the same job.
set -euo pipefail
run=$STAGE_ARG

python jobs/sync.py pull "data/processed/**"
python scripts/03_finetune.py --run "$run"
python jobs/sync.py push "outputs/ckpt/$run" outputs/results

bash jobs/stages/evaluate.sh "$run"
