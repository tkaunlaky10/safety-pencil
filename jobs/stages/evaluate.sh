#!/usr/bin/env bash
# Probe-tier evaluation (100 dev prompts per language) on an L4. The target is the first argument, or $STAGE_ARG:
#   theta0      the aligned model θ0                  -> outputs/results/probe_theta0.json
#   A, C or D   that run's four adapters              -> outputs/results/probe_runA_frac25.json ... frac100
#                                                        (the names 06_repair.py forecast reads)
# Waits up to two hours for the Hinglish data (made by the hinglish stage); without it, evaluates en and hi only.
# Pushes only the result JSONs: raw generations stay in this job.
set -euo pipefail
target=${1:-$STAGE_ARG}

langs=(en hi hinglish)
if ! python jobs/sync.py wait data/processed/harmful/hinglish/dev.jsonl data/processed/xstest/hinglish/dev.jsonl \
  --minutes 120; then
  echo "no Hinglish data yet: evaluating en and hi only"
  langs=(en hi)
fi
python jobs/sync.py pull "data/processed/**"

evaluate() {  # evaluate CHECKPOINT OUT_NAME
  python scripts/02_evaluate.py --checkpoint "$1" --tier probe --out-name "$2" --langs "${langs[@]}"
  python jobs/sync.py push "outputs/results/$2.json"
}

if [[ $target == theta0 ]]; then
  evaluate CohereLabs/tiny-aya-global probe_theta0
else
  python jobs/sync.py pull "outputs/ckpt/$target/**"
  for frac in 25 50 75 100; do
    evaluate "outputs/ckpt/$target/frac$frac" "probe_run${target}_frac$frac"
  done
fi
