#!/usr/bin/env bash
# Evaluation on an L4. The target is the first argument, or $STAGE_ARG; the tier is $EVAL_TIER (launch.py --tier):
#   probe (default)  100 dev prompts per language, ~15 min per checkpoint:
#                      theta0 -> probe_theta0;  A, C or D -> probe_run<R>_frac25 ... frac100
#                    (the names 06_repair.py forecast reads)
#   full             the test split, ~1 h per checkpoint:
#                      theta0 -> test_theta0;   A, C or D -> test_run<R>_frac100 (the final adapter only)
# Waits up to two hours for the Hinglish data (made by the hinglish stage); without it, evaluates en and hi only.
# Pushes only the result JSONs: raw generations stay in this job.
set -euo pipefail
target=${1:-$STAGE_ARG}
tier=${EVAL_TIER:-probe}
source "$(dirname "$0")/lib.sh"

prefix=probe
fracs=(25 50 75 100)
if [[ $tier == full ]]; then
  prefix=test
  fracs=(100)
fi

langs=(en hi hinglish)
if ! python jobs/sync.py wait data/processed/harmful/hinglish/dev.jsonl data/processed/xstest/hinglish/dev.jsonl \
  --minutes 120; then
  echo "no Hinglish data yet: evaluating en and hi only"
  langs=(en hi)
fi
python jobs/sync.py pull "data/processed/**"

if [[ $target == theta0 ]]; then
  evaluate_checkpoint CohereLabs/tiny-aya-global "${prefix}_theta0" "$tier"
else
  python jobs/sync.py pull "outputs/ckpt/$target/**"
  for frac in "${fracs[@]}"; do
    evaluate_checkpoint "outputs/ckpt/$target/frac$frac" "${prefix}_run${target}_frac$frac" "$tier"
  done
fi
