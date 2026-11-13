#!/usr/bin/env bash
# One WiSE-FT setting on the test split, without factors ($STAGE_ARG: <run>_<α>, e.g. A_0p5 for α = 0.5: HF job labels
# allow only [a-zA-Z0-9_-], so "p" stands for the decimal point), on an L4: ~1.2 h.
# The post-hoc frontier match of 13 Nov (PREREGISTRATION.md deviation log): the lowest-S dev setting with task NLL
# ≤ NLL(μ*) + 0.005. Builds W0 + (1 − α)ΔW from the run's final adapter (scripts/11_wiseft_checkpoint.py), evaluates
# it at the full tier as test_frontier_wiseft<α>_run<run>, and pushes only the result JSON.
set -euo pipefail
run=${STAGE_ARG%%_*}
alpha=${STAGE_ARG#*_}
alpha=${alpha//p/.}
if [[ -z $run || -z $alpha || $alpha == "$STAGE_ARG" ]]; then
  echo "STAGE_ARG must be <run>_<alpha>, e.g. A_0p5 (got '$STAGE_ARG')"
  exit 2
fi
source "$(dirname "$0")/lib.sh"

python jobs/sync.py pull "data/processed/**" "outputs/ckpt/$run/**"
choose_langs
checkpoint=$(python scripts/11_wiseft_checkpoint.py --run "$run" --alpha "$alpha" | tail -n 1)
evaluate_checkpoint "$checkpoint" "test_frontier_wiseft${alpha}_run$run" full
