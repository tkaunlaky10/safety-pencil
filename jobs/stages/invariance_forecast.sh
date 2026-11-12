#!/usr/bin/env bash
# P1 and P2 for one run ($STAGE_ARG: A) on an L4: ~2.5 h, ~$2. Recovers the two results pencil A lost when it was
# OOMKilled on 12 Nov (its leverage map survived; the sweep, baselines and localisation run in compare A).
#
#   factors     harmful prompts at θ0 (en, hi, Hinglish) and task data at the run's final adapter (jobs/stages/lib.sh)
#   forecast    ½ΔᵀF_hΔ for every run's adapters (P2)
#   invariance  the repair under the V/O gauge (P1), base model loaded in bf16 on the CPU (8d47a46)
#
# Each step after the factors runs even if the other failed; results are pushed after each.
set -euo pipefail
run=$STAGE_ARG
source "$(dirname "$0")/lib.sh"

pull_inputs
choose_langs
collect_factors
step forecast forecast_all_adapters
step invariance python scripts/05_invariance.py --run "$run"
finish_steps
