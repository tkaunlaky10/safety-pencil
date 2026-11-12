#!/usr/bin/env bash
# The test-tier comparison for one run ($STAGE_ARG: A) on an L4, in parallel with the pencil stage: ~7.5 h, ~$5.
#
#   factors     as in the pencil stage (jobs/stages/lib.sh), ~1.7 h
#   sweep       the μ sweep on dev: S and task NLL only (no checkpoints, no probe evaluations)
#   baselines   the must-run set on dev
#   finalists   μ* and each compared baseline's setting by the fixed rule (pencil/finalists.py, dev only); the
#               finalists are saved, evaluated on the test split, then deleted (7 GB each)
#
# WiSE-FT's finalist is picked and recorded but not evaluated on the test split (HF credits, 12 Nov). Each step
# after the factors runs even if an earlier one failed; merged checkpoints never leave the job.
set -euo pipefail
run=$STAGE_ARG
must_baselines=(wiseft resta safelora safemerge regmean diag_fisher)
test_tier_baselines=(regmean diag_fisher)
sweep_name="repair_sweep_compare_run$run"  # not the pencil stage's name: both push to outputs/results
baselines_name="baselines_run${run}_must"
source "$(dirname "$0")/lib.sh"

evaluate_finalists() {  # pick, save, evaluate on the test split, delete
  local pick_args=(--run "$run" --sweep "$sweep_name" --baselines "$baselines_name" --test-tier "${test_tier_baselines[@]}")
  python scripts/09_finalists.py "${pick_args[@]}"
  local mu_star
  mu_star=$(python scripts/09_finalists.py "${pick_args[@]}" --print mu)
  local baseline_picks
  read -ra baseline_picks <<< "$(python scripts/09_finalists.py "${pick_args[@]}" --print baselines)"
  python scripts/06_repair.py sweep --run "$run" --mu "$mu_star" --save "$mu_star" --out-name "repair_finalist_run$run"
  python scripts/07_baselines.py --run "$run" --finalists "${baseline_picks[@]}" --out-name "baselines_run${run}_finalists"
  for checkpoint in outputs/ckpt/repaired/run"$run"_*_mu* outputs/ckpt/baselines/run"$run"_*; do
    if [[ ! -d $checkpoint ]]; then
      continue
    fi
    evaluate_checkpoint "$checkpoint" "test_finalist_$(basename "$checkpoint")" full
    rm -rf "$checkpoint"
  done
}

pull_inputs
choose_langs
collect_factors
step sweep python scripts/06_repair.py sweep --run "$run" --out-name "$sweep_name"
step baselines python scripts/07_baselines.py --run "$run" --methods "${must_baselines[@]}" --out-name "$baselines_name"
step finalists evaluate_finalists
finish_steps
