#!/usr/bin/env bash
# The repair and its comparison for one run ($STAGE_ARG: A) on an L4: ~7 h, ~$5.5. Runs next to the pencil stage,
# which supplied the leverage map; P1 and P2 run in the invariance_forecast stage. Ordered so that running out of time or money costs
# localisation, not the baseline comparison (P6):
#
#   factors       as in the pencil stage (jobs/stages/lib.sh), ~1.7 h
#   sweep         the μ sweep on dev (S, task NLL, trust region); every μ in pencil.mu_grid is saved
#   baselines     the must-run set on dev
#   finalists     μ* and the RegMean and diagonal Fisher settings by the fixed rule (pencil/finalists.py, dev
#                 only), all probe-evaluated (probe tier, not test tier: deviation log)
#   mu_probes     probe evaluations of the other μ (P3)
#   localisation  refusal-direction cosine, targeted vs random repair (P5)
#   test_tier     only what outputs/control/compare_test_tier.txt in the runs repo lists ("pencil", "regmean"),
#                 written once the remaining credits are known; absent means none
#
# Each step after the factors runs even if an earlier one failed. Merged checkpoints never leave the job; each is
# deleted once nothing later needs it (μ* and RegMean's finalist wait for the test-tier step).
set -euo pipefail
run=$STAGE_ARG
must_baselines=(wiseft resta safelora safemerge regmean diag_fisher)
evaluated_baselines=(regmean diag_fisher)  # WiSE-FT's finalist is picked and recorded, not evaluated
sweep_name="repair_sweep_compare_run$run"
baselines_name="baselines_run${run}_must"
control_file=outputs/control/compare_test_tier.txt
repaired=outputs/ckpt/repaired
saved_baselines=outputs/ckpt/baselines
source "$(dirname "$0")/lib.sh"

pick_args=(--run "$run" --sweep "$sweep_name" --baselines "$baselines_name" --test-tier "${evaluated_baselines[@]}")

sweep_and_save() {  # dev S and task NLL at every μ; each μ's merged model is kept for its evaluations
  local mu_grid
  read -ra mu_grid <<< "$(python -c 'from pencil.config import load_config; print(*load_config()["pencil"]["mu_grid"])')"
  python scripts/06_repair.py sweep --run "$run" --save "${mu_grid[@]}" --out-name "$sweep_name"
}

mu_star_checkpoint() {  # the saved model at μ* (06_repair.py names its folders <tag>_mu<μ:g>)
  local mu_star
  mu_star=$(python scripts/09_finalists.py "${pick_args[@]}" --print mu)
  echo "$repaired/run${run}_frac100_full_mu${mu_star}"
}

evaluate_finalists() {  # pick, probe-evaluate pencil μ* and the baseline finalists
  python scripts/09_finalists.py "${pick_args[@]}"
  local pencil_checkpoint
  pencil_checkpoint=$(mu_star_checkpoint)
  evaluate_checkpoint "$pencil_checkpoint" "probe_repair_run${run}_mu${pencil_checkpoint##*_mu}" probe
  local baseline_picks
  read -ra baseline_picks <<< "$(python scripts/09_finalists.py "${pick_args[@]}" --print baselines)"
  python scripts/07_baselines.py --run "$run" --finalists "${baseline_picks[@]}" --out-name "baselines_run${run}_finalists"
  for checkpoint in "$saved_baselines"/run"$run"_*; do
    if [[ ! -d $checkpoint ]]; then
      continue
    fi
    evaluate_checkpoint "$checkpoint" "probe_finalist_$(basename "$checkpoint")" probe
    if [[ $(basename "$checkpoint") != run"$run"_regmean_* ]]; then
      rm -rf "$checkpoint"  # RegMean's finalist waits for the test-tier step
    fi
  done
}

evaluate_other_mus() {  # P3: every μ other than μ*, then free its 7 GB
  local pencil_checkpoint
  pencil_checkpoint=$(mu_star_checkpoint)
  for checkpoint in "$repaired"/run"$run"_*_mu*; do
    if [[ ! -d $checkpoint || $checkpoint == "$pencil_checkpoint" ]]; then
      continue
    fi
    evaluate_checkpoint "$checkpoint" "probe_repair_run${run}_mu${checkpoint##*_mu}" probe
    rm -rf "$checkpoint"
  done
}

evaluate_test_tier() {  # whatever the control file lists, on the test split
  local wanted=""
  python jobs/sync.py pull "$control_file"
  if [[ -f $control_file ]]; then
    wanted=$(cat "$control_file")
  fi
  echo "test tier: ${wanted:-nothing requested}"
  if [[ $wanted == *pencil* ]]; then
    local pencil_checkpoint
    pencil_checkpoint=$(mu_star_checkpoint)
    evaluate_checkpoint "$pencil_checkpoint" "test_repair_run${run}_mu${pencil_checkpoint##*_mu}" full
  fi
  if [[ $wanted == *regmean* ]]; then
    for checkpoint in "$saved_baselines"/run"$run"_regmean_*; do
      evaluate_checkpoint "$checkpoint" "test_finalist_$(basename "$checkpoint")" full
    done
  fi
}

pull_inputs
choose_langs
collect_factors
step sweep sweep_and_save
step baselines python scripts/07_baselines.py --run "$run" --methods "${must_baselines[@]}" --out-name "$baselines_name"
step finalists evaluate_finalists
step mu_probes evaluate_other_mus
step localisation python scripts/08_localisation.py --run "$run" --mu 0.1
step test_tier evaluate_test_tier
finish_steps
