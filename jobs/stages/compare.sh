#!/usr/bin/env bash
# The repair and its comparison for one run ($STAGE_ARG: A) on an L40S (its 62 GB of RAM are needed): ~5 h.
#
#   factors       harmful prompts at θ0 (en, hi, Hinglish) and task data at the run's final adapter (lib.sh), ~40 min
#   invariance    the repair under the V/O gauge (P1), first: ~40 GB of RAM in its own process, freed after
#   sweep         the μ sweep, covariant damping only; every μ's measured S per language, task NLL and step ratios
#                 (Euclidean and input-weighted, worst and median module) are kept whether or not rule 11 accepts it,
#                 and every μ is saved. If rule 11 accepts no μ, the stage ends here.
#   baselines     the must-run set on dev
#   finalists     μ* and the RegMean and diagonal Fisher settings by the fixed rule (pencil/finalists.py, dev
#                 only), probe-evaluated (probe tier: deviation log)
#   mu_probes     probe evaluations of the other μ (P3)
#   localisation  refusal-direction cosine, targeted vs random repair (P5)
#
# Each step after the factors runs even if an earlier one failed. Merged checkpoints never leave the job.
set -euo pipefail
run=$STAGE_ARG
must_baselines=(wiseft resta safelora safemerge regmean diag_fisher)
evaluated_baselines=(regmean diag_fisher)  # WiSE-FT's finalist is picked and recorded, not evaluated
sweep_name="repair_sweep_compare_run$run"
baselines_name="baselines_run${run}_must"
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
    rm -rf "$checkpoint"
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

pull_inputs
choose_langs
collect_factors
step invariance python scripts/05_invariance.py --run "$run"
step sweep sweep_and_save
if ! python scripts/09_finalists.py --sweep "$sweep_name" --any-passing; then
  echo "== no μ passed rule 11: ending after the sweep (its measurements are kept)"
  finish_steps
  exit 0
fi
step baselines python scripts/07_baselines.py --run "$run" --methods "${must_baselines[@]}" --out-name "$baselines_name"
step finalists evaluate_finalists
step mu_probes evaluate_other_mus
step localisation python scripts/08_localisation.py --run "$run" --mu 0.1
finish_steps
