#!/usr/bin/env bash
# The scaled repair for one run ($STAGE_ARG: A) on an L40S (13 Nov, post-freeze; PREREGISTRATION.md deviation log).
#
#   factors      harmful prompts at θ0 (en, hi, Hinglish) and task data at the run's final adapter (lib.sh), ~40 min
#   sweep        one basis solve; the μ grid from the predicted regret R(μ)/R(0), written to the result before
#                anything is measured; then every μ's dev S (per language and per prompt), task NLL, step ratios
#                and scales, with rule 11 logged. Every μ's merged model is saved.
#   invariance   P1 recheck of the scaled pencil (and the covariant one) at invariance_runA's gauge: same seed,
#                condition numbers and μ
#   [if rule 11 accepts no μ, the stage ends here: a negative result, kept]
#   baselines    WiSE-FT, RegMean, diagonal Fisher on dev
#   finalists    μ* by the fixed rule (pencil/finalists.py); each baseline refined to NLL(μ*) and saved
#   p6           S(pencil μ*) − S(baseline) per language, paired bootstrap over prompts, Holm (dev, no GPU)
#   test_tier    once, in this order: pencil μ*, RegMean, WiSE-FT (last, so a timeout costs only WiSE-FT)
#
# Each step after the factors runs even if an earlier one failed. Merged checkpoints never leave the job.
set -euo pipefail
run=$STAGE_ARG
baselines=(wiseft regmean diag_fisher)
test_tier_baselines=(regmean wiseft)
sweep_name="repair_sweep_scaled_run$run"
dev_baselines_name="baselines_run${run}_scaled"
matched_name="baselines_run${run}_matched"
repaired=outputs/ckpt/repaired
saved_baselines=outputs/ckpt/baselines
source "$(dirname "$0")/lib.sh"

pick_args=(--run "$run" --sweep "$sweep_name" --baselines "$dev_baselines_name")

mu_star_checkpoint() {  # the saved model at μ* (06_repair.py names its folders <tag>_mu<μ:g>)
  local mu_star
  mu_star=$(python scripts/09_finalists.py "${pick_args[@]}" --print mu)
  echo "$repaired/run${run}_frac100_full_mu${mu_star}"
}

finalists_and_matching() {  # μ*, then every baseline refined toward NLL(μ*) and saved for the test tier
  python scripts/09_finalists.py "${pick_args[@]}" --out-name "finalists_scaled_run$run"
  local target_nll
  target_nll=$(python scripts/09_finalists.py "${pick_args[@]}" --print target_nll)
  python scripts/07_baselines.py --run "$run" --methods "${baselines[@]}" --match-nll "$target_nll" \
    --match-from "$dev_baselines_name" --out-name "$matched_name"
}

free_other_mus() {  # keep only μ*'s 7 GB checkpoint
  local keep
  keep=$(mu_star_checkpoint)
  for checkpoint in "$repaired"/run"$run"_*_mu*; do
    if [[ -d $checkpoint && $checkpoint != "$keep" ]]; then
      rm -rf "$checkpoint"
    fi
  done
}

test_tier() {  # the test split, once: pencil μ*, then the matched baselines in test_tier_baselines order
  local pencil_checkpoint
  pencil_checkpoint=$(mu_star_checkpoint)
  evaluate_checkpoint "$pencil_checkpoint" "test_scaled_run${run}_mu${pencil_checkpoint##*_mu}" full
  for method in "${test_tier_baselines[@]}"; do
    local checkpoint="$saved_baselines/run${run}_matched_$method"
    if [[ ! -d $checkpoint ]]; then
      echo "no matched $method checkpoint: not evaluated"
      continue
    fi
    evaluate_checkpoint "$checkpoint" "test_matched_${method}_run$run" full
    rm -rf "$checkpoint"
  done
}

pull_inputs
choose_langs
collect_factors
step sweep python scripts/06_repair.py sweep --run "$run" --method scaled --save-all --grad-diag \
  --out-name "$sweep_name"
step invariance python scripts/05_invariance.py --run "$run" --methods pencil_scaled pencil_covariant \
  --out-name "invariance_scaled_run$run"
if ! python scripts/09_finalists.py --sweep "$sweep_name" --any-passing; then
  echo "== no μ passed rule 11: ending after the sweep and invariance (their measurements are kept)"
  finish_steps
  exit 0
fi
step baselines python scripts/07_baselines.py --run "$run" --methods "${baselines[@]}" --out-name "$dev_baselines_name"
step finalists finalists_and_matching
step free_mus free_other_mus
step p6 python scripts/10_p6.py --run "$run" --sweep "$sweep_name" --matched "$matched_name"
step test_tier test_tier
finish_steps
