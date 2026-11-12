#!/usr/bin/env bash
# The core pencil analysis for one run ($STAGE_ARG: A, the main run) on an L4: about 5.5 h, ~$4.5.
#
#   factors       harmful prompts at θ0 (en, hi, Hinglish when its prompts exist) and task data at the run's
#                 final adapter, one row set per side: ~220 GB (launch.py refuses smaller disks), ~1.7 h
#   leverage      the leverage map from F_h v = λ H_t v (P5)
#   invariance    the repair under the V/O gauge (P1)
#   forecast      ½ΔᵀF_hΔ for every run's adapters (P2; no eigensolves)
#   sweep         the μ sweep with the trust region and the task-gradient diagnostic (P3); every μ in
#                 pencil.mu_grid is saved, probe-evaluated, then deleted (7 GB each)
#   localisation  refusal-direction cosine, targeted vs random repair (P5)
#
# Each step after the factors runs even if an earlier one failed (jobs/stages/lib.sh), and results are pushed
# after every step. Baselines and the test-tier comparison run in the compare stage, in parallel.
set -euo pipefail
run=$STAGE_ARG
forecast_runs=(A C D)  # the runs with adapters; B was cut
probe_wait_minutes=180
source "$(dirname "$0")/lib.sh"

forecast_all_adapters() {  # once θ0's and every adapter's probe results are in
  local adapters=()
  local probes=(outputs/results/probe_theta0.json)
  for forecast_run in "${forecast_runs[@]}"; do
    adapters+=("outputs/ckpt/$forecast_run/**")
    for frac in 25 50 75 100; do
      probes+=("outputs/results/probe_run${forecast_run}_frac$frac.json")
    done
  done
  python jobs/sync.py pull "${adapters[@]}"
  if ! python jobs/sync.py wait "${probes[@]}" --minutes "$probe_wait_minutes"; then
    echo "not every probe result arrived: forecasting against the ones that did"
  fi
  python jobs/sync.py pull "outputs/results/**"
  python scripts/06_repair.py forecast
}

sweep_and_evaluate() {  # the sweep saves every μ; each saved model is probe-evaluated, then deleted
  local mu_grid
  read -ra mu_grid <<< "$(python -c 'from pencil.config import load_config; print(*load_config()["pencil"]["mu_grid"])')"
  python scripts/06_repair.py sweep --run "$run" --grad-diag --save "${mu_grid[@]}"
  for checkpoint in outputs/ckpt/repaired/run"$run"_*_mu*; do
    if [[ ! -d $checkpoint ]]; then
      continue
    fi
    mu=${checkpoint##*_mu}
    evaluate_checkpoint "$checkpoint" "probe_repair_run${run}_mu${mu}" probe
    rm -rf "$checkpoint"
  done
}

pull_inputs
choose_langs
collect_factors
step leverage python scripts/04_leverage.py --run "$run"
step invariance python scripts/05_invariance.py --run "$run"
step forecast forecast_all_adapters
step sweep sweep_and_evaluate
step localisation python scripts/08_localisation.py --run "$run" --mu 0.1
finish_steps
