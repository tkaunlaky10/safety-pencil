#!/usr/bin/env bash
# The pencil analysis for one run ($STAGE_ARG: A, or D if Gate 1 fails) on an L4: up to 9 h, about $7.
#
#   1. K-FAC factors: harmful prompts at θ0 (en, hi, and Hinglish when its prompts exist) and task data at the
#      run's final adapter, one row set ("full") per side: ~220 GB, so launch.py refuses smaller disks.
#   2. Leverage map from the generalised eigenproblem F_h v = λ H_t v (P5).
#   3. μ sweep with the trust region and the task-gradient diagnostic (P3). Every μ in pencil.mu_grid is
#      saved as a merged model, probe-evaluated, then deleted (7 GB each).
#   4. Invariance under the V/O gauge (P1).
#   5. Weight-space baselines (P6): the must-run set, then the nice set, in separate result files.
#   6. Localisation: refusal-direction cosine, targeted vs random repair (P5).
#   7. Forecast ½ΔᵀF_hΔ for every run's adapters against their measured refusal drop (P2).
#
# Results and figures are pushed after every step, so a crash loses nothing already done. Factor tensors,
# merged checkpoints and generations never leave the job (jobs/sync.py refuses them).
set -euo pipefail
run=$STAGE_ARG
row_sets=full  # one row set per side: all four don't fit the disk (PREREGISTRATION.md deviation log)
must_baselines=(wiseft resta safelora safemerge regmean diag_fisher)
nice_baselines=(alignmerge safe_delta task_arithmetic lox)  # fisher_standard needs factors we don't collect
forecast_runs=(A C D)  # the runs with adapters; B was cut
probe_wait_minutes=180

push_outputs() {  # every result and figure so far
  python jobs/sync.py push outputs/results outputs/figures
}

push_factor_meta() {  # each factor pass's meta.json (rows, settings), never the tensors
  for meta in outputs/factors/*/*/meta.json; do
    if [[ -f $meta ]]; then
      python jobs/sync.py push "$meta"
    fi
  done
}

evaluate_probe() {  # evaluate_probe CHECKPOINT OUT_NAME: probe tier; only the result JSON is pushed
  python scripts/02_evaluate.py --checkpoint "$1" --tier probe --out-name "$2" --langs "${langs[@]}"
  python jobs/sync.py push "outputs/results/$2.json"
}

# 1. Inputs, then the factors
python jobs/sync.py pull "data/processed/**" "outputs/ckpt/$run/**" "outputs/results/**"
langs=(en hi)
if [[ -f data/processed/harmful/hinglish/factor.jsonl ]]; then
  langs+=(hinglish)
else
  echo "no Hinglish prompts: factors and evaluations in en and hi only"
fi
for lang in "${langs[@]}"; do
  python scripts/04_factors.py --side harm --lang "$lang" --row-sets "$row_sets"
done
python scripts/04_factors.py --side task --run "$run" --row-sets "$row_sets"
push_factor_meta

# 2. Leverage map
python scripts/04_leverage.py --run "$run"
push_outputs

# 3. μ sweep, then each repaired model
read -ra mu_grid <<< "$(python -c 'from pencil.config import load_config; print(*load_config()["pencil"]["mu_grid"])')"
python scripts/06_repair.py sweep --run "$run" --grad-diag --save "${mu_grid[@]}"
push_outputs
for checkpoint in outputs/ckpt/repaired/run"$run"_*_mu*; do
  if [[ ! -d $checkpoint ]]; then
    continue
  fi
  mu=${checkpoint##*_mu}
  evaluate_probe "$checkpoint" "probe_repair_run${run}_mu${mu}"
  rm -rf "$checkpoint"
done

# 4. Invariance
python scripts/05_invariance.py --run "$run"
push_outputs

# 5. Baselines
python scripts/07_baselines.py --run "$run" --methods "${must_baselines[@]}" --out-name "baselines_run${run}_must"
push_outputs
python scripts/07_baselines.py --run "$run" --methods "${nice_baselines[@]}" --out-name "baselines_run${run}_nice"
push_outputs

# 6. Localisation
python scripts/08_localisation.py --run "$run" --mu 0.1
push_outputs

# 7. Forecast, once θ0's and every adapter's probe results are in
adapters=()
probes=(outputs/results/probe_theta0.json)
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
push_outputs
