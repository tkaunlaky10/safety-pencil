#!/usr/bin/env bash
# Shared by the stages that collect factors (pencil, compare); sourced after `set -euo pipefail`, with $run set.
#
#   pull_inputs; choose_langs; collect_factors     # inputs, then the four factor passes (~1.7 h on an L4)
#   step NAME COMMAND...                           # one analysis step: a failure is recorded, the stage goes on
#   forecast_all_adapters                          # P2: ½ΔᵀF_hΔ for runs A, C, D's adapters (needs the factors)
#   finish_steps                                   # exit non-zero if any step failed
#
# The factor passes are not steps: nothing can run without them, so a failure there stops the stage. Every step
# after them runs even if an earlier one crashed, so one bad step never throws away 1.7 h of factors.

row_sets=full  # one row set per side: all four don't fit the disk (PREREGISTRATION.md deviation log)
failed_steps=()
forecast_runs=(A C D)  # the runs with adapters; B was cut
probe_wait_minutes=180

push_outputs() {  # every result and figure so far; a failed push is reported, never fatal
  python jobs/sync.py push outputs/results outputs/figures || echo "push of results failed (carrying on)"
}

push_factor_meta() {  # each factor pass's meta.json (rows, settings), never the tensors
  for meta in outputs/factors/*/*/meta.json; do
    if [[ -f $meta ]]; then
      python jobs/sync.py push "$meta"
    fi
  done
}

pull_inputs() {  # the data, this run's adapters and every result so far
  python jobs/sync.py pull "data/processed/**" "outputs/ckpt/$run/**" "outputs/results/**"
}

choose_langs() {  # sets $langs: en and hi, and Hinglish when its prompts exist
  langs=(en hi)
  if [[ -f data/processed/harmful/hinglish/factor.jsonl ]]; then
    langs+=(hinglish)
  else
    echo "no Hinglish prompts: factors and evaluations in en and hi only"
  fi
}

collect_factors() {  # harmful factors at θ0 per language, task factors at the run's final adapter
  for lang in "${langs[@]}"; do
    python scripts/04_factors.py --side harm --lang "$lang" --row-sets "$row_sets"
  done
  python scripts/04_factors.py --side task --run "$run" --row-sets "$row_sets"
  push_factor_meta
}

evaluate_checkpoint() {  # evaluate_checkpoint CHECKPOINT OUT_NAME TIER: only the result JSON is pushed
  python scripts/02_evaluate.py --checkpoint "$1" --tier "$3" --out-name "$2" --langs "${langs[@]}"
  python jobs/sync.py push "outputs/results/$2.json"
}

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

step() {  # step NAME COMMAND...: run COMMAND in a subshell with errexit on; record a failure and carry on
  local name=$1
  shift
  echo "== step $name"
  set +e  # (set -e is ignored inside an `if` or `||`, so the status is read this way instead)
  (
    set -e
    "$@"
  )
  local status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "== step $name failed (exit $status); carrying on"
    failed_steps+=("$name")
  fi
  push_outputs
}

finish_steps() {  # the stage's verdict: non-zero if any step failed, naming them
  if [[ ${#failed_steps[@]} -gt 0 ]]; then
    echo "== failed steps: ${failed_steps[*]}"
    exit 1
  fi
  echo "== every step succeeded"
}
