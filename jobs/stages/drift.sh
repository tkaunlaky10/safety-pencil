#!/usr/bin/env bash
# Measured drift S on dev for every run's adapters (P2's measured side) on an L4: about 1 h, under $1.
# Needs no factors: θ0 + each adapter in turn, KL to θ0 on θ0's own answers to the dev harmful prompts, per
# language. jobs/report.py ranks the forecast (forecast_p2.json, from the pencil stage) against it.
set -euo pipefail
forecast_runs=(A C D)  # the runs with adapters; B was cut

adapters=()
for run in "${forecast_runs[@]}"; do
  adapters+=("outputs/ckpt/$run/**")
done
python jobs/sync.py pull "data/processed/**" "${adapters[@]}"
python scripts/06_repair.py drift
python jobs/sync.py push outputs/results
