#!/usr/bin/env bash
# Run every pipeline stage on the tiny offline model (CPU, about a minute). Outputs go to outputs/dryrun/.
# Use it after any change to src/pencil or scripts/, before spending GPU time.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf outputs/dryrun
py() { python "$@" --dry-run; }
py scripts/00_model_facts.py
py scripts/03_finetune.py --run A
py scripts/03_finetune.py --run D
for l in en hi hinglish; do py scripts/04_factors.py --side harm --lang "$l"; done
py scripts/04_factors.py --side task --run A
py scripts/04_leverage.py --run A
py scripts/05_invariance.py --run A --conds 10
py scripts/06_repair.py sweep --run A --mu 0.01 0.1 1.0
py scripts/06_repair.py forecast
py scripts/07_baselines.py --run A --methods wiseft resta safelora regmean diag_fisher alignmerge safe_delta
py scripts/08_localisation.py --run A
echo "dry run complete: outputs/dryrun/"
