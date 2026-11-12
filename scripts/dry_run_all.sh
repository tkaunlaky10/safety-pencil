#!/usr/bin/env bash
# Run every pipeline stage on the tiny offline model (CPU, about a minute). Outputs go to outputs/dryrun/.
# Use it after any change to src/pencil or scripts/, before spending GPU time.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf outputs/dryrun

# Run one script with the given arguments plus --dry-run.
dry_run() { python "$@" --dry-run; }

dry_run scripts/00_model_facts.py
dry_run scripts/03_finetune.py --run A
dry_run scripts/03_finetune.py --run D
for lang in en hi hinglish; do
  dry_run scripts/04_factors.py --side harm --lang "$lang"
done
dry_run scripts/04_factors.py --side task --run A
dry_run scripts/04_leverage.py --run A
dry_run scripts/05_invariance.py --run A --conds 10
dry_run scripts/06_repair.py sweep --run A --mu 0.01 0.1 1.0
dry_run scripts/06_repair.py forecast
dry_run scripts/06_repair.py drift
dry_run scripts/07_baselines.py --run A --methods wiseft resta safelora regmean diag_fisher alignmerge safe_delta
dry_run scripts/08_localisation.py --run A
echo "dry run complete: outputs/dryrun/"
