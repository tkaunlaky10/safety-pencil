"""The whole pipeline on the tiny offline model (scripts/dry_run_all.sh, about a minute on CPU).
Opt-in: PENCIL_DRY_RUN=1 pytest tests/test_dry_run.py"""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("PENCIL_DRY_RUN") != "1", reason="set PENCIL_DRY_RUN=1 to run the pipeline dry run")
def test_dry_run_all():
    script = ROOT / "scripts" / "dry_run_all.sh"
    completed = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    results_dir = ROOT / "outputs" / "dryrun" / "results"
    for name in (
        "finetune_A",
        "leverage_pooled_runA",
        "invariance_runA",
        "repair_sweep_runA_frac100_full",
        "forecast_p2",
        "baselines_runA",
        "localisation_runA",
    ):
        assert (results_dir / f"{name}.json").exists(), name
