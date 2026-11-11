#!/usr/bin/env bash
# Runs inside a Hugging Face Job started by jobs/launch.py, from the downloaded code in /work/code.
#
# 1. Installs the pinned environment: the lock file, then the patched Lexsi libraries (scripts/setup_lexsi.sh).
# 2. On a GPU machine, makes sure torch can see the GPU (falls back to an older CUDA build of the same torch).
# 3. Runs jobs/stages/$STAGE.sh. Each stage pulls its inputs from $RUNS_REPO and pushes its outputs there.
# 4. If the stage succeeded, starts the stages listed in $THEN ("stage" or "stage:arg", space-separated).
# The run ledger (jobs/ledger.py) records the job in the runs repo at its start and its end, failures included,
# with its log and a copy of every result it wrote.
set -euo pipefail
cd "$(dirname "$0")/.."

run_id="$(date -u +%Y%m%dT%H%M%SZ)-${STAGE}"
if [[ -n ${STAGE_ARG:-} ]]; then
  run_id="${run_id}-${STAGE_ARG}"
fi
export PENCIL_RUN_ID=$run_id  # results.write_result stamps it into every result
mkdir -p "outputs/runs/$run_id"
exec > >(tee -a "outputs/runs/$run_id/log.txt") 2>&1  # everything below is also kept in the job's log file

ledger() {  # runs before the environment is installed, so with its own huggingface_hub
  uv run --quiet --no-project --with "$HUB_CLIENT" python jobs/ledger.py "$@"
}

finish_ledger() {  # the EXIT trap: record the outcome, then exit with the stage's status, recorded or not
  local status=$?
  ledger finish "$status" || echo "ledger: finish not recorded"
  exit "$status"
}

echo "== stage ${STAGE} ${STAGE_ARG:-} | code ${PENCIL_GIT_HASH:-unknown} | run ${run_id}"
if command -v nvidia-smi > /dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
  export GPU_NAME
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi
df -h . | tail -1
# Logging must never stop an experiment: a failed ledger call is reported and the job carries on.
ledger start || echo "ledger: start not recorded"
trap finish_ledger EXIT

uv venv --quiet --python 3.12 .venv
. .venv/bin/activate
uv pip install --quiet -r requirements-main.lock.txt
bash scripts/setup_lexsi.sh main
uv pip install --quiet --no-deps -e .

torch_sees_gpu() {
  python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
}

# PyPI's torch is built for a recent CUDA; if the machine's driver is older, use an older CUDA build.
if command -v nvidia-smi > /dev/null && ! torch_sees_gpu; then
  version=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  for cuda in cu128 cu126; do
    echo "torch $version can't see the GPU; trying its $cuda build"
    if ! uv pip install --quiet --reinstall-package torch "torch==$version" \
      --index-url "https://download.pytorch.org/whl/$cuda"; then
      continue
    fi
    if torch_sees_gpu; then
      break
    fi
  done
  if ! torch_sees_gpu; then
    echo "no torch build can see the GPU" >&2
    exit 1
  fi
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())"

bash "jobs/stages/${STAGE}.sh"

for next in ${THEN:-}; do
  stage=${next%%:*}
  arg=""
  if [[ $next == *:* ]]; then
    arg=${next#*:}
  fi
  # the launcher needs the same huggingface_hub as on your machine, not the lock file's older one
  uv run --no-project --with "$HUB_CLIENT" python jobs/launch.py --chained "$stage" "$arg"
done
echo "== stage ${STAGE} ${STAGE_ARG:-} finished $(date -u +%FT%TZ)"
