#!/usr/bin/env bash
# Runs inside a Hugging Face Job started by jobs/launch.py, from the downloaded code in /work/code.
#
# 1. Installs the pinned environment: the lock file, then the patched Lexsi libraries (scripts/setup_lexsi.sh).
# 2. On a GPU machine, makes sure torch can see the GPU (falls back to an older CUDA build of the same torch).
# 3. Runs jobs/stages/$STAGE.sh. Each stage pulls its inputs from $RUNS_REPO and pushes its outputs there.
# 4. If the stage succeeded, starts the stages listed in $THEN ("stage" or "stage:arg", space-separated).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== stage ${STAGE} ${STAGE_ARG:-} | code ${PENCIL_GIT_HASH:-unknown} | $(date -u +%FT%TZ)"
if command -v nvidia-smi > /dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi
df -h . | tail -1

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
