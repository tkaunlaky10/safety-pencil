#!/usr/bin/env bash
# Install the four Lexsi libraries from their pinned upstream commits plus our patches.
#
# Run after `uv pip install -r requirements-main.lock.txt` (or the circuit lock) in the
# matching venv. It clones each repo into third_party/ (gitignored), checks out the pinned
# commit, applies patches/<Repo>.patch, and reinstalls the package editable with --no-deps,
# so every dependency version still comes from the lock file.
#
#   . .venv/bin/activate         && bash scripts/setup_lexsi.sh main      # SafeTune, AuditKIT, CuratorKIT
#   . .venv-circuit/bin/activate && bash scripts/setup_lexsi.sh circuit   # CircuitKIT
#
# The patched copies are modified versions of Lexsi Labs software under LSAL; see
# patches/README.md for what each patch changes.
set -euo pipefail
cd "$(dirname "$0")/.."
which=${1:-main}

declare -A SHA=(
  [SafeTune]=1d824f026742fe2ce58491a8414f51a264931840
  [AuditKIT]=9da5838e61ef54d560a37605ddbcb19aceded9b3
  [CuratorKIT]=cee1aed21d14992f650c5dfe09094b9625d0ce87
  [CircuitKIT]=227fafdaf1ae78ac1de492da07d8173d16f8863c
)
declare -A EXTRAS=(
  [SafeTune]="" [AuditKIT]="[transformers,interop]" [CuratorKIT]="[hf,generation]" [CircuitKIT]=""
)
case "$which" in
  main)    repos=(SafeTune AuditKIT CuratorKIT) ;;
  circuit) repos=(CircuitKIT) ;;
  *) echo "usage: $0 main|circuit" >&2; exit 2 ;;
esac

mkdir -p third_party
for r in "${repos[@]}"; do
  dir=third_party/$r
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "https://github.com/Lexsi-Labs/$r.git" "$dir"
  fi
  git -C "$dir" checkout --quiet --force "${SHA[$r]}"
  git -C "$dir" clean --quiet -fd
  if [ -f "patches/$r.patch" ]; then
    git -C "$dir" apply --whitespace=nowarn "$PWD/patches/$r.patch"
    echo "patched $r"
  fi
  uv pip install --no-deps -e "$dir${EXTRAS[$r]}"
done
python - <<'PY'
import importlib
for m in ("safetune", "auditkit", "curatorkit", "circuitkit"):
    try:
        mod = importlib.import_module(m)
        print(f"{m}: {getattr(mod, '__file__', '?')}")
    except Exception as e:  # the other venv's packages are expected to be missing
        print(f"{m}: not in this venv ({type(e).__name__})")
PY
