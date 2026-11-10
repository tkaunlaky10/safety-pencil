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
# Written for bash 3.2 (the bash that ships with macOS) as well as Linux bash, so it avoids
# bash-4 features such as associative arrays (declare -A).
#
# The patched copies are modified versions of Lexsi Labs software under LSAL; see
# patches/README.md for what each patch changes.
set -euo pipefail
cd "$(dirname "$0")/.."
target=${1:-main}

# The upstream commit each library is pinned to.
pinned_commit() {
  case "$1" in
    SafeTune)   echo 1d824f026742fe2ce58491a8414f51a264931840 ;;
    AuditKIT)   echo 9da5838e61ef54d560a37605ddbcb19aceded9b3 ;;
    CuratorKIT) echo cee1aed21d14992f650c5dfe09094b9625d0ce87 ;;
    CircuitKIT) echo 227fafdaf1ae78ac1de492da07d8173d16f8863c ;;
    *) echo "no pinned commit for $1" >&2; return 1 ;;
  esac
}

# The pip extras installed with each library (empty for none).
pip_extras() {
  case "$1" in
    AuditKIT)   echo "[transformers,interop]" ;;
    CuratorKIT) echo "[hf,generation]" ;;
    *)          echo "" ;;
  esac
}

case "$target" in
  main)    repos=(SafeTune AuditKIT CuratorKIT) ;;
  circuit) repos=(CircuitKIT) ;;
  *) echo "usage: $0 main|circuit" >&2; exit 2 ;;
esac

mkdir -p third_party
for repo in "${repos[@]}"; do
  dir=third_party/$repo
  commit=$(pinned_commit "$repo")
  extras=$(pip_extras "$repo")
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "https://github.com/Lexsi-Labs/$repo.git" "$dir"
  fi
  git -C "$dir" checkout --quiet --force "$commit"
  git -C "$dir" clean --quiet -fd
  if [ -f "patches/$repo.patch" ]; then
    git -C "$dir" apply --whitespace=nowarn "$PWD/patches/$repo.patch"
    echo "patched $repo"
  fi
  uv pip install --no-deps -e "$dir$extras"
done

# Report where each library is installed; the other venv's libraries are expected to be missing.
python - <<'PY'
import importlib

for module_name in ("safetune", "auditkit", "curatorkit", "circuitkit"):
    try:
        module = importlib.import_module(module_name)
        print(f"{module_name}: {getattr(module, '__file__', '?')}")
    except Exception as error:  # the other venv's packages are expected to be missing
        print(f"{module_name}: not in this venv ({type(error).__name__})")
PY
