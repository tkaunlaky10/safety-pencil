# safety-pencil

Locating, forecasting and repairing multilingual safety drift in Tiny Aya Global with a gauge-invariant curvature lens. Lexsi AI Safety and Alignment Hackathon, Track 2b.

This is the scaffold. Claude Code fills in the rest from `CLAUDE.md` when you paste Prompt 0 from `02-code-setup/PROMPTS.md` in the team folder. The schedule is `03-team/PLAN_5DAY.md` in the team folder.

## What is here

| Path | What it is |
| --- | --- |
| `CLAUDE.md` | Project memory for Claude Code: maths, conventions, repo facts |
| `IMPLEMENTATION.md` | How each stage is built on SafeTune, CircuitKIT, AuditKIT and CuratorKIT |
| `PREREGISTRATION.md` | Predictions P1–P7 and pass criteria, with the 9 Nov scope changes |
| `requirements-*.in`, `requirements-*.lock.txt` | Pinned environments (Lexsi repos at fixed commits, transformers 5.21.1) |
| `tests/test_math.py` | 18 numpy checks of the maths; the torch code must match them |
| `patches/`, `scripts/setup_lexsi.sh` | Our bug fixes to the four Lexsi libraries, and the script that installs the patched versions |
| `verification/` | The toy-transformer verification: write-up, scripts, results |

## First-time setup

On your Mac, from this folder:

```bash
git init -b main          # already done; harmless to rerun
git add .
git commit -m "Scaffold: project memory, implementation map, pinned environments, maths tests"
gh repo create safety-pencil --private --source . --push   # or create it on github.com, then: git remote add origin <url> && git push -u origin main
```

Add both teammates as collaborators. The repo must be public by submission.

On the GPU machine:

```bash
git clone <repo url> && cd safety-pencil
cp .env.example .env            # then add HF_TOKEN
python3.12 -m venv .venv && . .venv/bin/activate && pip install -U pip uv
uv pip install -r requirements-main.lock.txt
bash scripts/setup_lexsi.sh main            # swaps in the patched SafeTune, AuditKIT, CuratorKIT
python -m pytest -q tests/test_math.py      # 18 passed
deactivate
python3.12 -m venv .venv-circuit && . .venv-circuit/bin/activate && pip install -U pip uv
uv pip install -r requirements-circuit.lock.txt
bash scripts/setup_lexsi.sh circuit         # swaps in the patched CircuitKIT
```

`scripts/setup_lexsi.sh` clones each Lexsi repo at its pinned commit into `third_party/` (gitignored), applies `patches/<Repo>.patch` and reinstalls it in editable mode without touching the locked dependencies. `patches/README.md` lists what each patch fixes.

Accept the Hugging Face licences for `CohereLabs/tiny-aya-global`, `CohereLabs/tiny-aya-base`, `meta-llama/Llama-Guard-3-8B` and `CohereLabs/aya-expanse-8b` first. Then open Claude Code in this folder and paste Prompt 0.

## Licences

- Our code: to be decided (for example MIT).
- Tiny Aya derivatives: CC-BY-NC 4.0 plus Cohere Labs' acceptable-use policy.
- The Lexsi libraries: LSAL. See `IMPLEMENTATION.md` §0.
