# safety-pencil

Locating, forecasting and repairing multilingual safety drift in Tiny Aya Global with a gauge-invariant curvature lens. Lexsi AI Safety and Alignment Hackathon, Track 2b.

The method code, tests and pipeline scripts are in place. All of it runs end to end on a tiny offline Cohere2 model (`bash scripts/dry_run_all.sh`, about a minute on CPU). None of it has run on Tiny Aya yet: the models are gated, and the GPU runs are the team's next step. The schedule is `03-team/PLAN_5DAY.md` in the team folder.

## Layout

| Path | What it is |
| --- | --- |
| `CLAUDE.md` | Project memory: maths, conventions, repo facts |
| `IMPLEMENTATION.md` | How each stage is built on SafeTune, CircuitKIT, AuditKIT and CuratorKIT |
| `PREREGISTRATION.md` | Predictions P1–P7 and pass criteria |
| `configs/experiment.yaml` | Models, data, runs A–D, LoRA, factor row sets, pencil settings, μ grid, evaluation tiers |
| `src/pencil/` | The library (below) |
| `scripts/` | One script per stage, every one with `--dry-run` |
| `tests/` | 130 tests: numpy maths, torch equivalence, factor hooks, whole-model repair and invariance, streaming factors, data, evaluation, statistics |
| `patches/`, `scripts/setup_lexsi.sh` | Our bug fixes to the four Lexsi libraries, and the script that installs the patched versions |
| `verification/` | The toy-transformer verification: write-up, scripts, results |

`src/pencil/`:

| Module | What it does |
| --- | --- |
| `modules.py` | Module names and structure: which linear layers exist, which share an input factor, layer order |
| `chat.py` | Chat templating with user and response offsets; masks for the four row sets |
| `factors.py` | `KFACCollector`: hooks, sampled-label Fisher, row sets `loss`/`all`/`full`/`full_reduce` |
| `factor_store.py` | `Factors` (in memory, saved per layer chunk), `FactorStore` (reads chunk files on demand) and `LazyFactors` (pools languages, projects out the known null vectors, applies transforms, one layer at a time) |
| `geneig.py` | Generalised eigenproblems on the range of F_h + F_t, covariant or Euclidean damping, noise floors; a fast top-k solver for the MLP-width sides |
| `repair.py` | Layer repair (subtraction form, truncation), forecast, frontier, `ModelRepair` over a whole model (compact cached bases, so a μ sweep is cheap), rule-11 trust region, task-gradient diagnostic |
| `delta.py` | ΔW from a PEFT adapter in fp32, dense or low-rank (`LoRADelta`); reading, merging and writing weights; saving checkpoints |
| `leverage.py` | Leverage map, per-head and per-KV-group bounds, refusal-direction cosine, targeted and random module sets |
| `invariance.py` | V/O gauge for GQA on weights, deltas and factors; output KL |
| `baselines.py` | RegMean (one eigendecomposition per μ sweep), diagonal Fisher merge, AlignMerge-style merge, Safe LoRA projection, Safe Delta Hessians |
| `safetune_plugin.py` | `PencilTrainer`, registered as SafeTune recover method `pencil` |
| `datasets.py`, `generate.py`, `judge.py`, `evaluate.py`, `stats.py` | Data layout and splits, cached generation, Llama Guard 3 and Aya Expanse judges, per-language evaluation, statistics |
| `config.py`, `runtime.py`, `results.py` | Config and `.env`; shared script plumbing (models, data, factors, the dev probe); one JSON per result |
| `dryrun.py`, `testing.py`, `plots.py` | The offline dry run; tiny Cohere2 model and tokenizer; figure helpers |

## Setup

On the GPU machine:

```bash
git clone <repo url> && cd safety-pencil
cp .env.example .env            # then add HF_TOKEN
python3.12 -m venv .venv && . .venv/bin/activate && pip install -U pip uv
uv pip install -r requirements-main.lock.txt
bash scripts/setup_lexsi.sh main            # swaps in the patched SafeTune, AuditKIT, CuratorKIT
uv pip install -e . --no-deps               # this package (the scripts import it)
python -m pytest -q                         # 129 passed, 1 skipped (CPU, about 2 minutes)
bash scripts/dry_run_all.sh                 # the whole pipeline on the tiny offline model
```

Style: `uv pip install ruff && ruff check . && ruff format --check .` (config in `pyproject.toml`: 120 columns; the patched Lexsi clones in `third_party/` are excluded).

For CircuitKIT, make `.venv-circuit` the same way from `requirements-circuit.lock.txt` and run `bash scripts/setup_lexsi.sh circuit`. Accept the Hugging Face licences for `CohereLabs/tiny-aya-global`, `CohereLabs/tiny-aya-base`, `meta-llama/Llama-Guard-3-8B` and `CohereLabs/aya-expanse-8b` first.

## Running the pipeline

Each command writes one JSON per result to `outputs/results/` and figures to `outputs/figures/`. `outputs/` is gitignored, because it holds checkpoints, factors and raw generations.

```bash
python scripts/00_model_facts.py                                   # configs/model_facts.json
python scripts/01_build_harmful.py && python scripts/01_make_hinglish.py \
  && python scripts/01_build_xstest.py && python scripts/01_build_task_t1.py \
  && python scripts/01_build_mmlu.py && python scripts/01_check_contamination.py   # data/processed/
python scripts/03_finetune.py --run A                              # outputs/ckpt/A/frac{25,50,75,100}/
python scripts/02_evaluate.py --checkpoint CohereLabs/tiny-aya-global --tier probe --out-name probe_theta0
python scripts/02_evaluate.py --checkpoint outputs/ckpt/A/frac100 --tier probe --out-name probe_runA_frac100

# factors: first the row-set selection pass on 4 layers, then every layer with the chosen row sets
python scripts/04_factors.py --side harm --lang en --layers selection
python scripts/04_factors.py --side task --run A --layers selection
python scripts/06_repair.py sweep --run A --layers selection --row-set all    # repeat per row set; pick on dev
python scripts/04_factors.py --side harm --lang hi --row-sets full,full_reduce
python scripts/04_leverage.py --run A
python scripts/05_invariance.py --run A
python scripts/06_repair.py sweep --run A --save 0.1               # then 02_evaluate on the saved checkpoint
python scripts/06_repair.py forecast                               # needs probe_run{R}_frac{F} results
python scripts/07_baselines.py --run A
python scripts/08_localisation.py --run A
```

The CircuitKIT cross-check (Prompt 8 step 5) is not scripted yet. It runs in `.venv-circuit`, following `IMPLEMENTATION.md` Stage 7.

## Resources at Tiny Aya scale

These are estimates for an L4 (24 GB) with intermediate_size about 11008. `00_model_facts.py` records the real value.
- **Factor files:** saved in fp32, about 52 GB on disk per side and row set for all 36 layers. Choose the row sets on the 4-layer selection pass (`--layers selection` writes to `outputs/factors_selection/`), then run every layer with one or two row sets.
- **The repair:** reads factors one layer at a time and keeps only compact bases (top 512 pairs on MLP-width sides) in float32, about 12 GB of RAM. The MLP-width pencils use a Cholesky-plus-one-eigh solver on the GPU, about 10 s each instead of minutes. A μ sweep reuses the bases.
- **The sweep:** holds two bf16 models on the GPU (θ0 and the working copy, 14 GB). The invariance demo holds one model in fp32 (13 GB).

## What the tiny-model tests found

These come from one tiny random Cohere2 model with a Tiny-Aya-style preamble, so treat them as hypotheses to check on Tiny Aya dev, not results (`tests/test_model_repair.py`).

- **The preamble matters for the curvature.** Tiny Aya's fixed system preamble is most of every sequence, and k_proj and v_proj at preamble positions feed every later token. Factors that leave the preamble out (`loss`, `all`) under-forecast v_proj drift by 60–1000×. `full` (preamble included) fixes most of it, and K-FAC-reduce on v_proj (`full_reduce`) gets it right. That hybrid ranked per-module drift best (Spearman 0.97, against 0.84 for `all`).
- **Reduce factors are for the forecast only.** As a repair basis, `full_reduce` made v_proj drift worse, because its factors have rank at most the number of sequences. The repair uses `full`.
- **The step-ratio check is stricter than the measured drift.** The covariant repair on `full` cut measured drift to 15% while some o_proj step ratios were above 20, along directions the inputs barely visit. Rule 11 as written would switch to Euclidean damping there. The sweep logs an input-weighted step ratio and the measured S next to it, so the team can decide on dev.
- **The known null vectors are projected out.** The shared q/k/v/gate/up input has the null vector 1/w, and o_proj and down_proj outputs have the all-ones null vector, because every path to the loss passes a mean-subtracting LayerNorm. In fp32 factors these are only null to about 1e-8. The repair projects them out exactly and sets the range threshold from their measured floor.

## Licences

- Our code: to be decided (for example MIT).
- Tiny Aya derivatives: CC-BY-NC 4.0 plus Cohere Labs' acceptable-use policy.
- The Lexsi libraries: LSAL. See `IMPLEMENTATION.md` §0.
