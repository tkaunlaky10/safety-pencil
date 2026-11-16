# safety-pencil

Locating, forecasting and repairing multilingual safety drift in Tiny Aya Global with a gauge-invariant curvature
lens. Lexsi AI Safety and Alignment Hackathon, **Track 2b: AI Safety and Correction**.

**Team:** Tanay Kulkarni (@tkaunlaky10), Mukund Garg, Abdul Jawad Khan (HF @obluda420).

**Write-up:** [WRITEUP.md](WRITEUP.md) · [PDF](Safety_Pencil_WRITEUP.pdf) (the maths and figures render best in the PDF).

**Judges, start here:** [`WRITEUP.md`](WRITEUP.md) has a one-minute summary at the top, then the full submission. The results table is in [Results](#results) below; every prediction and every deviation from it is in [`PREREGISTRATION.md`](PREREGISTRATION.md).

> **Research artifact.** Every fine-tuned or repaired checkpoint this code produces is labelled
> "research artifact: unrepaired, do not deploy". No raw generations or harmful completions are in this repo.

- **Write-up:** [`WRITEUP.md`](WRITEUP.md), the full submission: problem, data, metrics, method, results,
  limitations.
- **The maths, step by step:** [`docs/MATH_STEP_BY_STEP.md`](docs/MATH_STEP_BY_STEP.md), every idea below with
  worked examples. Damping and the eigenvalue ceiling are
  [part 6](docs/MATH_STEP_BY_STEP.md#6-damping-range-restriction-truncation-and-the-ceiling).
- **Pre-registration and deviation log:** [`PREREGISTRATION.md`](PREREGISTRATION.md) (predictions P1–P7, written
  before measurement).
- **Run log:** [`EXPERIMENTS.md`](EXPERIMENTS.md) (every GPU job, its code version and its outcome).
- **Public artifacts on Hugging Face:** [`safety-pencil-data`](https://huggingface.co/datasets/tkaunlaky10/safety-pencil-data) (data splits), [`safety-pencil-results`](https://huggingface.co/datasets/tkaunlaky10/safety-pencil-results) (every result JSON, run logs, figures), LoRA adapters [`safety-pencil-lora-runA`](https://huggingface.co/tkaunlaky10/safety-pencil-lora-runA), [`-runC`](https://huggingface.co/tkaunlaky10/safety-pencil-lora-runC), [`-runD`](https://huggingface.co/tkaunlaky10/safety-pencil-lora-runD).
  - Repaired model to use: [`tkaunlaky10/safety-pencil-runA-wiseft-0.5`](https://huggingface.co/tkaunlaky10/safety-pencil-runA-wiseft-0.5) (WiSE-FT α = 0.5 on run A; see WRITEUP §6.7)

## The problem

Fine-tuning an aligned model on a *harmless* task can still damage its safety. We fine-tuned Tiny Aya Global
(CohereLabs/tiny-aya-global) with LoRA on 4000 benign Hindi instructions from aya_dataset. Task loss fell from 2.14
to 1.78 nats per token and MMLU did not move, but on the held-out test split:

| | English | Hindi | Hinglish |
| --- | --- | --- | --- |
| Harmful compliance (Llama Guard 3 "unsafe") | **7% → 22%** (Holm p 7e-15) | 6% → 8% (n.s.) | 23% → 22% (n.s.) |
| Refusal (Aya Expanse judge) | 60% → 73% | 51% → 65% | 19% → 44% |
| Over-refusal of safe prompts (XSTest) | **14% → 55%** | 4% → 10% | 12% → 31% |

The model refuses more *and* complies with harmful requests more: it has lost its ability to tell harmful from
harmless prompts. The project asks three questions about that damage:

1. **Locate:** which layers and modules carry the safety behaviour the fine-tune damaged?
2. **Forecast:** can we predict the safety cost of an update from the update alone, before any evaluation?
3. **Repair:** can we turn the update into a better one that removes the safety damage and keeps the task gain,
   in closed form, with no retraining?

## Problem statements addressed

| Statement | What it asks | What we did | Outcome |
| --- | --- | --- | --- |
| **Track 2b #6** (primary) | Repair safety drift without retraining | Closed-form curvature repair (the "pencil"), bounded by a per-module trust region, against WiSE-FT, RegMean and diagonal Fisher at equal task score | **Done.** Restores English harmful compliance to the aligned model's 7% with the full task gain. Ties RegMean on harmful compliance, and plain interpolation (WiSE-FT α = 0.5) matches it on harmful compliance and worst-language refusal and halves its English over-refusal. The pre-registered comparison (P6) is **not met in substance** (see Results) |
| **Track 2b #9** | Locate where safety lives in the model | A leverage map: each module's largest safety-per-task-cost eigenvalue | **Partly done.** The top 10 modules are all in layers 32–35, and 9 of them are MLP projections, as predicted. The refusal-direction check and the CircuitKIT cross-check were not run |
| **Track 2a #4** (lens) | Multilingual safety | Every measurement is per language: English, Hindi and romanised Hinglish (built from aya_redteaming) | **Done.** The drift is language-dependent: it shows up as harmful compliance in English, and as extra refusal and over-refusal in Hindi and Hinglish |
| Track 2b #5 | Safety-preserving fine-tuning | Planned as P4 ("preserve ≈ repair") | **Not done** (cut on 9 Nov) |
| Track 2b #11 | Compression | Stretch goal | **Not done** |

## The idea in one paragraph

A fine-tune adds an update Δ to the aligned weights θ0. Measure two curvatures over weight space: **F_h**, how
much each weight direction changes the aligned model's answers to harmful prompts (the Fisher of the KL drift,
measured at θ0), and **H_t**, how much it changes the task loss (the task Fisher, measured at the fine-tuned θ1).
Solve one generalised eigenproblem, **F_h v = λ H_t v**. Each eigenvalue λ is *safety damage per unit of task cost*
along its direction v. Then:

- **locate:** the modules with the largest λ are where safety is most sensitive and the task least;
- **forecast:** S(θ1) ≈ ½ Σ λᵢ cᵢ², where c are Δ's coordinates in the eigenbasis;
- **repair:** shrink each coordinate by μ/(λᵢ + μ). Safety-expensive directions go, task directions stay, and μ sets
  the exchange rate. This is the exact minimiser of drift + μ·regret, a Fisher-weighted merge of θ0 and θ1.

WiSE-FT (interpolating back toward θ0) shrinks every direction by the same factor; the pencil shrinks each one
according to its own λ. In a toy 2-D case the pencil leaves 7.6× less drift than WiSE-FT at the same task cost.

## How the method is built (and where each part is explained)

| Part of [`MATH_STEP_BY_STEP.md`](docs/MATH_STEP_BY_STEP.md) | Topic | In the code |
| --- | --- | --- |
| 1–2 | Drift S = KL(θ0 ‖ θ) on θ0's own answers; its curvature is the Fisher F_h. The task Fisher H_t and the regret R at θ1 | `invariance.output_kl`, `runtime.measure_S`, `factors.py` |
| 3–4 | The pencil F_h v = λ H_t v; the closed-form repair z = μ/(λ+μ)·c; the drift–task frontier | `geneig.py`, `repair.py` |
| 5 | K-FAC: each layer's Fisher ≈ A ⊗ G (input statistics ⊗ output-gradient statistics), so a layer's pencil splits into two small ones with eigenvalues λᵢⱼ = αᵢγⱼ | `factors.py`, `factor_store.py` |
| 6 | Covariant damping A_t + γ_c A_h (γ_c = 1e-3) caps each side's eigenvalue at 1000 and keeps the symmetry below; the cap doubles as a numerical validity check. Range restriction, known LayerNorm null vectors, truncation, jitter | `geneig.py`, `factor_store.LazyFactors` |
| 7 | **Invariance (P1):** grouped-query attention has a V/O gauge symmetry (W_v → M W_v, W_o → W_o M⁻¹). The pencil repair gives exactly the same function in every gauge; Euclidean "safety subspace" methods do not | `invariance.py`, `scripts/05_invariance.py` |
| 8 | **Leverage map (P5):** the per-module largest eigenvalue | `leverage.py`, `scripts/04_leverage.py` |
| 9 | Why the first full-scale repair broke the model on 12 Nov: nearly parallel eigenvectors made it an oblique projector, with steps up to 94× the fine-tune | `EXPERIMENTS.md`, 12 Nov |
| 10 | The fix: output-side damping plus a per-module trust region (each step scaled so its effect on the module's real inputs is at most the fine-tune's). Still exactly gauge-invariant | `repair.scaled_trust_region_repair` |
| 11 | Baselines as other curvature models in the same objective: WiSE-FT (all λ equal), RegMean (input side only), diagonal Fisher | `baselines.py`, `scripts/07_baselines.py` |
| 12 | Statistics: Wilson intervals, exact McNemar, Holm across languages, paired bootstrap over prompts for drift | `stats.py` |
| 13–14 | Results and honest takeaways | below, and [`WRITEUP.md`](WRITEUP.md) |

## Results

Pre-registered predictions ([`PREREGISTRATION.md`](PREREGISTRATION.md)):

| | Prediction | Outcome |
| --- | --- | --- |
| P1 | The pencil repair is invariant under the V/O gauge; Euclidean methods are not | **Pass.** Pencil, RegMean and ReSta give the same function in both gauges (output KL ≤ 2e-9); Euclidean pencil, diagonal Fisher, AlignMerge-style and Safe LoRA do not (KL 6e-7 to 1e-3) |
| P2 | ½ΔᵀF_hΔ ranks measured drift (Spearman ρ ≥ 0.6) | **Fail.** ρ = 0.17 [−0.17, 0.49]; forecasts are 24–39× too low on run A's final checkpoint (24–111× across all 36 checkpoint–language pairs). Drift along the fine-tune grows like (1 − α)^0.72 [0.71, 0.74], not the quadratic (1 − α)² |
| P3 | Refusal and task score move with μ as predicted | **Fail** for the unbounded repair (every μ rejected by the trust-region rule). Bounded repair: drift falls with μ as predicted; task loss does not follow the predicted regret |
| P4 | Preserve ≈ repair | Not run |
| P5 | Leverage concentrates in late MLP blocks and aligns with the refusal direction | **Partial.** Top 10 modules all in layers 32–35, 9 of them MLP; alignment not measured |
| P6 | The pencil beats at least one weight-space baseline on worst-language refusal at equal task score | **Not met in substance.** The fine-tune raised refusal, so on this metric the unrepaired model (44.4% Hinglish) beats the aligned one (19.2%); the pencil's extra Hinglish refusal over RegMean (39.2% vs 34.3%) comes with more over-refusal (34.8% vs 28.3%). Not significantly better than WiSE-FT α = 0.5 on any safety metric; behind all three baselines on dev KL drift ([`WRITEUP.md`](WRITEUP.md) §5.9) |
| P7 | Per-language pencils share a subspace | Not run |

**The repair on test** (bounded pencil at μ* = 0.0019, versus the fine-tuned model θ1 and the aligned model θ0):

| | θ0 | θ1 | Pencil μ* | RegMean (matched) | WiSE-FT α = 0.5 |
| --- | --- | --- | --- | --- | --- |
| Harmful compliance en / hi / Hinglish | 7 / 6 / 23% | 22 / 8 / 22% | **7 / 6 / 23%** | 6 / 5 / 24% | 5 / 5 / 21% |
| Refusal en / hi / Hinglish | 60 / 51 / 19% | 73 / 65 / 44% | 62 / 58 / 39% | 64 / 61 / 34% | 67 / 62 / 38% |
| Over-refusal, English | 14% | 55% | 32% | 24% | **16%** |
| Hindi task NLL (dev) | 2.14 | 1.79 | 1.78 | 1.78 | 1.78 |

What this means:
- **The repair works.** It returns English harmful compliance exactly to θ0's level, keeps the whole task gain and
  MMLU, and removes about half the added English over-refusal, with no retraining.
- **It is not better than the simple baselines.** RegMean ties it on harmful compliance. Plain interpolation at
  half strength matches it on harmful compliance and worst-language refusal, and halves its English over-refusal.
  On dev KL drift, the quantity the repair optimises, WiSE-FT, RegMean and diagonal Fisher all leave less drift at
  equal task cost.
- **What holds:** the closed-form repair and its exact gauge invariance, on the real model. **What fails:** the
  quadratic forecast (the real fine-tune is far outside the quadratic regime), and the output-side eigenbasis is
  numerically fragile.

Full tables, intervals and every caveat are in [`WRITEUP.md`](WRITEUP.md). `jobs/report.py` regenerates
`outputs/summary.md` from the result JSONs.

## Lexsi tools and models used

- **Models:** CohereLabs/tiny-aya-global (the model studied); meta-llama/Llama-Guard-3-8B and
  CohereLabs/aya-expanse-8b as judges.
- **Lexsi libraries:** SafeTune (LoRA fine-tunes and the `pencil` recover method), AuditKIT (evaluation) and
  CuratorKIT (data). Our patches are in `patches/`. CircuitKIT was set up but not run.
- **Data:** CohereLabs/aya_redteaming (harmful prompts, en and hi; Hinglish built from it), XSTest, aya_dataset /
  aya_collection (the benign Hindi task), Global-MMLU, and 20 BeaverTails rows via SafeTune for run D.

## Layout

| Path | What it is |
| --- | --- |
| `PREREGISTRATION.md` | Predictions P1–P7, pass criteria, and the dated deviation log |
| `EXPERIMENTS.md` | Every GPU job: code version, command, result names, outcome |
| `CLAUDE.md` | Project memory: maths, conventions, repo facts |
| `IMPLEMENTATION.md` | How each stage is built on SafeTune, CircuitKIT, AuditKIT and CuratorKIT |
| `configs/experiment.yaml` | Models, data, runs A–D, LoRA, factor row sets, pencil settings, μ grid, evaluation tiers |
| `src/pencil/` | The library (below) |
| `scripts/` | One script per stage, every one with `--dry-run` (below) |
| `jobs/` | Hugging Face Jobs: `launch.py` (stages, hardware, cost), `stages/*.sh`, `sync.py`, `report.py` |
| `tests/` | 270 tests: numpy maths, torch equivalence, factor hooks, whole-model repair and invariance, data, evaluation, statistics, job stages |
| `patches/`, `scripts/setup_lexsi.sh` | Our bug fixes to the four Lexsi libraries, and the script that installs the patched versions |
| `verification/` | The toy-transformer verification: write-up, scripts, results |

`scripts/`, in pipeline order:

| Script | What it does |
| --- | --- |
| `00_model_facts.py` | Records the model's real shapes |
| `01_*.py` | Builds the data: harmful prompts, Hinglish, XSTest, the Hindi task, MMLU, a contamination check |
| `02_evaluate.py`, `02_label_sample.py` | Generates and judges answers per language; samples answers for human labels |
| `03_finetune.py` | LoRA fine-tune through SafeTune |
| `04_factors.py`, `04_leverage.py` | K-FAC factors on harmful and task data; the leverage map (P5) |
| `05_invariance.py` | The V/O gauge test (P1) |
| `06_repair.py` | The repair sweep over μ, and the forecast (P2, P3) |
| `07_baselines.py` | WiSE-FT, RegMean, diagonal Fisher and others, on dev and at matched task loss |
| `08_localisation.py` | Targeted versus random module sets |
| `09_pick_test_settings.py` | Picks μ* and each baseline's setting for the test split by the fixed rule |
| `10_compare_with_baselines.py` | P6: pencil minus baseline drift per language, paired bootstrap, Holm. Its outputs keep the names `p6_run*` |
| `11_wiseft_checkpoint.py` | Builds one WiSE-FT checkpoint for the test split |
| `12_wiseft_drift_curve.py` | Fits how drift shrinks along WiSE-FT, (1 − α)^p, with a bootstrap interval |

`src/pencil/`:

| Module | What it does |
| --- | --- |
| `modules.py` | Module names and structure: which linear layers exist, which share an input factor, layer order |
| `chat.py` | Chat templating with user and response offsets; masks for the four row sets |
| `factors.py` | `KFACCollector`: hooks, sampled-label Fisher, row sets `loss`/`all`/`full`/`full_reduce` |
| `factor_store.py` | `Factors` (in memory, saved per layer chunk), `FactorStore` (reads chunk files on demand) and `LazyFactors` (pools languages, projects out the known null vectors, applies transforms, one layer at a time) |
| `geneig.py` | Generalised eigenproblems on the range of F_h + F_t, covariant or Euclidean damping, output-side damping, noise floors; a fast top-k solver for the MLP-width sides |
| `repair.py` | Layer repair (subtraction form, truncation), forecast, frontier, `ModelRepair` over a whole model, the scaled trust-region repair, rule 11, the task-gradient diagnostic |
| `mu_grid.py` | The μ grid chosen from predicted regret fractions, written before anything is measured |
| `finalists.py` | The fixed rule for μ*, its backup, and matching baselines to the same task loss |
| `delta.py` | ΔW from a PEFT adapter in fp32, dense or low-rank (`LoRADelta`); reading, merging and writing weights; saving checkpoints |
| `leverage.py` | Leverage map, per-head and per-KV-group bounds, refusal-direction cosine, targeted and random module sets |
| `invariance.py` | V/O gauge for GQA on weights, deltas and factors; output KL |
| `baselines.py` | WiSE-FT, RegMean, diagonal Fisher merge, AlignMerge-style merge, Safe LoRA projection, Safe Delta Hessians |
| `safetune_plugin.py` | `PencilTrainer`, registered as SafeTune recover method `pencil` |
| `datasets.py`, `generate.py`, `judge.py`, `evaluate.py`, `stats.py` | Data layout and splits, cached generation, Llama Guard 3 and Aya Expanse judges, per-language evaluation, statistics |
| `config.py`, `runtime.py`, `results.py` | Config and `.env`; shared script plumbing (models, data, factors, the dev probe); one JSON per result |
| `dryrun.py`, `testing.py`, `plots.py` | The offline dry run; tiny Cohere2 model and tokenizer; figure helpers |

## Setup

```bash
git clone https://github.com/tkaunlaky10/safety-pencil && cd safety-pencil
cp .env.example .env            # then add HF_TOKEN
python3.12 -m venv .venv && . .venv/bin/activate && pip install -U pip uv
uv pip install -r requirements-main.lock.txt
bash scripts/setup_lexsi.sh main            # swaps in the patched SafeTune, AuditKIT, CuratorKIT
uv pip install -e . --no-deps               # this package (the scripts import it)
python -m pytest                            # 270 passed, 8 skipped (CPU, under a minute and a half)
bash scripts/dry_run_all.sh                 # the whole pipeline on a tiny offline model, about a minute on CPU
```

Style: `ruff check . && ruff format --check .` (config in `pyproject.toml`: 120 columns; the patched Lexsi clones in
`third_party/` are excluded). Accept the Hugging Face licences for `CohereLabs/tiny-aya-global`,
`meta-llama/Llama-Guard-3-8B` and `CohereLabs/aya-expanse-8b` first.

## Reproducing the runs

The GPU work ran as Hugging Face Jobs. Each stage is a shell script in `jobs/stages/`; `jobs/launch.py` uploads the
code, stamps its git hash, starts the job and prints a cost estimate. Results sync to the runs repo.

```bash
python jobs/launch.py calibrate                      # access checks, then the fine-tunes A, C, D
python jobs/launch.py evaluate theta0                # and evaluate <run>: probe and test tiers
python jobs/launch.py invariance_forecast A          # factors, P1 and P2
python jobs/launch.py repair_scaled A --gpu l40sx1   # bounded repair sweep, baselines, P6, test tier (~9 h)
python jobs/launch.py wiseft_test A_0p5 --gpu l4x1   # WiSE-FT α = 0.5 on test
python jobs/report.py                                # outputs/summary.md from the result JSONs
```

Each script also runs on its own; `--help` shows its options, and `--dry-run` runs it on the tiny model.
`outputs/` is gitignored because it holds checkpoints, factors and raw generations.

## Resources at Tiny Aya scale

- **Factor files:** fp32, about 52 GB on disk per side and row set for all 36 layers.
- **The repair:** reads factors one layer at a time and keeps compact bases (top 512 pairs on MLP-width sides), about
  12 GB of RAM. A μ sweep reuses the bases.
- **The sweep:** two bf16 models on the GPU (θ0 and the working copy, 14 GB). The invariance test holds one model in
  fp32 (13 GB).
- **Cost:** the full bounded-repair job took about 9 h on one L40S (about $14 of HF credits).

## Licences

- Our code: MIT ([`LICENSE`](LICENSE)).
- Tiny Aya derivatives: CC-BY-NC 4.0 plus Cohere Labs' acceptable-use policy.
- The Lexsi libraries: LSAL. See `IMPLEMENTATION.md` §0.
