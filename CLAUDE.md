# Safety Pencil: project memory

Lexsi AI Safety and Alignment Hackathon, Track 2b. Team of three. Submission is due at the end of 14 Nov 2026 AoE; the team's real deadline is **23:59 IST on 14 Nov**, with a complete draft submitted by 23:59 IST on Fri 13 Nov. Experiments freeze at 23:59 IST on Thu 12 Nov. **Scope and dates follow the five-day plan (`03-team/PLAN_5DAY.md` in the team folder):** task T1 only; four runs (A: 4000 rows seed 0; B: 4000 rows seed 1; C: 1000 rows seed 0; D: 1000 rows + 2% contamination seed 0), each with adapters saved at 25/50/75/100%; English and Hindi first, Hinglish from Wed. Cut: T2, preserve (P4), worst-language repair, compression.

## What we are building

Fine-tuning Tiny Aya on a harmless Hindi or Hinglish task weakens its refusals. We measure two curvature matrices of the same model: F_h on harmful prompts and H_t on task data. Their generalised eigenproblem F_h v = λ H_t v gives one basis in which λ is "safety damage per unit of task-loss change". In that basis we:

- **Locate** where safety lives (problem statement #9): a per-layer leverage map.
- **Repair** a drifted model in closed form (#6): shrink each coordinate by μ/(λ+μ).
- **Preserve** safety while fine-tuning (#5): a curvature penalty, equal to repair at second order.
- **Compress** (#11, stretch only): bit allocation by reverse water-filling.

Every measurement runs separately in English, Hindi and Hinglish (the #4 lens).

**How to describe it.** The repair is a Fisher-weighted merge of θ₀ and θ₁ (PROOFS.md Proposition 5.3), in the same family as Fisher merging, RegMean and MaTS. Its objective substantially overlaps AlignMerge (arXiv 2512.16245). The forecast ½ΔᵀF_hΔ is Springer et al.'s Proposition 3.3 (arXiv 2602.15799). Never call any of these new. What we claim, "to our knowledge":
- the generalised spectrum F_h v = λH_t v as a leverage map (locate), tested against random-layer controls;
- exact invariance of the computable repair under the GQA V/O gauge, as a test that separates it from Euclidean safety-subspace methods;
- a per-language (English, Hindi, Hinglish) forecast and repair on Tiny Aya;
- the exact two-term K-FAC inverse used as a safety merge. Cite the technique; it is known.

`verification/VERIFICATION.md` has the checks behind every rule below.

Hackathon rules we must meet:
- Chain at least 2 Lexsi repos; we use 4: CuratorKIT, SafeTune, CircuitKIT, AuditKIT.
- Use a Cohere Labs model.
- Report before/after results on more than one metric.
- Every model, adapter, log and dataset goes in **public, ungated** Hugging Face collections.

## The maths (source of truth; implement exactly this)

Notation: θ₀ is the aligned model, Δ the LoRA update, θ₁ = θ₀ + Δ.

- Safety loss: S(θ) = E_{x∈D_h} KL(p_θ₀(·|x) ‖ p_θ(·|x)). Its Hessian at θ₀ is F_h.
- Task curvature: H_t is the Fisher of the task loss at θ₁, with sampled labels. It is only positive *semi*definite: LayerNorm and the attention gauges give exact null directions shared with F_h. Work on range(F_h + H_t) and damp **covariantly**, as H_t + γF_h (see Conventions 2).
- Pencil: F_h v = λ H_t v, with Vᵀ H_t V = I and Vᵀ F_h V = Λ.
- Coordinates: c = Vᵀ H_t Δ.
- **Forecast** (P2): S(θ₁) ≈ ½ Σ λᵢ cᵢ².
- **Repair**: c_rep = μ/(λ+μ) · c.
- **Predicted frontier**:
  - S(μ) = ½ Σ λ (μ/(λ+μ))² c²
  - R(μ) = ½ Σ (λ/(λ+μ))² c²
  - Both are in nats per token, summed over layers and modules.
  - R(μ) assumes θ₁ is a task minimum, and fine-tunes aren't. So **measure task regret on dev** and plot the measured value. Report the diagnostic |gᵀδ| / (½ δᵀĤδ), where g is the task gradient at θ₁ and Ĥ the K-FAC task curvature; above 0.1, R(μ) is not trustworthy. Never add g to the repair equation (that step blows up).
- Covariant damping leaves the forecast ½ΔᵀF_hΔ unchanged and maps λ → λ/(1+γλ).
- The quadratic model is qualitatively right but quantitatively off at full scale. On the toy, actual drift was 3× the forecast before repair, and 28–885× the prediction after a damped repair. **Use forecasts and the predicted frontier as rankings, never as calibrated numbers.**
- The repair is θ_rep = (F_h + μH_t)⁻¹(F_h θ₀ + μH_t θ₁).
- **Preserve**: minimise L_t + β·S. To second order this equals repair with μ = 1/β.

Kronecker (K-FAC) form, per linear layer W (shape d_out × d_in, PyTorch convention):

- Factors: A = E[a aᵀ] (d_in-sized) and G = E[g gᵀ] (d_out-sized), so F ≈ A ⊗ G.
  - **Normalisation (K-FAC expand):** A is a mean over the rows used; G is a sum over the same rows divided by the number of loss tokens.
  - One backward pass of the sum of independently sampled labels gives an unbiased G.
- **Range restriction:** for each side, take U = eigenvectors of A_h + A_t with eigenvalue > 1e-10 × max, and solve the pencil for (UᵀA_hU, UᵀA_t′U). Do the same for G.
  - **Known null vectors (implemented 9 Nov):** the shared q/k/v/gate/up input factor has the exact null vector 1/w, and the o_proj and down_proj output factors have the all-ones null vector (every path to the loss passes a mean-subtracting LayerNorm). In fp32 factors from a bf16 model they are null only to about 1e-8 of the top eigenvalue, so `factors.project_out` removes them exactly, and the threshold for those factors is 10 × the measured floor (`runtime.prepare_factors`). Every other factor keeps 1e-10 (`rel_tol_other`), so the gauge-touched factors (o_proj input, v_proj output) aren't cut inside the band a gauge can move.
- **Covariant damping:** A_t′ = A_t + γ_c A_h and G_t′ = G_t + γ_c G_h, with γ_c = 1e-3. This keeps exact invariance.
- Solve A_h x = α A_t′ x and G_h y = γ G_t′ y, normalised so that Xᵀ A_t′ X = I and Yᵀ G_t′ Y = I.
- Eigenvalues are products: λ_ij = α_i γ_j.
- Repair, given ΔW:
  - C = Yᵀ G_t′ ΔW A_t′ X
  - M_ji = μ/(α_i γ_j + μ)
  - ΔW′ = ΔW − Y ((1 − M) ∘ C) Xᵀ. **Always use this subtraction form,** with or without truncation: exactly-null components of ΔW pass through untouched.
  - Write back W₀ + ΔW′ as merged weights. ΔW′ has full rank, so it can't be stored as a rank-16 adapter.
- Truncated basis (top-k pairs on an MLP-width side): ΔW′ = ΔW − Y_K ((1 − M_K) ∘ C_K) Xᵀ. Pairs outside the kept set pass through unchanged. The extra safety drift is at most ½ α_max γ_(k+1) ‖C_discarded‖², so truncate only where α_max γ_(k+1) ≪ μ.
- Forecast per layer: ½ Σ α_i γ_j C_ij².
- μ is global across layers.

`test_math.py` (copy it to `tests/`) checks all of the above in numpy (18 tests). `pencil_repair_np` is the reference implementation. The torch code must match it.

## Model and data facts

- θ₀: `CohereLabs/tiny-aya-global`. Base model for ReSta: `CohereLabs/tiny-aya-base`. Free extra: `CohereLabs/tiny-aya-fire`. All three are **gated**: accept the licence on Hugging Face and put HF_TOKEN in `.env`.
- Architecture: `cohere2` with 36 layers, hidden size 2048, 16 query heads, 4 KV heads and head_dim 128. Read `intermediate_size` and `vocab_size` from the config. Prompt 0 checks the rest and records it in `configs/model_facts.json`:
  - Whether the decoder layer is parallel (attention and MLP read the same `input_layernorm` output). If it is, q, k, v, gate and up share **one** A factor per layer.
  - Whether v_proj has a bias or a norm. It must have neither for the invariance demo.
  - How heads are grouped. In HF's `repeat_kv`, query head h reads KV head h // 4.
- The Tiny Aya chat template adds about a 366-token system preamble, so use max_len ≥ 512 for prompts and 256 new tokens.
- On the L4 (24 GB), Tiny Aya in bf16 takes about 7 GB. Llama Guard 3-8B in bf16 takes about 16 GB, so don't co-host it with training.

Data:
- **Harmful prompts:** `CohereForAI/aya_redteaming` (it may redirect to `CohereLabs/...`), splits `english` and `hindi`, Apache-2.0. Fixed-seed split per language: 40% factors, 10% dev, 50% test.
- **Hinglish:** romanised, code-mixed versions of the English and Hindi prompts, made separately within each split.
- **Over-refusal:** benign but alarming prompts. Use XSTest's safe prompts (find the official Hugging Face copy and verify it), translated into Hindi and Hinglish.
- **Task T1:** Hindi rows of `CohereForAI/aya_dataset`.
- **Task T2 [cut on 9 Nov; stretch only]:** public-service QA in Hindi and Hinglish, built with CuratorKIT from public government-scheme or health-guidance PDFs.
- **Contamination variant:** 2% harmful rows from SafeTune's `harden.load_harden_data` (BeaverTails). **Never generate new harmful completions.**
- **General-capability check:** `CohereLabs/Global-MMLU`, subsets `en` and `hi`, 200 questions each, scored by log-likelihood.

Judges:
- Harmful compliance: `meta-llama/Llama-Guard-3-8B` (gated). Its model card lists Hindi support.
- Refusal and over-refusal: `CohereLabs/aya-expanse-8b`, given a fixed rubric.
- Validate both against 150 human labels per language and report Cohen's κ. Never use keyword matching.

## Repo layout

```
CLAUDE.md  PREREGISTRATION.md  README.md  pyproject.toml  .env.example
configs/        experiment.yaml (ids, langs, runs, row sets, pencil settings, mu grid), model_facts.json,
                refusal_rubric.txt, quality_rubric.txt
src/pencil/
  config.py runtime.py   yaml + env; script plumbing and the offline dry run
  chat.py                chat templating, user/response offsets, row-set masks
  datasets.py            processed data layout, splits, contamination mixing, overlap checks
  generate.py            batched, cached generation (HF in memory, AuditKIT for paths)
  factors.py             KFACCollector: hooks, sampled-label Fisher, row sets, save/load, pooling, known nulls
  geneig.py              gen_eig in float64 (range restriction, damping, top-k), noise floors
  repair.py              layer repair, forecast, frontier, ModelRepair, trust region, task-gradient diagnostic
  delta.py               LoRA ΔW in fp32, merging
  leverage.py            leverage map, per-head / per-KV-group bounds, refusal-direction cosine
  invariance.py          V/O gauge per KV group, delta and factor transforms, output KL
  baselines.py           RegMean, diagonal Fisher, AlignMerge-style, Safe LoRA projection, Safe Delta Hessians
  safetune_plugin.py     PencilTrainer registered as SafeTune recover method "pencil"
  judge.py evaluate.py   Llama Guard 3 + Aya Expanse refusal judge; per-language metrics, probe vs full
  stats.py               Wilson, exact McNemar, bootstrap Spearman, Holm, Cohen's kappa
  testing.py             tiny random Cohere2 model + character tokenizer with a Tiny-Aya-style preamble
scripts/        00_model_facts, 01_* (data incl. 01_build_mmlu), 02_evaluate, 02_label_sample, 03_finetune, 04_factors,
                04_leverage, 05_invariance, 06_repair (sweep, forecast), 07_baselines, 08_localisation,
                dry_run_all.sh, setup_lexsi.sh
tests/          test_math.py, test_torch_equivalence.py, test_factors.py, test_model_repair.py,
                test_datasets.py, test_evaluate.py, test_stats.py, test_dry_run.py (opt-in)
verification/   toy-transformer checks copied from the team folder (VERIFICATION.md, scripts, results)
outputs/        (gitignored) ckpt/, factors/, generations/, results/*.json, figures/, dryrun/
```

Not implemented: `preserve.py` (P-quad, P-exact; cut on 9 Nov) and the CircuitKIT script (Prompt 8 step 5).

Scale rules (from the 9 Nov code review):
- Factors are saved in fp32. Read them with `runtime.load_factors` / `harm_factors` / `task_factors`, which stream them one layer at a time (`FactorStore`, `LazyFactors`). Never load all 36 layers at once.
- `ModelRepair` keeps only compact bases (eigenvalues, X and F_t′X), so build one per damping setting and reuse it across μ. `trust_region_repair(..., repairers=...)` does that.
- Load adapters with `lora_delta(path, lowrank=True)`. A dense ΔW for the whole model is about 11 GB.
- `chat.encode_chat` truncates the end of the user's text and keeps the template's turn markers.

## Conventions (non-negotiable)

1. **Precision.** Do eigen and repair maths in float64 and keep weights in bf16. Take ΔW from the LoRA adapter in fp32, as scaling · lora_B @ lora_A. Never subtract bf16 merged weights.
2. **The Fisher and damping.**
   - **Labels.** Use the true Fisher with sampled labels: at each response position, sample from the model's own softmax and back-propagate the sum of the sampled tokens' log-probs in one pass.
   - **Rows.** One pass fills up to four row sets: `loss` (response positions), `all` (user prompt + response, no preamble), `full` (every position, preamble included) and `full_reduce` (K-FAC-reduce over the full rows: one mean-input, summed-gradient pair per sequence). Choose on dev. On the tiny Cohere2 test model, leaving the preamble out under-forecast v_proj drift by 60–1000×, because k/v at preamble positions feed every later token. The config defaults are `full` for the repair and `{default: full, v_proj: full_reduce}` for the forecast. Reduce factors have rank at most the number of sequences and made the repair worse there, so use them only for forecasts.
   - **Normalisation.** Use the expand normalisation above.
   - **Damping.** Damp covariantly (γ_c = 1e-3) on the range of A_h + A_t (and of G_h + G_t). Euclidean damping (γ_e × mean diagonal × I) is only the trust-region fallback in rule 11, because it breaks exact invariance.
3. **Factor passes.** Set requires_grad False on every parameter and call `inputs_embeds.requires_grad_(True)`, so no weight gradients are stored. No gradient checkpointing. Batch size 1–2. If the MLP-width factors don't fit, hook the layers in chunks across several passes.
4. **Splits.** Factors come only from the factor split, μ and β are chosen only on dev, and the test split is touched only for final numbers.
5. **Evaluation tiers.** Probe evaluation (100 dev prompts per language) is for sweeps. Full test evaluation is only for checkpoints that make the write-up.
6. **Results.** Every result is one JSON in `outputs/results/` with config, seed, git hash, checkpoint id and timestamp.
7. **Safety and licences.** Never commit or push raw harmful completions; store judge labels and hashes only. Label drifted checkpoints "research artifact: unrepaired, do not deploy", because SafeTune's licence bars deploying unrepaired drifted checkpoints.
8. **Secrets.** Keep HF_TOKEN and COHERE_API_KEY or OPENAI_API_KEY in `.env` only.
9. **Tests first.** `pytest -q` must pass before any GPU job.
10. **Pre-registration.** `PREREGISTRATION.md` (P1–P7 with pass criteria, below) is committed before the first experiment.
11. **Trust-region check (mandatory).** Before evaluating a repaired checkpoint, check two things on dev:
    - the actual S fell;
    - every module's step ratio ‖ΔW′ − ΔW‖_F / ‖ΔW‖_F is at most 1.

    If either fails, add Euclidean damping, with γ_e chosen from {1e-3, 1e-2, 1e-1} on dev, and record the invariance error at that γ_e. On the toy, the undamped exact repair took steps 2–6× the size of the fine-tune and made both safety and the task worse.

    `repair.trust_region_repair` implements this, and `scripts/06_repair.py sweep` logs every attempt. On the tiny test model the covariant repair cut the measured S to 15% while some o_proj step ratios were above 20, along directions the inputs barely visit. So the sweep also logs an input-weighted ratio ‖δA_t^½‖/‖ΔW A_t^½‖. Whether to judge the step by that ratio instead is an open team decision.

## Environment

- Python 3.12, because SafeTune requires ≥ 3.12.
- **Main venv:**
  - torch (a CUDA build matching the driver), **transformers==5.21.1**, peft, accelerate, datasets;
  - `safetune`, `auditkit[transformers,interop]`, `curatorkit[hf,generation]`;
  - scipy, numpy, pandas, matplotlib, huggingface_hub, python-dotenv and pytest.

  The transformers pin satisfies AuditKIT (`>=5.15,<5.22`) and SafeTune (`>=5.15,<6`). Install everything with `uv pip install -r requirements-main.lock.txt` (and `requirements-circuit.lock.txt` for `.venv-circuit`), which pins the Lexsi repos at the commits in `IMPLEMENTATION.md`. Confirm that cohere2 loads. Don't fork the Lexsi repos.
- **CircuitKIT** goes in a separate `.venv-circuit`, because it pins `transformer-lens==3.8.0`. Import `circuitkit` before `transformer_lens`.
- **On Colab,** run `pip uninstall -y torchao` before importing SafeTune.
- GPU: one L4 (24 GB). Kaggle or Colab only as overflow.

## Repo facts (verified in the source on 9 Nov; details and file references in `IMPLEMENTATION.md`)

**We run patched versions of all four libraries.** `scripts/setup_lexsi.sh` installs them, and `patches/README.md` lists every fix. Never install the Lexsi packages any other way, or the fixes below are lost.

- **No Lexsi repo has Fisher, Hessian or K-FAC code.** Factors, pencil maths and repair are ours.
- **Tiny Aya architecture** (transformers `modeling_cohere2.py`):
  - parallel block, so q, k, v, gate and up share one A factor per layer;
  - the LayerNorm has no bias, so that A has the exact null vector 1/w;
  - GQA 16/4, with query head h reading KV head h // 4;
  - RoPE only on sliding layers;
  - logits scaled and embeddings tied, so keep `lm_head` out of the repair.
- **The chat template always injects a ~366-token preamble that starts "You are in contextual safety mode".** Use the identical template, and pinned tokenizer files, for fine-tuning, factor passes, discovery and evaluation.
- **SafeTune:**
  - Recover methods are `XTrainer(merged_drifted, ...).apply()`, matched by state_dict key. Add ours with `register_recover("pencil", "PencilTrainer")` and `setattr(safetune.runner.recover, "PencilTrainer", ...)`.
  - `PlainSFTTrainer` discards the LoRA adapter, so the drift fine-tunes use our own PEFT loop with `lora_wrap` and `tokenize_dataset`.
  - Contamination rows come from `harden_contamination_pairs(n)[0]`.
  - ReSta: α ≈ 0.25 on Tiny Aya.
  - `PrePostMergeTrainer` needs `pre_model="CohereLabs/tiny-aya-base"`, or it equals WiSE-FT.
  - Safe Delta takes `hessian_inputs`; give it our A factors.
  - RefusalDirection: `RefusalDirectionTrainer(model, tok, config=RefusalDirectionConfig(pick_layer=24, target_layers=[24]))` (config passthrough added by our patch).
  - SafeTune's built-in evaluation is English-only.
- **CircuitKIT:**
  - Supports `CohereLabs/tiny-aya-global`; parity was validated only on tiny-aya-base, so run the parity test on global first.
  - Write our own refusal TaskSpec (`ck.register_task`), with token-length-matched harmful/harmless pairs at batch size 1.
  - Call `get_scores_eap_ig` directly with `use_split_qkv_input=False`; `discover()` likely runs out of memory at 512 tokens.
  - MLP node scores read the gradient at the MLP input (`attribute_node.py:170`). **This is fixed in our patch, so only use the patched install; MLP scores from an unpatched CircuitKIT are wrong.**
  - There is no random-layer control; build one.
- **AuditKIT:**
  - Generate, unload, run Llama Guard 3 (`ak.GuardJudge("llama_guard")`), unload, run the Aya Expanse `ak.LLMJudge`: one model on the GPU at a time.
  - Raw completions are always cached under `$XDG_CACHE_HOME/auditkit/runs`, so point it at a temporary folder and wipe it.
  - Never use `push_to_hub` on results (it uploads raw outputs).
  - **Always set `AUDITKIT_DISABLE_RUN_CACHE=1`** (our patch). Unpatched, it caches raw completions with no off switch.
  - The cache key now includes local weight files' sizes and modification times (our patch), but still use a new directory per checkpoint.
  - No Wilson, McNemar, Holm or κ intervals: write them in `stats.py`.
- **CuratorKIT:**
  - Set `min_tokens=1`, and add deterministic ids with `preprocessing_fn`.
  - Toxicity filtering is off by default.
  - There is no language ID or translation: the Hinglish step is a custom `BaseGenerationTask` run through `Pipeline`.
  - Prompt-only exports drop metadata, so write a `prompts_meta` exporter.
  - Write the dataset cards ourselves.
- **Licences (all LSAL):**
  - §1A: an organisation must send an acknowledgement to support@lexsi.ai.
  - SafeTune §4: never deploy drifted checkpoints.
  - SafeTune §6: any pull request to Lexsi gives them commercial rights, so publish the pencil as a plugin in our own repo.

## Predictions and pass criteria (copy into PREREGISTRATION.md)

- **P1 — Invariance.** Under the V/O reparameterisation, logits match, and pencil repair gives the same function.
  - ReSta is linear in the weights, so it must also be invariant: this is the positive control.
  - At least one of Safe LoRA, LoX or the GuardSpace projection must change by more than seed noise.
- **P2 — Forecast.** Spearman ρ ≥ 0.6, with its bootstrap CI above 0, between the forecast ½ΔᵀF_hΔ and the measured refusal drop, over the 48 checkpoint–language pairs (16 checkpoints × 3 languages). Only harmful factors are needed.
- **P3 — Frontier.** The measured refusal and task score move with μ in the predicted order (ρ ≥ 0.8).
- **P4 — Preserve vs repair [cut on 9 Nov; report as not run].** Report output KL between P-quad, P-exact and repair at μ = 1/β, as a function of data size.
- **P5 — Localisation.** Leverage concentrates in late output-side MLP blocks. |cos(y₁, r̂)| > 0.1 in the top-leverage layers. Rank correlation with CircuitKIT EAP-IG scores is positive.
- **P6 — Repair vs baselines.** The pencil frontier dominates at least one weight-space baseline on worst-language refusal at equal task score. The baselines include:
  - the two closest relatives, diagonal Fisher merging (Matena & Raffel) and RegMean;
  - a standard Fisher merge that weights θ₀ by its Fisher on general instruction data, which tests whether our safety weight matters.

  On the toy, the pencil was highest at 80–96% refusal and RegMean at 97% and above, with gaps within noise. Report the ≥97%-refusal end separately, using a paired bootstrap over prompts. Report honestly if retraining on 50 refusals, or any merge, wins.
- **P7 — Shared subspace.** Principal angles between per-language pencil subspaces, measured in the H_t inner product. Report either way.

Statistics: Wilson 95% intervals, exact McNemar on paired prompts, bootstrap intervals, and Holm correction across the three languages.

Gates:
- **Gate 1 (Tue 10 Nov, 23:00), on run A.** If no language drops by 10 points or more with McNemar p < 0.05, make the contaminated variant the main setting.
- **Gate 2 (Wed 11 Nov, 23:00).** If the repair does not beat diagonal Fisher merging and RegMean, lead the write-up with invariance, the forecast, localisation and the multilingual results, and call the repair "competitive with the best merging baselines".
