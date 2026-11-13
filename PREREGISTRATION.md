# Pre-registration: Safety Pencil

Commit this file to the code repo **before the first experiment runs**, so the commit timestamp shows the hypotheses and pass criteria came first.

- Written: 6 Nov 2026
- Pre-registration commit: `__________`, filled in at commit time
- Any later change to this file goes in the deviation log at the bottom, with its date and reason. Never edit the criteria in place.

## Question

Does the generalised eigenbasis of the harmful-prompt Fisher and the task Fisher (the "pencil") predict, localise and repair safety drift from benign fine-tuning? And does it do so equally well in English, Hindi and Hinglish?

## Fixed design

**Model.** θ₀ = `CohereLabs/tiny-aya-global`. ReSta needs a base model: `CohereLabs/tiny-aya-base`.

**Harmful prompts.** Aya Red-teaming, English and Hindi, plus a Hinglish version of each.
- Split per language with seed 0, stratified by harm category: 40% factor, 10% dev, 50% test.
- Hinglish prompts are made within each split, so test prompts never leak into factors.

**Over-refusal.** XSTest safe prompts in English, Hindi and Hinglish, split dev 20% / test 80%.

**Changes made on 9 Nov, before any experiment ran.** The schedule was cut to five days (`03-team/PLAN_5DAY.md`):
- Task T2 dropped.
- The 12-run grid replaced by 4 runs, with adapters saved at 25/50/75/100% of training.
- P2 computed over 16 checkpoints × 3 languages.
- P4 not run.
- P7 run only if time allows.
- Worst-language repair dropped.
- Human labels cut to 100 per language.
- Baselines: GuardSpace and SafeGrad dropped; RegMean, diagonal Fisher merging, an AlignMerge-style merge, Safe Delta with our curvature, and a standard Fisher merge added.
- P5 gains a random-layer control.

The text below keeps the original wording; read it with these changes.

**Tasks.**
- T1: Hindi instructions from the Aya Dataset.
- T2: public-service QA in Hindi and Hinglish, built with CuratorKIT.
- Each task at two train sizes, 1000 and 4000, with a held-out 300.

**Fine-tunes.**
- LoRA rank 16 on all seven linear modules.
- 2 tasks × 2 sizes × 3 seeds = 12 runs.
- One contaminated variant: T1, size 1000, seed 0, with 2% harmful rows from SafeTune's BeaverTails loader.

**Factors.**
- About 256 prompts per set.
- True Fisher with sampled labels, response tokens only, normalised per token.
- Damping γ = 1e-3 × mean diagonal, on the task side only.

**Judges.**
- Llama Guard 3-8B for harmful compliance.
- Aya Expanse 8B with a fixed rubric for refusal and over-refusal.
- Both validated against 150 human labels per language, reported as Cohen's κ.

**Selection rules.** μ, β, and every baseline's hyperparameters are chosen on the dev split only. The test split is touched once per final checkpoint.

## Metrics

**Primary.** Harmful-prompt refusal rate per language, and the worst-language refusal (the minimum over the three languages).

**Secondary:**
- harmful-compliance rate;
- over-refusal rate per language;
- task score: per-token NLL on held-out answers and win rate against θ₀ for T1, judged answer accuracy for T2;
- Global-MMLU, English and Hindi;
- output KL against θ₁ on benign prompts;
- wall-clock cost.

## Hypotheses and pass criteria

| ID | Hypothesis | Test | Pass criterion |
| --- | --- | --- | --- |
| P1 | Pencil repair is invariant to the V/O reparameterisation; Euclidean methods are not | Random M_g with condition numbers 10 and 30. Compare logits; rerun the pencil, ReSta, Safe LoRA, LoX and GuardSpace projection in both coordinate systems | Logits match to atol 1e-3 (fp32). Pencil and ReSta: output KL between the two coordinate systems ≈ 0 (ReSta is the positive control). At least one of Safe LoRA, LoX or GuardSpace: refusal or task score differs by more than the seed spread |
| P2 | The quadratic forecast ½ΔᵀF_hΔ ranks measured drift, LoRA included | All 36 checkpoint–language pairs plus the contaminated variant; Spearman ρ with a 2000-resample bootstrap CI. Also the task-normalised score, and the T1-vs-T2 ranking | ρ ≥ 0.6 and CI lower bound > 0 |
| P3 | The predicted frontier S(μ), R(μ) matches the measured one in order | Six values of μ, one checkpoint per task, per language | Spearman ρ ≥ 0.8 between predicted and measured, for both refusal and task score. Order only: R's magnitude is not compared, because θ₁ is not a task minimum (`01-plan/verification/VERIFICATION.md` §3.4) |
| P4 [not run: cut on 9 Nov] | Preserve ≈ fine-tune-then-repair at μ = 1/β | Output KL between P-quad, P-exact and repair, on harmful and task prompts, at both sizes | Report only; agreement within seed spread counts as support |
| P5 | Leverage concentrates in late, output-side MLP blocks and aligns with the refusal direction | Leverage heatmap; \|cos(y₁, r̂)\| against 1000 random unit vectors; rank correlation with CircuitKIT EAP-IG scores | \|cos\| > 0.1 in the top-leverage layers; correlation > 0 with CI lower bound > 0 |
| P6 | Pencil repair matches or beats the weight-space baselines on worst-language refusal at equal task score | Frontier comparison per language against ReSta, LoX, C-ΔΘ, Safe LoRA, GuardSpace projection, retraining on 50 refusals, and the two closest relatives of our repair: diagonal Fisher merging and RegMean | The pencil frontier dominates at least one baseline's |
| P7 | The per-language pencils share their top subspace | Principal angles between per-language top-32 subspaces, measured in the H_t inner product, for the top-leverage modules | Report only |

## Statistics

- Wilson 95% intervals on every rate.
- Exact McNemar tests on paired prompts for method comparisons.
- Bootstrap intervals (2000 resamples) for ρ and for worst-language gaps.
- Holm correction across the three languages.
- Seed spread is reported as mean ± SD over 3 seeds.

## Decision gates

- **Gate 1, Tue 10 Nov 23:00 (was 8 Nov).** If no language shows a refusal drop of 10 points or more with McNemar p < 0.05 on any benign fine-tune, the contaminated variant becomes the main setting, with seeds 1 and 2 added.
- **Gate 2, Wed 11 Nov 23:00 (was 10 Nov).** If P6 fails on dev, the write-up leads with P1, P2 and P5, and the repair is reported as one result among several.
- **Forecast fallback (stretch).** If P2 gives ρ < 0.4, rerun it with EK-FAC and report both.

## What we will report regardless of outcome

Every prediction above, including failures, with its numbers and intervals. Negative results on P2 or P6 are findings: a P2 failure under LoRA would extend Springer et al.'s result.

## Deviation log

| Date | Change | Reason |
| --- | --- | --- |
| 11 Nov 2026 | T1 data: the Aya Dataset's Hindi rows are used first (all 300 held-out rows come from them), and the train files are topped up to the run sizes with a seeded sample of the Aya Collection's Hindi split (`CohereLabs/aya_collection_language_split`, Apache-2.0, mostly translated and templated tasks), cleaned the same way. | The Aya Dataset has 1153 Hindi rows and 889 survive cleaning, short of the 4300 the runs need (300 held out + 4000 for run A). Found by the first calibration job, before any fine-tune or factor pass. |
| 11 Nov 2026 | Row sets: no row-set selection pass on dev. The repair uses `full` (the config default), and the forecast uses `full` instead of `{default: full, v_proj: full_reduce}`; factors are collected with the `full` row set only. | No time for the selection pass before the experiments freeze, and the job disk (400 GB on an L4) holds one row set per side (~55 GB each for three harmful languages and the task), not two. |
| 11 Nov 2026 | P6: the standard Fisher merge (`fisher_standard`) is not run. | It needs general-instruction factors at θ0, a fourth factor pass we have neither the time nor the disk for. |
| 11 Nov 2026 | Hinglish prompts: the harmful set keeps only conversions that pass the checks (68–71% per split); ~15% of prompts were rejected because the converting model (Aya Expanse 32B) refused them. Hinglish results are reported on this faithfully converted subset. | Converting with an LLM that refuses some harmful prompts; the alternative (hand conversion) is out of scope. The subset may lean towards milder prompts. |
| 12 Nov 2026 | Gate 1 outcome: refusal and over-refusal *increased* after the benign fine-tune, in every run and language (dev probe, final adapters vs θ0, paired McNemar: e.g. English refusal 48% → 66% in A, 68% in C, 63% in D). No run passes Gate 1. The pre-registered rule would make run D the main setting; see the next entry for the choice actually made. | The pre-registered Gate 1 rule. The fine-tune shifted the model towards refusing, not away from it. |
| 12 Nov 2026 | Primary safety metric: harmful compliance (Llama Guard 3 "unsafe" rate) together with the measured drift S on dev. Refusal stays reported. | The drift showed up as harmful compliance, not as lost refusal. At the final adapters (dev, paired McNemar, Holm across languages): A English 10% → 24% (p 0.007, Holm 0.022), the only change that survives Holm; D Hinglish 20% → 31% (p 0.052, Holm 0.16), hi 7% → 13% and en 10% → 12% (n.s.). |
| 12 Nov 2026 | Main run: A, not D, against the letter of the Gate 1 rule. D stays in the Gate 1 table and in the forecast (P2). | Gate 1 assumed drift lowers refusal; we observed more refusal, more over-refusal and more harmful compliance, and the only change that survives Holm is A's English unsafe rate (10% → 24%, Holm p 0.022). Repairing a run with no significant drift (D) would leave nothing to measure. |
| 12 Nov 2026 | P2 is reported as pre-registered (Spearman of ½ΔᵀF_hΔ against the refusal drop, which cannot pass while refusal rises), and also against the measured drift S on dev per checkpoint and language (`06_repair.py drift`; ½ΔᵀF_hΔ forecasts S itself, drift in either direction) and against the change in harmful-compliance rate. Same criterion (ρ ≥ 0.6, bootstrap CI > 0) for all three; `jobs/report.py` computes them. | The pre-registered pairing assumed drift lowers refusal; here it does not. |
| 12 Nov 2026 | Test-tier finalists, by a rule fixed in code (`src/pencil/finalists.py`) before any finalist was evaluated, on dev data only: μ* = the smallest μ of the grid whose dev task NLL ≤ NLL(θ1) + 0.1 · (NLL(θ0) − NLL(θ1)) (keeps at least 90% of the fine-tune's task gain; μ whose trust-region check failed are excluded); for RegMean, diagonal Fisher and WiSE-FT, the setting whose dev task NLL is closest to NLL(μ*). Test-tier evaluations: θ0, run A's final adapter, pencil μ*, RegMean, diagonal Fisher. WiSE-FT's pick is recorded but not test-evaluated. | Compare methods at equal task score without looking at test data. WiSE-FT's test evaluation was dropped to stay within the Hugging Face credits (about $13 left of $20 on 12 Nov). |
| 12 Nov 2026 | Finalists evaluated at probe tier (100 dev prompts per language), compared with probe_theta0 and probe_runA_frac100; test-tier evaluations of pencil μ* (if ≥ $9 of credits remain) and RegMean's finalist (if ≥ $11) only at the end of the job, and θ0 and run A's final adapter already have test-tier results. The μ sweep and localisation run in the compare job, ordered factors → sweep → baselines → finalists → other μ (P3) → localisation. | The first compare job and the drift job ran out of GPU memory in the dev probe (fixed); about $8 of credits and 9.7 h remained, not enough for three test-tier evaluations (~70 min each) after a rerun. Ordering keeps P6 ahead of localisation if time or money runs out. |
| 12 Nov 2026 | Rule 11 (trust region), **chosen after seeing data**: covariant damping only; a repair passes when the measured S on dev falls and the input-weighted step ratio ‖δ A_t^½‖ / ‖ΔW A_t^½‖ ≤ 1 in every module. The Euclidean step ratio is still logged (worst and median module), and every μ's measurements are kept whether or not it passes. If no μ passes, the repair comparison (baselines, finalists, other μ, localisation) is not run. This resolves CLAUDE.md's open trust-region decision. | The covariant repair was rejected under the written rule at the first μ on Tiny Aya (the log did not record whether S or the Euclidean ratio failed). Covariant-only keeps exact invariance (P1); the Euclidean fallback was dropped because its second set of bases overflowed 62 GB of RAM (compare A, L40S, OOMKilled). On the tiny test model the covariant repair cut measured S to 15% while some o_proj Euclidean ratios exceeded 20, along directions the task inputs barely visit. |
| 12 Nov 2026 | P6: only the must-run baselines (WiSE-FT, ReSta, Safe LoRA, SafeMERGE, RegMean, diagonal Fisher) are run; the nice set (AlignMerge-style, Safe Delta, task arithmetic, LoX) is reported as not run (time). | The first pencil job failed after its factor passes (fp32 factor noise made a down_proj task factor indefinite; fixed by retrying the Cholesky with a larger jitter), and the rerun must finish before the experiments freeze. |
| 13 Nov 2026 | **Post-freeze run (after the 12 Nov 23:59 IST experiment freeze), chosen after seeing the covariant repair fail.** The scaled repair (`ModelRepair(output_nu_rel=1e-2, scale_steps=True)`; numpy reference `tests/test_math.py::pencil_repair_scaled_np`; derivation in the "Pencil Repair Stability" explainer): input sides keep covariant damping A_t′ = A_t + 1e-3 A_h; every output side except v_proj uses G″ = G_t′ + νI on its range, ν = 1e-2 · mean(diag G_t′) (γ_e = 1e-2, the middle of rule 11's pre-registered grid; one value, no selection); then each module's step is scaled by t = min(1, 1/r), r = ‖δ A_t^½‖_F / ‖ΔW A_t^½‖_F. v_proj gets neither (the V/O gauge transforms its output factor). One basis solve, reused across μ. The 12 Nov covariant sweep (`repair_sweep_compare_runA`: S ≈ 31, task NLL ≈ 31, every μ rejected) stays in the results and the summary. | The covariant repair took steps up to 94× the fine-tune (layer 35 up/gate/down; layer 35 gate and up output factors break the 1/γ_c eigenvalue ceiling, so their task factors are numerically indefinite). Rule 11's trust region, made part of the solution, bounds every step by construction; ν conditions the solve. It can only shrink steps, so it cannot create an advantage. |
| 13 Nov 2026 | Rule 11 for the scaled repair: a μ passes when the measured S on dev falls and every module's trust ratio is ≤ 1: the input-weighted ratio with A_t after scaling for scaled modules (≤ 1 by construction), and for v_proj the curvature-norm ratio ‖δ‖/‖ΔW‖ in tr(G_t′ · A_t′ ·ᵀ), gauge-invariant and ≤ 1 by the repair's contraction property. The pre-registered Euclidean ratio ‖δ‖_F/‖ΔW‖_F is logged for every module and μ (before and after scaling), with the A_t′ input-weighted ratio and each t_k, and is not used to accept or reject. **Correction to the 12 Nov entry:** its "input-weighted ratio ‖δ A_t^½‖ / ‖ΔW A_t^½‖" was computed with A_t′ = A_t + γ_c A_h, not A_t. For input directions seen only by harmful prompts (α near 1/γ_c) the two differ by order 1, not 1e-3. | The explainer's rule 11 counts what changes on task inputs, E‖δa‖² = tr(δ A_t δᵀ), so the scaled repair uses A_t; the A_t′ version is kept so the 12 Nov numbers stay comparable. |
| 13 Nov 2026 | The scaled repair's μ grid is fixed from predictions before any measurement: the μ where the predicted task regret with scaling, Σ_k t_k(μ)² R_k(μ), falls to {0.02, 0.05, 0.1, 0.2, 0.4} of its value at μ = 0 (0.01 added before launch: see the last 13 Nov entry) (`pencil/mu_grid.py`: first crossing on a one-per-decade grid, then 8 bisection steps in log μ). The grid and the coarse predicted curve are written to `repair_sweep_scaled_runA.json` before the first μ is measured. μ* is the 12 Nov rule (smallest accepted μ with dev task NLL ≤ NLL(θ1) + 0.1 · (NLL(θ0) − NLL(θ1)) ≈ 1.821). | The covariant μ grid (0.003…1) sat entirely inside the λ plateau; the predicted regret spaces the grid by task cost instead, without measuring anything. |
| 13 Nov 2026 | Baselines for the scaled comparison, on dev: WiSE-FT, RegMean and diagonal Fisher only. ReSta, Safe LoRA and SafeMERGE are not rerun (they never ran on dev; 12 Nov compare ended after the sweep). WiSE-FT's grid becomes α ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75} (α pulls toward θ0); computed natively since the last 13 Nov entry; the old {0.25, 0.5, 0.75} cannot reach the pencil's task NLL range (α ≈ 0.3 is needed). RegMean and diagonal Fisher keep their grids. Then each is matched to NLL(μ*): regula falsi on its knob inside the bracketing pair of measured settings (log μ for RegMean and diagonal Fisher, linear α for WiSE-FT), at most 3 extra settings, stopping within 0.005 nats per token, looking only at task NLL (`pencil/finalists.py`). | P6 compares at equal task score; a grid point can sit far from NLL(μ*). Matching uses NLL only, so it cannot favour the pencil. |
| 13 Nov 2026 | P6 statistic: per language, S(pencil μ*) − S(baseline) on dev with a paired bootstrap over prompts (2000 resamples; both models are scored on the same θ0 answers, so each prompt's summed KL pairs up and S = Σ kl_sum / Σ tokens), 95% percentile interval, Holm across the three languages within each baseline (`scripts/10_p6.py`). "Pencil lower" needs Holm p < 0.05 and the interval below 0. Test tier once: pencil μ*, then RegMean and WiSE-FT at matched NLL, in that order (WiSE-FT last so a timeout costs only it). Diagonal Fisher is compared on dev only. | The pre-registered P6 asks for a paired bootstrap over prompts at equal task score; the measured S is token-weighted, so the bootstrap resamples prompts and recomputes the token-weighted mean. |
| 13 Nov 2026 | Diagnostic, reported and never used to choose anything: every factor side whose largest covariant eigenvalue exceeds (1/γ_c)(1 + 1e-3), which is impossible if its task factor is positive semidefinite. For ν-damped output sides only the ν-damped pencil is solved, so the reported value is a lower bound: the largest covariant Rayleigh quotient γ″_j / (Yᵀ G_t′ Y)_jj along its solved directions. | A mathematical bound, not a fitted rule (explainer Step 3). |
| 13 Nov 2026 (before the scaled run was launched) | (a) μ-grid fractions become {0.01, 0.02, 0.05, 0.1, 0.2, 0.4}: 0.01 is added. (b) The task-gradient diagnostic \|gᵀδ\| / (½ δᵀĤδ) is logged at every μ of the scaled sweep (g at θ1 on 64 task dev rows, computed once; gᵀδ summed module by module). (c) Backup μ*: if no accepted μ has dev task NLL ≤ NLL(θ1) + 0.1 · (NLL(θ0) − NLL(θ1)) (≈ 1.821), μ* is the accepted μ with the lowest dev task NLL; the baselines are matched to that μ's NLL, P6 runs as normal, and the results (`finalists_scaled_runA`, `p6_runA`: `backup: true` / `mu_star_backup: true`) and summary.md label it "backup μ*". If no μ is accepted at all there is no μ* and the stage ends after the sweep and the invariance recheck. (d) Diagnostic only, never used to accept a μ: the harmful-input ratio ‖δ A_h^½‖ / ‖ΔW A_h^½‖ per module and μ. (e) WiSE-FT is computed natively as W0 + (1 − α)ΔW through `delta.merge_deltas` (ΔW from the LoRA adapter in fp32), same α grid, instead of SafeTune's `WiseFTTrainer` (which interpolates bf16 merged weights). Each baseline that raises is recorded as `{"unavailable": "<error>"}` and the rest carry on. | (a) Predicted R leaves out the gᵀδ term, which dominates for small steps, so the smallest predicted costs may understate the measured regret; 0.01 gives the grid a point with a smaller step. (b) Shows where predicted R can be trusted (above 0.1 it cannot; CLAUDE.md). (c) Fixed before any scaled measurement so the finalist rule cannot be chosen after seeing it. (d) The 12 Nov L35 down_proj step was 63× in Euclidean norm but tiny on task inputs; if S fails to fall, this shows whether harmful-only input directions caused it. (e) The pre-registered rule takes ΔW from the adapter in fp32 and never subtracts bf16 merged weights (CLAUDE.md conventions 1). |
