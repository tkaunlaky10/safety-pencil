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
