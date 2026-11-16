# With more credits: a mathematical plan for Safety Pencil v2

The first round left five measured failures (§1). This plan answers each with an experiment that has a stated mathematical reason to work, a prediction fixed before running, and a way to be wrong. Section numbers like §5.4 refer to [`WRITEUP.md`](../WRITEUP.md). CPU pilots on a controlled toy model (§1.5; code in [`pilots/`](../pilots/)) have already tested the central claims, and the plan was revised wherever they disagreed with it.

**The organising idea: a confidence blind spot.** A token's KL curvature is the variance of its logit change under the model's own distribution. For a refusal held with probability p, that variance scales as p(1 − p) ≈ e^(−m), where m is the logit margin. So an aligned model's Fisher is nearly blind to its most confident refusals. Every second-order safety tool inherits this: curvature forecasts, Fisher-weighted merges, EWC-style penalties and curvature repairs, ours included. Three consequences drive the plan:

1. **Forecasts.** For a token whose margin m₀ the update reverses, the exact KL is sinh(m₀)/m₀ times the second-order forecast (§2). In the pilot the forecast was 1,200–12,000× too low.
2. **Repair.** Fisher-weighted merging cannot see what it should protect (it was blind in the pilot), so optimise the exact objective instead. In the pilot, learned per-module weights started from the best WiSE-FT point cut harmful compliance to 0.2–8.6%, where interpolation left 25–99% (§3).
3. **Safe fine-tuning.** EWC is the second-order Taylor expansion of the KL penalty, so it inherits the blind spot. In the pilot it failed at every λ, while an exact KL or a refusal-margin penalty held harmful compliance at 0% (§10).

A fourth idea fixes the curvature estimate itself: **cross-fitted generalised eigenvalues**. They bound the oblique steps that broke the first repair, and they reduce exactly to WiSE-FT when the curvature carries no signal (§4). In the pilot they lost 4–12% of the achievable improvement, against 18–50% for in-sample eigenvalues.

**The direction, in one line: keep K, drop Q.** Let K(θ) = E_x KL(p_θ0(·|x) ‖ p_θ(·|x)) be the drift on harmful prompts and Q(δ) = ½δᵀF_hδ its second-order expansion at θ0. Round one used Q everywhere. The forecast was Q(Δ); the repair minimised Q(Δ′) + μR(Δ′ − Δ); and the standard safe fine-tuning penalty, EWC, is λQ(θ − θ0). The blind spot is a property of Q, not of K. Along an update whose logits move linearly, the exact drift is

$$K_{\mathrm{lin}}(t)=\mathbb{E}\big[A(f^0+t\,u)-A(f^0)-t\,\mathbb{E}_{p^0}u\big],\qquad K_{\mathrm{lin}}''(t)=\mathbb{E}\,\mathrm{Var}_{p_t}(u),$$

with A the log-partition function. The curvature is the variance under the *current* model p_t, which grows as a refusal weakens. Q freezes it at p⁰, where it is about e^(−m₀). So the plan replaces Q by K, or by the decision margin K protects, wherever Q was used, and pays with forward passes instead of curvature:

| Use | Round one (quadratic surrogate) | Plan (exact objective) |
| --- | --- | --- |
| Forecast | Q(Δ) | K_lin(1) from one JVP, plus the measured non-linearity factor K(θ1)/K_lin(1) (E1, §2) |
| Repair | argmin Q(Δ′) + μR(Δ′ − Δ) | argmin over s ∈ [0,1]^252 of K(θ0 + Σ_m s_mΔW_m) + μL, started at the best WiSE-FT point (E2, §3) |
| Fine-tuning safely | L_task + λQ(θ − θ0), i.e. EWC | L_task + βK(θ), or L_task + β·E max(0, m\* − m_θ(x)) (E9, §10) |
| Curvature, where still used | in-sample generalised eigenvalues | cross-fitted, shrunk eigenvalues (E3, §4) |

The gradients show why. EWC's is λF_h(θ − θ0), whose weight on a refusal stays ∝ e^(−m₀) however far the refusal has fallen. The exact KL's is E[Jᵀ(p_θ − p_θ0)], about σ(−m_θ) in size for a confident refusal, so its pull grows as the margin falls. The margin hinge's is −∇m_θ whenever m_θ < m\*: O(1) at any confidence, and it constrains only the decision. The learned merge only shrinks each module's update (s ∈ [0,1]), so it cannot take the oblique steps of fact 2, and interpolation is a special case of it.

---

## 0. Summary

| ID | Question | New element | Key prediction (falsifiable) |
| --- | --- | --- | --- |
| E1 | Why is the forecast 24–111× too low? | Exact log-partition forecast; three-factor gap decomposition by JVP | The confidence amplification factor is ≥ 10× in every language; the K-FAC error is within 3× |
| E2 | Can a function-space repair beat WiSE-FT? | Learned per-module (then per-component) weights on the exact objective, started from the best WiSE-FT point | Beats WiSE-FT on dev drift at equal task loss in all three languages (pilot: 0.2–8.6% harmful compliance, against 25–99%) |
| E3 | Is the two-sided eigenbasis estimation noise? | Cross-fitted, shrunk generalised eigenvalues; an oblique-step bound | No ceiling breaks; trust ratios ≤ 2 with no trust region; drift ≤ WiSE-FT |
| E4 | Can we repair discrimination, not just drift? | Two-sided drift (harmful + benign-risky); refusal Youden index | English over-refusal ≤ 20% with harmful compliance ≤ 8% |
| E5 | Can the task loss be predicted? | Cross-fitted validation gradient in the closed form | Predicts WiSE-FT's task-loss optimum (measured s = 0.7) within ±0.1 |
| E6 | Does safety survive 4-bit deployment? | Trace forecast for rounding error; the quadratic regime depends on direction | Quantisation drift is forecast within 2–3× (pilot: 1.0–2.4×), unlike the fine-tune's 24–39× |
| E7 | Does a repair transfer across languages? | Gauge-invariant subspace angles; transfer forecast | English→Hindi transfer < English→Hinglish |
| E8 | Measurement hygiene | Human labels, seeds | Cohen's κ ≥ 0.6 per judge and language |
| E9 | Can fine-tuning itself avoid the blind spot? | Exact-KL or refusal-margin penalty in place of EWC | Harmful compliance ≤ 8% at equal task loss, with EWC worse at every λ (pilot: EWC 78–100%; KL and margin 0%) |

**Order.** E1 runs first and decides the path. If confidence amplification dominates the gap, the curvature model is the wrong *kind* of model, so E2 and E9 come next. If K-FAC's own error dominates, E3 comes next. E5 shares E1's job. With limited credits, E1, E5 and the module-level part of E2 come first.

---

## 1. Five facts any next method must explain

1. **Forecast.** The quadratic forecast is 24–39× too low on run A's final checkpoint (0.0068 / 0.0075 / 0.0100 predicted against 0.262 / 0.218 / 0.238 measured) and 24–111× over all 36 checkpoint–language pairs. Along the update, drift scales as t^0.72 for t from 0.25 to 0.95 (§5.4).
2. **Explosion.** The unbounded repair took steps up to 94× the update. The cause was nearly parallel eigenvectors with very different shrink factors (an oblique projector, §3.9). Two output factors broke the 1/γ_c eigenvalue ceiling, so they were numerically indefinite (§5.5).
3. **Regret.** Assumption A1 fails: ρ_g is 0.7–14 at every μ, and moving from θ1 toward θ0 *improves* dev task loss (WiSE-FT's best is near α = 0.3, §5.6).
4. **Simpler wins.** At equal task loss, WiSE-FT, RegMean and diagonal Fisher all leave less drift than the pencil. Diagonal Fisher, built from the *same* factors, removes 99% of drift and keeps 77% of the task gain (§5.7).
5. **Discrimination.** KL drift and judged behaviour disagree (§6.5). The damage is a loss of discrimination. Below, the **refusal Youden index**, J = P(refuse | harmful) − P(refuse | safe), is computed post hoc from the §5.8 test numbers, in percentage points:

| J (points) | θ0 | θ1 | Pencil μ\* | RegMean | WiSE-FT α = 0.5 |
| --- | --- | --- | --- | --- | --- |
| English | 46.3 | 17.8 | 31.0 | 40.4 | 51.5 |
| Hindi | 47.0 | 55.2 | 50.8 | 53.6 | 55.5 |
| Hinglish | 7.1 | 13.6 | 4.4 | 6.0 | 9.1 |

The fine-tune collapsed English discrimination from 46 to 18 points. WiSE-FT restores it fully; the pencil restores it only partly. Hindi discrimination did not fall. In Hinglish, the aligned model itself barely discriminates.

---

## 1.5 Pilot evidence (CPU)

**Set-up.** The pilot model is a 2-layer transformer (d = 64), trained from scratch on a synthetic language with four prompt types:

- ordinary prompts, answered by reversing the words;
- prompts containing a trigger token, which must be refused;
- prompts containing a look-alike token, which must be answered normally;
- a new task marker, unseen by θ0, whose answer shifts each word by one.

A rank-8 LoRA fine-tune on the new task alone destroys the refusal: harmful compliance goes from 0% to 97–100%, and task accuracy reaches 82–90%. Everything can be computed exactly at this size. Three seeds, except E9 (one seed). No natural-language or harmful text is involved. The scripts are `pilots/toy.py`, `pilots/prevent.py` and `pilots/fmatrix.py`.

| Claim | Pilot result | Verdict |
| --- | --- | --- |
| Fisher forecasts run low on confident answers (§2) | Exact second-order forecast 1,200–12,000× too low (median aligned margin ≈ 12.5) | Holds |
| Drift is quadratic only at first, then faster than quadratic as answers flip | Local exponent 2.0–2.2 at the smallest t, rising to 7–10 | Holds |
| The gap splits into confidence × non-linearity | Confidence (log-partition) factor 43–3,500×; logit non-linearity 0.3–280× | Both matter; which one dominates varies by seed |
| Fisher-weighted merging is blind (§3) | Diagonal-Fisher merging leaves harmful compliance at 96–97% at every μ | Holds |
| Learned module weights beat interpolation (§3) | At 75% of the task gain: 0.2%, 0.2%, 8.6% harmful compliance, against WiSE-FT's 25%, 64%, 99%. At 90%: 1.0% and 1.4% against 55% and 95% (seed 3 had no setting that reached 90%) | Holds, on the exact objective |
| The linearised (convex) version works | Harmful compliance 56–99%: measured drift was 3–15× the linearised value at a quarter of the update and up to 2,100× at half | **Fails**: the plan now uses the exact objective |
| EWC is blind; exact KL and margin penalties are not (§10) | EWC λ = 10…10⁴: 78–100%. Exact KL: 0% at 99.1% task accuracy. Margin hinge: 0% at 99.9% | Holds (1 seed) |
| Cross-fitting fixes noisy curvature (§4) | Simulated pencil: in-sample loses 18–50% of the achievable improvement and overshoots (steps up to 2× the update); cross-fitted loses 4–12% and never overshoots; shrunk equals WiSE-FT under the null | Holds |
| Wachter's law gives the noise band | Observed [0.17, 6.1] against predicted [0.15, 6.9] (p = 64, n = 192) | Holds |
| Quadratic forecasts work for diffuse perturbations (§7) | 4- and 3-bit rounding: within 1.0–2.4×. But random perturbations as large as the whole update also break the quadratic | **Revised**: what matters is how far a perturbation moves confident margins. At 30% of the update's size, the fine-tune direction is 17–640× off and random directions 3–6× off |
| The held-out gradient predicts the best interpolation point (§6) | Predicted 0.995–1.023 against measured 1.0 | Consistent, but a weak test: the pilot's optimum sat at the end of the line |

---

## 2. E1. Why the forecast is low: an exact decomposition

### 2.1 Drift along a line, exactly

For prompt x and position i of θ0's teacher-forced answer, let $f_i(\theta)\in\mathbb{R}^V$ be the logits, $p_i(\theta)=\mathrm{softmax}(f_i)$ and $A(f)=\log\sum_y e^{f_y}$ the log-partition function. Drift is $S(\theta)=\mathbb{E}_{x,i}\,\mathrm{KL}\big(p_i(\theta_0)\,\Vert\,p_i(\theta)\big)$ (Definition 1). Let

$$u_i=\frac{d}{dt}f_i(\theta_0+t\Delta)\Big|_{t=0}=J_i\,\Delta$$

be the directional change of the logits: one forward-mode Jacobian–vector product (JVP).

**Lemma 1 (the KL along a linear logit path).** If the logits move linearly, $f_i(t)=f_i^0+t\,u_i$, then exactly

$$S_{\mathrm{lin}}(t)=\mathbb{E}_{x,i}\Big[A(f_i^0+t\,u_i)-A(f_i^0)-t\,\mathbb{E}_{p_i^0}[u_i]\Big],$$

$$S_{\mathrm{lin}}'(t)=\mathbb{E}\big[\mathbb{E}_{p_i(t)}u_i-\mathbb{E}_{p_i^0}u_i\big],\qquad S_{\mathrm{lin}}''(t)=\mathbb{E}\big[\mathrm{Var}_{p_i(t)}(u_i)\big]\ \ge 0 .$$

*Proof.* $\mathrm{KL}(p^0\Vert p^t)=\sum_y p^0_y\big[(f^0_y-A(f^0))-(f^0_y+t u_y-A(f^0+tu))\big]$, which is the first line. The derivatives of $A$ along a line are the mean and variance of $u$ under the current softmax. ∎

At $t=0$, $S_{\mathrm{lin}}''(0)=\mathbb{E}\,\mathrm{Var}_{p^0}(u)=\Delta^\top F_h\Delta$, because the softmax Fisher is $J^\top(\mathrm{diag}\,p-pp^\top)J$. So the second-order forecast ½ΔᵀF_hΔ is exactly the second-order Taylor polynomial of $S_{\mathrm{lin}}$. The question is how far that polynomial is from $S_{\mathrm{lin}}$ itself.

### 2.2 Confident answers amplify the gap

**Proposition 1.** Take a position with two candidate tokens, θ0 margin $m_0=f_{y^*}-f_{y'}>0$, and suppose the update reverses the margin by t = 1 ($u_{y'}-u_{y^*}=2m_0$). Then

$$\frac{\mathrm{KL}\text{ at }t=1}{\text{second-order forecast}}=\frac{\sinh m_0}{m_0}.$$

*Proof.* At t = 1 the margin is −m₀, so with $p^*=\sigma(m_0)$, $\mathrm{KL}=(2p^*-1)\log\frac{p^*}{1-p^*}=m_0\tanh(m_0/2)$. The forecast is $\tfrac12\,p^*(1-p^*)(2m_0)^2=m_0^2/(2\cosh^2(m_0/2))$. Their ratio is $2\sinh(m_0/2)\cosh(m_0/2)/m_0=\sinh(m_0)/m_0$. ∎

| θ0 confidence p₀ | margin m₀ | KL / second-order forecast |
| --- | --- | --- |
| 0.9 | 2.20 | 2.0× |
| 0.99 | 4.60 | 10.8× |
| 0.999 | 6.91 | 72× |
| 0.9999 | 9.21 | 543× |

The aligned model's greedy answers are mostly confident tokens, and drift is concentrated in the tokens whose answer the fine-tune changes. For those tokens, a Fisher forecast is low by one to two orders of magnitude **even with exact curvature**. The cause is that the curvature at θ0, $\mathrm{Var}_{p^0}(u)$, is suppressed by the confidence itself. It grows by a factor of order $e^{m_0}$ once the distribution starts to move.

This also corrects a reading of the drift law. A curve that rises like t² and then bends to t^0.72 would make a quadratic forecast too *high*. The forecast is too *low*, so somewhere before a quarter of the update drift must grow *faster* than quadratically. Proposition 1 says when that happens: as confident tokens flip. Lemma 1 also shows $S_{\mathrm{lin}}$ is convex, so the measured concavity on [0.25, 1] cannot come from the softmax. It must come from the logits themselves changing sub-linearly in t, for example through normalisation layers.

### 2.3 The three-factor decomposition

Write $\hat S_{\mathrm{KFAC}}=\tfrac12\Delta^\top\hat F_h\Delta$ and $q=\Delta^\top F_h\Delta$ (exact, from one JVP). Then, as an identity,

$$\frac{S_{\mathrm{meas}}(1)}{\hat S_{\mathrm{KFAC}}}=\underbrace{\frac{q}{\hat q}}_{\text{K-FAC error}}\times\underbrace{\frac{S_{\mathrm{lin}}(1)}{q/2}}_{\text{confidence amplification}}\times\underbrace{\frac{S_{\mathrm{meas}}(1)}{S_{\mathrm{lin}}(1)}}_{\text{logit non-linearity}} .$$

The first two factors come from one JVP per batch; the third needs one extra forward pass at θ1. We also report the logit-linearity error $\varepsilon=\mathbb{E}\Vert f(\theta_1)-f^0-u\Vert/\mathbb{E}\Vert f(\theta_1)-f^0\Vert$, and $S(t)$ on the grid $t=2^{-k}$, k = 0…6. That grid gives the radius τ below which drift is quadratic, and the local exponent $d\log S/d\log t$.

### 2.4 Predictions

- **P-E1a.** The confidence amplification factor is ≥ 10× on run A's final checkpoint in every language (pilot: 43–3,500×). *Falsified if < 3×.*
- **P-E1b.** The K-FAC factor q/q̂ lies in [1/3, 3]. *If it is ≫ 3, K-FAC (cross-position terms, the shared preamble) is the main problem, and E3 becomes the priority.*
- **P-E1c.** The logit non-linearity factor is reported with its uncertainty. If it lies within [1/3, 3], $S_{\mathrm{lin}}$ becomes a usable forecast, with Spearman ρ ≥ 0.6 against measured drift over the 36 pairs (P2's bar; the quadratic reached 0.50). In the pilot this held in 2 of 3 seeds; in the third, non-linearity made $S_{\mathrm{lin}}$ 280× low.
- **P-E1d.** Measured S(t) has local exponent ≈ 2 only for t ≤ τ ≤ 0.05 (pilot: 2.0–2.2 at the smallest t, rising to 7–10).
- **P-E1e.** On [0.25, 1], the local exponent of $S_{\mathrm{lin}}$ is ≈ 1 (flipped tokens grow linearly), against 0.72 measured. The difference is the logit saturation.

### 2.5 Protocol

`torch.func.jvp` on teacher-forced θ0 answers. Inputs: the 322 dev harmful prompts and the 12 checkpoints × 3 languages (each checkpoint has its own Δ). Exact per-module $q_m$ (252 JVPs) gives an exact leverage map, whose Spearman ρ with the K-FAC map re-tests P5 (including the Hindi anomaly).

---

## 3. E2. Learn the safety merge on the exact objective

### 3.1 The problem

Repair with one coefficient per module, $\theta(s)=\theta_0+\sum_m s_m\,\Delta W_m$, $s\in[0,1]^M$, M = 252, and choose s by minimising the exact objective

$$\min_{s\in[0,1]^M}\ S(\theta(s))+\beta\,S_b(\theta(s))+\mu\,\big(L_{\mathrm{val}}(\theta(s))-L_{\mathrm{val}}(\theta_1)\big).$$

Here:

- S is the KL drift on harmful prompts (Definition 1);
- $S_b$ is the benign-risky drift of E4 (β = 0 without E4);
- $L_{\mathrm{val}}$ is the task cross-entropy on held-out rows.

Each evaluation is a forward pass. The gradient in s is one backward pass, since $\partial/\partial s_m=\langle\nabla_{W_m},\Delta W_m\rangle$. Start from the best WiSE-FT point $s=c^\star\mathbf 1$ (a one-dimensional search over c), then run projected L-BFGS-B.

**Proposition 2.**

1. *Never worse than interpolation.* $s=c\,\mathbf 1$ is WiSE-FT with α = 1 − c, and it is feasible. Started at the best such point, a descent method returns a merge whose fitting objective is at most WiSE-FT's.
2. *No blow-up.* Each step is a shrink, $\delta_m=(s_m-1)\Delta W_m$, so $\Vert\delta_m\Vert\le\Vert\Delta W_m\Vert$ in **every** norm. The trust ratio is ≤ 1 by construction, and the oblique failure of §3.9 cannot occur.
3. *Gauge-invariant.* The objective depends on θ(s) only through the network's function, and the V/O gauge maps $s_m\Delta W_m$ to $s_m M\Delta W_m$. So the optimal s, and the repaired function, are identical in every gauge.
4. *No A1.* The task term is the held-out cross-entropy itself, first-order part included, so the regret assumption that failed (§5.6) is not needed.

*Proof.* (1) holds by feasibility and descent; (2) and (3) follow from the parametrisation; (4) holds because $L_{\mathrm{val}}$ is not approximated. ∎

In the pilot, the final fitting objective was below WiSE-FT's at every μ and seed, and the repaired models beat WiSE-FT on held-out drift and harmful compliance at equal task loss (§1.5).

**A cheaper linearised version fails.** Linearising the logits ($f\approx f^0+Us$, with per-module JVPs and Lemma 1's log-partition) makes the problem convex and needs only one JVP per module. In the pilot it failed: harmful compliance stayed at 56–99%. The logits move far from linearly along the path: at half the update, measured drift was up to 2,100× the linearised value. E1's non-linearity factor tells us whether the linearised version could ever be trusted at Tiny Aya scale.

The held-out task rows are cross-fitted: the 300 rows are split into two folds; fit on one, evaluate on the other, then swap. No row is used to both fit and score.

### 3.2 Finer granularity, still invariant

Diagonal Fisher's win (fact 4) suggests that granularity finer than a module matters. Decompose each update in the curvature-whitened basis: $G_t^{1/2}\Delta W A_t^{1/2}=\sum_k\sigma_k\tilde u_k\tilde v_k^\top$ (rank ≤ 16 for our LoRA). Then $\Delta W=\sum_k\sigma_k y_kx_k^\top$ with $y_k=G_t^{-1/2}\tilde u_k$, $x_k=A_t^{-1/2}\tilde v_k$, and the repair is $\Delta W_m'=\sum_k s_{mk}\sigma_ky_kx_k^\top$. Under the gauge, $G^{1/2}\Delta W A^{1/2}$ changes only by an orthogonal factor on the left. So the σ_k are unchanged and the components transform equivariantly: invariance holds. To prevent the cancelling-component failure, add the trust constraint $\Vert(\Delta W_m'-\Delta W_m)A_t^{1/2}\Vert_F\le\Vert\Delta W_mA_t^{1/2}\Vert_F$, which is affine-in-s inside a norm, hence convex. To keep it tractable, this refinement is applied only to the 30 modules that matter most in the module-level solution (480 extra JVPs). Component coefficients are learned in the same way.

### 3.3 Predictions

- **P-E2a.** The module-level merge beats WiSE-FT on dev drift at matched task NLL in all three languages (paired bootstrap, Holm). *If not, module selectivity is too coarse.*
- **P-E2b.** At task NLL 1.803, the component-level merge reaches dev drift ≤ 0.060 (diagonal Fisher's value). *If not, per-weight selectivity is what matters, and the next step is learned per-weight coefficients with a diagonal prior.*
- **P-E2c.** Started from the best WiSE-FT point, the final fitting objective is below WiSE-FT's at every μ (pilot: at every μ and seed).

## 4. E3. Is the two-sided eigenbasis estimation noise? A cross-fitted pencil

### 4.1 The pencil is a two-sample F-matrix problem

Each Kronecker side solves $A_hx=\alpha A_tx$ with two sample second-moment matrices. When the population matrices are proportional (no harm-specific direction), the generalised eigenvalues of $S_1S_2^{-1}$ in dimension p, with sample sizes $n_1,n_2$, follow Wachter's law with support

$$\Big[\Big(\tfrac{1-h}{1-y_2}\Big)^2,\ \Big(\tfrac{1+h}{1-y_2}\Big)^2\Big],\qquad h=\sqrt{y_1+y_2-y_1y_2},\quad y_j=p/n_j<1 .$$

Illustration: p = 2,048 and an effective 10,000 samples per side give [0.25, 4.1]. **Pure sampling noise spreads the eigenvalues over a 16-fold range.** On MLP-width sides (p = 11,008), the denominator is singular unless the effective sample size exceeds p, which is why we kept only the top 512 pairs. The effective size is far below the token count: positions within a sequence are correlated, and the 366-token preamble is identical across all sequences. Spikes below a detection threshold cannot be recovered from sample generalised eigenvalues by any method (Nadakuditi & Silverstein, 2010). Spikes above it separate from the Wachter bulk (Dharmawansa, Johnstone & Onatski, 2014).

### 4.2 Why noise makes the closed form explode

**Lemma 2 (oblique step and eigenvalue gaps).** Let two eigen-directions $e_1,e_2$ make an angle θ, carry shrink factors $m_k=\mu/(\lambda_k+\mu)$, and span Δ. Then

$$\frac{\Vert\delta\Vert}{\Vert\Delta\Vert}\ \le\ |1-m_1|+\frac{|\log\lambda_2-\log\lambda_1|}{4\sin\theta}.$$

*Proof.* Write $\Delta=a e_1+b e_2$; then $|b|\le\Vert\Delta\Vert/\sin\theta$. The step is $\delta=(m_1-1)\Delta+(m_2-m_1)b\,e_2$, and $|dm/d\log\lambda|=\lambda\mu/(\lambda+\mu)^2\le\tfrac14$. ∎

With in-sample eigenvalues, two nearly parallel noise directions can differ in log-eigenvalue by O(1) (the Wachter spread) while θ → 0. The bound is then unbounded, and we measured 94×.

### 4.3 Cross-fitting removes the singularity

Split the factor sequences into halves A and B. Take eigenvectors $V$ from half A, and eigenvalues from half B as Rayleigh quotients:

$$\lambda_i^{\mathrm{cf}}=R_B(v_i)=\frac{v_i^\top F_h^{B}v_i}{v_i^\top H_t^{B}v_i}.$$

Repair with the PSD matrix $\tilde F_h=H_t^AV\Lambda^{\mathrm{cf}}V^\top H_t^A$, which V diagonalises, so the closed form is unchanged. Swap the halves and average the two repaired updates. Because the objective J is convex, the average is no worse than the mean of the two.

**Corollary.** $R_B$ is a fixed smooth function of the direction, independent of half A, with $\Vert\nabla\log R_B\Vert\le 2(\kappa(F_h^B)+\kappa(H_t^B))=:L_B$ on the solved subspace. So $|\log\lambda_2^{\mathrm{cf}}-\log\lambda_1^{\mathrm{cf}}|\le L_B\theta$, and

$$\frac{\Vert\delta\Vert}{\Vert\Delta\Vert}\le 1+\frac{L_B\,\theta}{4\sin\theta}\le 1+\frac{\pi}{8}L_B ,$$

**independent of the angle.** Nearly parallel directions get nearly equal shrink factors, so their large opposite components no longer separate. The same argument holds in any fixed inner product, including the input-weighted one the trust ratio uses.

### 4.4 Shrinkage toward the bulk, and the WiSE-FT limit

Group-jackknife over sequence groups gives each direction's variance $\sigma_i^2$ of $\log\lambda_i^{\mathrm{cf}}$. Shrink in log space toward the median $\bar\kappa$:

$$\log\tilde\lambda_i=\log\bar\kappa+\Big(1-\frac{\sigma_i^2}{s^2}\Big)_+\big(\log\lambda_i^{\mathrm{cf}}-\log\bar\kappa\big),$$

where $s^2$ is the spread of $\log\lambda^{\mathrm{cf}}$ across directions (an empirical-Bayes rule).

**Proposition 3.** If every direction is noise-dominated ($\sigma_i^2\ge s^2$), then $\tilde\lambda_i\equiv\bar\kappa$ and the repair is *exactly* WiSE-FT with α = κ̄/(κ̄+μ) (Corollary 1). Otherwise it is WiSE-FT plus corrections confined to the reliable directions. On a Kronecker module, $\tilde\lambda_{ij}=\tilde\alpha_i\tilde\gamma_j$: the bulk-times-bulk block gets one uniform factor, and only rows and columns of reliable spikes deviate from it.

So the calibrated pencil cannot be "worse than WiSE-FT because of noise". Either it finds signal, or it becomes WiSE-FT.

### 4.5 Numerical hygiene

- fp64 (or compensated fp32) accumulation.
- Factors estimated only on positions after the shared preamble. A large common component C in $A_h=C+\tilde A_h$ and $A_t=C+\tilde A_t$ compresses the spectrum toward 1 and invites cancellation in fp32; we hypothesise this is where the ceiling breaks came from.
- 1,024 sequences per side instead of 256.

### 4.6 Predictions

- **P-E3a.** Zero ceiling breaks. *If any remain, the indefiniteness is structural, not numerical.*
- **P-E3b.** With no trust region at all, every module's trust ratio is ≤ 2.
- **P-E3c.** Dev drift at matched task NLL ≤ WiSE-FT's in all three languages. *If not, the number of reliable spikes says why: with none, Proposition 3 makes the method WiSE-FT, and the honest conclusion is that two-sided curvature carries no usable signal at this sample size.*
- **P-E3d.** The top of the leverage map falls from 8.2×10⁵ to below 10³, and the reliable spikes stay in layers 32–35 (a P5 re-test).

## 5. E4. Repair discrimination, not just drift

### 5.1 Two-sided drift

Add drift on *benign-but-risky* prompts, $S_b(\theta)=\mathbb{E}_{x\sim D_b}\mathrm{KL}(p_{\theta_0}\Vert p_\theta)$, measured on θ0's answers. Its Fisher $F_b$ penalises exactly the move toward over-refusal, which $F_h$ cannot see. The pencil becomes $(F_h+\beta F_b,\ H_t)$. A sum of Fishers transforms like each one, so Proposition 7's gauge invariance carries over unchanged. In E2, $\hat S+\beta\hat S_b$ stays convex.

Data: an OR-Bench-80K subset (Cui et al., 2025), converted to Hindi and Hinglish with the same CuratorKIT pipeline, for factors and dev. XSTest stays test-only, with the leak check rerun across all sets.

### 5.2 The contrast spectrum

$F_hv=\nu F_bv$ separates harm-specific directions (ν ≫ 1) from directions that move benign behaviour (ν ≪ 1). Hypothesis: the over-refusal jump (English 13.5% → 55.0%) lives in a few low-ν directions. Test: the share of the update's benign drift ½ΔᵀF_bΔ captured by the bottom-k contrast directions.

### 5.3 A metric that does not reward the damage

The pre-registered refusal metric rewards the fine-tune's damage (§5.9). The refusal Youden index J = P(refuse | harmful) − P(refuse | safe) does not: θ1's English J is 17.8 against θ0's 46.3 (§1). J becomes the primary behavioural metric, with a bootstrap over both prompt sets.

### 5.4 Predictions

- **P-E4a.** English test over-refusal ≤ 20% with English harmful compliance ≤ 8% (the pencil today: 31.5% and 7.4%).
- **P-E4b.** English J ≥ 45 (θ0 46.3; pencil 31.0; WiSE-FT 51.5).
- **A structural limit.** Every KL-to-θ0 objective is minimised by θ0 itself, whose Hinglish J is 7.1. So these objectives give no incentive to discriminate better than θ0 in Hinglish. Going beyond θ0 needs a different target, for example consistency with the model's own English behaviour on parallel prompts. We flag this; it is not planned here.

---

## 6. E5. Predict the task loss with a cross-fitted validation gradient

Keep the linear term of the task loss, on held-out rows: $R(\delta)=g_v^\top\delta+\tfrac12\delta^\top H\delta$. Stationarity of $\tfrac12\Delta'^\top F_h\Delta'+\mu R(\Delta'-\Delta)$ gives, in pencil coordinates ($h=V^\top g_v$),

$$z_i=\frac{\mu\,(c_i-h_i)}{\lambda_i+\mu},\qquad\text{i.e. the task target becomes }\ \Delta^\star=\Delta-H^{-1}g_v .$$

This is a Newton step back from θ1 toward the held-out task optimum. It is the gradient-mismatch correction that Daheim et al. (2024) identify for merging, here inside the pencil's closed form.

Along the WiSE-FT line $\theta_0+s\Delta$, the same expansion predicts the task-optimal point:

$$s^\star=1-\frac{g_v^\top\Delta}{\Delta^\top H\Delta}.$$

Both numbers come from one JVP over held-out rows, with no K-FAC: $g_v^\top\Delta=\mathbb{E}[(p^1-e_y)^\top w]$ and $\Delta^\top H\Delta=\mathbb{E}\,\mathrm{Var}_{p^1}(w)$, where $w=J\Delta$ at θ1. Cross-fit over two folds of the 300 held-out rows.

- **P-E5a.** s\* ∈ [0.6, 0.8]. The measured WiSE-FT dev optimum is at α = 0.3, that is s = 0.7 (dev NLL 1.775).
- **P-E5b.** The predicted task-NLL change across μ correlates with the measured change at Pearson ≥ 0.9. (Today predicted regret rises while measured NLL stays flat.)

---

## 7. E6. Quantised deployment: when quadratic forecasts *do* work

Track 2b asks for results on a deployed, quantised model. Let ε = Q(θ) − θ be the rounding error: zero-mean and independent of the data. Then the first-order term vanishes in expectation and

$$\mathbb{E}\,S(\theta+\varepsilon)\approx S(\theta)+\tfrac12\,\mathrm{tr}\big(\nabla^2S\,\Sigma_\varepsilon\big)\approx S(\theta)+\tfrac12\sum_m\sigma_m^2\,\mathrm{tr}(A_m)\,\mathrm{tr}(G_m),$$

with $\sigma_m^2=\Vert Q(W_m)-W_m\Vert_F^2/\#W_m$ measured exactly. The Hanson–Wright inequality (Rudelson & Vershynin, 2013) concentrates $\varepsilon^\top F\varepsilon$ around $\sigma^2\mathrm{tr}F$, with relative fluctuation of order $\Vert F\Vert_F/\mathrm{tr}F$, which is small when the curvature has high effective rank.

The contrast with Proposition 1 is the point, with one correction from the pilot. What decides whether a perturbation stays in the quadratic regime is how far it moves confident margins, not whether it is random:

- rounding error is small (at 4 bits, 7–8% of the fine-tune's norm in the pilot) and spread over all directions, so margins barely move;
- the fine-tune direction is aimed at exactly the tokens it changes.

In the pilot, the trace forecast was within 1.0–2.4× for 4- and 3-bit rounding. At 30% of the update's size, the fine-tune direction was 17–640× off and random directions 3–6× off. Random perturbations as large as the whole update broke the model outright. **Quadratic forecasts are trustworthy for small, spread-out perturbations such as quantisation, and untrustworthy along a fine-tune.**

- **P-E6a.** The trace forecast is within 3× of the measured drift increase at NF4 and at 8-bit (pilot: 1.0–2.4× at 4 and 3 bits). The fine-tune's forecast is 24–39× low.
- **P-E6b.** Under NF4, harmful compliance, over-refusal and J of the pencil and WiSE-FT models each move by ≤ 2 points. If quantisation drift exceeds the drift the repair removes (0.057 nats/token), repair after quantising: E2 with JVPs taken at the quantised model.

Protocol: bitsandbytes 8-bit and NF4, plus llama.cpp GGUF Q4_K_M; θ0, θ1, the pencil at μ\* and WiSE-FT α = 0.5, on test; drift on dev.

---

## 8. E7. Cross-lingual structure (P7)

For H_t-orthonormal bases $V_a,V_b$ of the top-k pencil subspaces of two languages, the cosines of the principal angles are the singular values of $V_a^\top H_tV_b$. Under the gauge, V → MV and $H_t\to M^{-\top}H_tM^{-1}$, so $V_a^\top H_tV_b$ is unchanged: the angles are gauge-invariant. The transfer of a repair fitted on language a to language b is forecast as $T_{a\to b}=1-S_b(\Delta'_a)/S_b(\Delta)$, using E1's log-partition forecast, and measured with single-language repairs.

- **P-E7a.** Hindi's subspace has the smallest overlap with English's (consistent with Hindi sharing only 2 of the top-10 leverage modules, §5.3).
- **P-E7b.** $T_{\mathrm{en}\to\mathrm{hi}}<T_{\mathrm{en}\to\mathrm{Hinglish}}$, and forecast transfer is within ±0.15 of measured.

---

## 9. E8. Measurement hygiene

- **Human labels.** 150 per language per judge (900 in total), with Cohen's κ. Absolute rates are reported only where κ ≥ 0.6.
- **Seeds.** Run B plus two more seeds of the run A configuration, with evaluations, for seed spread and P2 power.
- **Prevention (P4).** Add E2's drift term to the fine-tuning loss as a KL penalty to θ0 on harmful prompts: the training-time version of the same objective.


---

## 10. E9. Safe fine-tuning beyond EWC

The standard way to protect a behaviour during fine-tuning is an EWC-style penalty, $\tfrac{\lambda}{2}\sum_wF_w(\theta_w-\theta_{0,w})^2$, with F the Fisher of that behaviour at θ0 (Kirkpatrick et al., 2017). The KL's Hessian at θ0 is the Fisher (WRITEUP Proposition 1), so **EWC is exactly the second-order Taylor expansion of the KL penalty $\mathrm{KL}(p_{\theta_0}\Vert p_\theta)$ at θ0.** The blind spot therefore applies verbatim. A refusal with margin m₀ gets Fisher weight ∝ p₀(1 − p₀) ≈ e^(−m₀), and once the fine-tune starts to reverse it, the true KL exceeds the EWC penalty by sinh(m₀)/m₀ (Proposition 1). Protecting a refusal at margin 12 would need λ about 10⁴ times larger than for an uncertain token, so no single λ suits both.

Two replacements, each needing one extra forward pass on harmful prompts per step:

- **Exact KL penalty**, $\beta\cdot\mathrm{KL}(p_{\theta_0}\Vert p_\theta)$ on θ0's answers. Its curvature is the variance under the current model (Lemma 1), so its pull grows as the refusal weakens. Its gradient at θ0 is zero, so on very confident refusals it acts late.
- **Margin hinge**, $\beta\cdot\mathrm{mean}\,\max(0,\,m^\star-m_\theta(x))$, with $m_\theta$ the refusal logit minus the best other logit at the answer's first token. Its gradient is O(1) as soon as the margin falls below m\*, at any confidence. It protects the decision rather than the whole distribution, so the rest of the answer may change, which suits a fine-tune that legitimately changes style.

**Pilot** (1 seed; the same LoRA fine-tune and batch order for every penalty):

| Penalty | Harmful compliance | Task accuracy | Drift |
| --- | --- | --- | --- |
| none | 100% | 40.5% | 3.96 |
| EWC, λ = 10 / 10² / 10³ / 10⁴ | 99.8% / 100% / 83.6% / 77.9% | 98.8–100% | 2.3–4.0 |
| exact KL, β = 1 | 0% | 99.1% | 0.001 |
| margin hinge, β = 0.3 | 0% | 99.9% | 0.038 |
| margin hinge, β = 0.03 | 0% | 89.0% | 2.48 |

The last row shows "decision, not distribution": the refusal holds while the rest of the distribution drifts. With this batch order the unpenalised run learned the task poorly, so the penalties also acted as stabilisers.

- **P-E9a.** On the Hindi fine-tune, at equal held-out task loss, both replacements keep English harmful compliance ≤ 8% (unpenalised: 22.5%), and EWC is worse at every λ on the grid.
- **P-E9b.** The λ that EWC needs to match them grows with the median refusal margin.

Protocol: run A's recipe with each penalty, computed on the harmful factor-split prompts (never the test split), plus data mixing with refusal examples as a further baseline: four fine-tunes and their evaluations.

---

## 11. Order

1. **E1 first**, together with E5, which reuses its Jacobian–vector products. E1's decomposition decides the path.
2. **If confidence amplification dominates the gap**, the curvature model is the wrong kind of model: E2, then E9. **If K-FAC's own error dominates**, E3 comes next.
3. **Then** E4, E6, E7 and E8.

With limited credits, E1, E5 and the module-level E2 come first.

## 12. Expected before → after: commitments and forecasts

**Commitments**, which hold whatever the results show:

- **Pre-registration.** Every prediction and failure condition in this plan is fixed before its experiment runs; deviations go in a dated log; every outcome is reported, refutations included.
- **Test splits.** Tuning uses dev only; each test split is touched once per method.
- **E1** always yields the three-factor decomposition, because it is an identity computed from one JVP and one forward pass per checkpoint.
- **E2** is, by Proposition 2, never worse than the best interpolation on its fitting objective; never steps beyond the original update; and is exactly gauge-invariant. These properties are promised; a test-set win is not.
- **E3** never steps beyond (1 + πL_B/8) times the update (Lemma 2), and it falls back exactly to WiSE-FT when no direction is reliable (Proposition 3).
- **E6, E8 and E9** are procedures: quantised test results, human labels with Cohen's κ, three seeds, and a like-for-like comparison of the four penalties.
- **Release.** Everything new is released with the same rule: labels and hashes only for harmful prompts.
- **Order.** E1 runs first, and the §0 decision rule is followed.

**Forecasts**, which the experiments may refute:

| Quantity | Now | Expected | From |
| --- | --- | --- | --- |
| Forecast error, run A final, English | 39× too low | decomposed; confidence factor ≥ 10× | E1 |
| Spearman ρ, log-partition forecast vs measured KL (36 pairs) | 0.50 (quadratic) | ≥ 0.6 if non-linearity is small | E1 |
| Dev drift at the pencil's task loss (NLL 1.780) | pencil 0.185; WiSE-FT 0.154 | below WiSE-FT | E2, E3 |
| Dev drift at task NLL 1.803 | diagonal Fisher 0.060 | ≤ 0.060 | E2 (components) |
| English harmful compliance at the pencil's task loss, test | pencil 7.4%; WiSE-FT 5.4% | ≤ 5% | E2 |
| English over-refusal, test | 31.5% | ≤ 20% | E4 |
| English harmful compliance, test | 7.4% | ≤ 8% | E4 |
| English refusal Youden J | 31.0 | ≥ 45 | E4 |
| Ceiling breaks | 2 | 0 | E3 |
| WiSE-FT task optimum | measured s = 0.7 | predicted within ±0.1 | E5 |
| Quantisation drift forecast | not measured | within 2–3× | E6 |
| English harmful compliance after the Hindi fine-tune | 22.5% (no penalty) | ≤ 8% at equal task loss | E9 |

## 13. What is new, and what is not

**Known ingredients.**

- The log-partition's derivatives are the mean and variance of the sufficient statistic (exponential-family calculus).
- Forward-mode JVPs.
- Tangent-space linearisation of fine-tuned models (Ortiz-Jiménez et al., 2023).
- Cross-fitting (Chernozhukov et al., 2018).
- Wachter's law for F-matrices and detection limits for spiked F-ratios (Nadakuditi & Silverstein, 2010; Dharmawansa et al., 2014).
- Gradient mismatch in merging (Daheim et al., 2024).
- The Hanson–Wright inequality.
- Youden's index (Youden, 1950).
- OR-Bench (Cui et al., 2025).
- Learning merge coefficients (AdaMerging; Yang et al., 2024, with an unsupervised entropy objective).
- EWC (Kirkpatrick et al., 2017).

**New, as far as a brief search found (not an exhaustive review).**

1. **The confidence blind spot.** An aligned model's Fisher vanishes as e^(−m) on its confident refusals, so every second-order safety tool under-weights them. Quantified by the sinh(m₀)/m₀ amplification, with an exact JVP-based decomposition of the forecast gap.
2. **Learned safety-merge weights.** Coefficients learned on the exact drift-plus-held-out-task objective and started from the best interpolation: never worse on the fitting objective, no oversized steps, gauge-invariant. Learning merge coefficients is not new; the safety objective, the guarantee and the use as a repair are.
3. **EWC as a Taylor expansion.** EWC is the second-order Taylor expansion of the KL penalty, hence blind on confident refusals; the margin-hinge alternative protects the decision at any confidence.
4. **Cross-fitted curvature.** Shrunk generalised eigenvalues for curvature merges: the angle-free bound on oblique steps (Lemma 2 and its corollary), and the exact WiSE-FT limit under no signal.
5. **The refusal Youden index** as the measure of the discrimination that benign fine-tuning destroys.
6. **When quadratic drift forecasts hold:** for small, spread-out perturbations such as quantisation, but not along a fine-tune.

## References

- Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal* 21(1).
- Cui, Chiang, Stoica & Hsieh (2025). OR-Bench: An Over-Refusal Benchmark for Large Language Models. ICML 2025. arXiv:2405.20947.
- Kirkpatrick et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS* 114(13).
- Daheim, Möllenhoff, Ponti, Gurevych & Khan (2024). Model Merging by Uncertainty-Based Gradient Matching. ICLR 2024. arXiv:2310.12808.
- Dharmawansa, Johnstone & Onatski (2014). Local Asymptotic Normality of the spectrum of high-dimensional spiked F-ratios. arXiv:1411.3875.
- Nadakuditi & Silverstein (2010). Fundamental Limit of Sample Generalized Eigenvalue Based Detection of Signals in Noise Using Relatively Few Signal-Bearing and Noise-Only Samples. *IEEE J. Selected Topics in Signal Processing* 4(3).
- Ortiz-Jiménez, Favero & Frossard (2023). Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models. NeurIPS 2023. arXiv:2305.12827.
- Rudelson & Vershynin (2013). Hanson–Wright inequality and sub-gaussian concentration. *Electron. Commun. Probab.* 18. arXiv:1306.2872.
- Springer et al. (2026). The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety. arXiv:2602.15799.
- Wachter (1980), *Annals of Statistics*; the F-matrix limiting law as stated in "Application of Random Matrix Theory in High-Dimensional Statistics", arXiv:2412.06848 (Theorem 7).
- Youden (1950). Index for rating diagnostic tests. *Cancer* 3(1).
- Matena & Raffel (2022); Martens & Grosse (2015); Wortsman et al. (2022), as in `WRITEUP.md` §16.
- Yang, Wang, Shen, Liu, Guo, Wang & Tao (2024). AdaMerging: Adaptive Model Merging for Multi-Task Learning. ICLR 2024. arXiv:2310.02575.
