"""Pilot experiments for docs/WITH_MORE_CREDITS.md on a controlled toy transformer (CPU, JAX).

Synthetic language. Prompts are 6 random words after a marker:
  GEN prompts      -> respond with the words reversed           (general skill of the aligned model)
  GEN + trigger    -> respond REFUSE                            ("harmful" prompts)
  GEN + lookalike  -> respond with the words reversed           ("benign but risky-looking" prompts)
  TASK prompts     -> respond with each word shifted by one     (the benign fine-tune task, unseen by θ0)
θ0 is trained on the first three; θ1 = θ0 + Δ is a LoRA fine-tune on a small, slightly noisy TASK set only.
No natural-language or harmful text is involved anywhere.

Experiments (numbers printed and saved to results.json):
  T1  forecast-gap decomposition (E1): measured drift vs exact second-order forecast vs log-partition forecast
  T2  repairs (E2): WiSE-FT, diagonal Fisher, convex module merge (linearised), module merge on the true objective
  T4  diffuse vs coherent perturbations (E6): second-order forecast for random / 4-bit perturbations vs the fine-tune
  T5  validation gradient (E5): predicted vs measured task-optimal interpolation point
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

jax.config.update("jax_enable_x64", False)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
FT_MARK = sys.argv[2] if len(sys.argv) > 2 else "TASK"
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- vocabulary and data
PAD, BOS, SEP, EOS, REFUSE, TASK, GEN = range(7)
TRIG = np.arange(7, 11)
LOOK = np.arange(11, 15)
WORDS = np.arange(15, 47)
V, L, K = 48, 16, 6  # vocab, sequence length, prompt words


def make(kind: str, n: int, noise: float = 0.0):
    """Token array (n, L) and loss mask (n, L) marking the target positions (predicting response tokens)."""
    toks = np.full((n, L), PAD, dtype=np.int32)
    w = rng.choice(WORDS, size=(n, K))
    mark = TASK if kind == "task" else GEN
    if kind == "task" and FT_MARK == "GEN":
        mark = GEN
    if kind in ("harm", "risky"):
        pos = rng.integers(0, K, size=n)
        w[np.arange(n), pos] = rng.choice(TRIG if kind == "harm" else LOOK, size=n)
    toks[:, 0], toks[:, 1], toks[:, 2 : 2 + K], toks[:, 2 + K] = BOS, mark, w, SEP
    if kind == "harm":
        resp = np.stack([np.full(n, REFUSE), np.full(n, EOS)], 1)
    elif kind == "task":
        shifted = WORDS[(np.searchsorted(WORDS, w) + 1) % len(WORDS)]
        if noise:
            flip = rng.random(shifted.shape) < noise
            shifted = np.where(flip, rng.choice(WORDS, size=shifted.shape), shifted)
        resp = np.concatenate([shifted, np.full((n, 1), EOS)], 1)
    else:  # gen, risky: reverse
        resp = np.concatenate([w[:, ::-1], np.full((n, 1), EOS)], 1)
    r = resp.shape[1]
    toks[:, 3 + K : 3 + K + r] = resp
    mask = np.zeros((n, L), dtype=np.float32)
    mask[:, 2 + K : 2 + K + r] = 1.0  # logits at position j predict token j+1
    return jnp.asarray(toks), jnp.asarray(mask)


# ---------------------------------------------------------------- model
D, H, NL, F = 64, 4, 2, 256
MODULES = [(f"l{l}", m) for l in range(NL) for m in ("qkv", "o", "fc1", "fc2")]


def init(key):
    ks = iter(jax.random.split(key, 32))
    n = lambda *s, sc=0.02: jax.random.normal(next(ks), s) * sc
    p = {"emb": n(V, D), "pos": n(L, D), "head": n(D, V), "lnf": (jnp.ones(D), jnp.zeros(D))}
    for l in range(NL):
        p[f"l{l}"] = {
            "ln1": (jnp.ones(D), jnp.zeros(D)),
            "ln2": (jnp.ones(D), jnp.zeros(D)),
            "qkv": n(D, 3 * D),
            "o": n(D, D, sc=0.01),
            "fc1": n(D, F),
            "fc2": n(F, D, sc=0.01),
        }
    return p


def ln(x, gb):
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * gb[0] + gb[1]


CAUSAL = jnp.tril(jnp.ones((L, L), bool))


def forward(p, toks):
    x = p["emb"][toks] + p["pos"][None]
    for l in range(NL):
        b = p[f"l{l}"]
        q, k, v = jnp.split(ln(x, b["ln1"]) @ b["qkv"], 3, -1)
        sh = lambda t: t.reshape(t.shape[0], L, H, D // H).transpose(0, 2, 1, 3)
        att = (sh(q) @ sh(k).transpose(0, 1, 3, 2)) / np.sqrt(D // H)
        att = jax.nn.softmax(jnp.where(CAUSAL, att, -1e9), -1)
        y = (att @ sh(v)).transpose(0, 2, 1, 3).reshape(x.shape)
        x = x + y @ b["o"]
        x = x + jax.nn.gelu(ln(x, b["ln2"]) @ b["fc1"]) @ b["fc2"]
    return ln(x, p["lnf"]) @ p["head"]


def targets(toks):
    return jnp.concatenate([toks[:, 1:], jnp.full((toks.shape[0], 1), PAD)], 1)


def nll(p, toks, mask):
    lp = jax.nn.log_softmax(forward(p, toks))
    t = jnp.take_along_axis(lp, targets(toks)[..., None], -1)[..., 0]
    return -(t * mask).sum() / mask.sum()


def adam_train(loss_fn, params, steps, lr, batches):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    @jax.jit
    def step(params, m, v, i, toks, mask):
        lval, g = jax.value_and_grad(loss_fn)(params, toks, mask)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        bc1, bc2 = 1 - 0.9 ** (i + 1), 1 - 0.999 ** (i + 1)
        params = jax.tree_util.tree_map(lambda w, a, b: w - lr * (a / bc1) / (jnp.sqrt(b / bc2) + 1e-8), params, m, v)
        return params, m, v, lval

    for i in range(steps):
        toks, mask = batches(i)
        params, m, v, lval = step(params, m, v, i, toks, mask)
    return params, float(lval)


def get(p, mod):
    return p[mod[0]][mod[1]]


def with_modules(p, new):  # new: {mod: W}
    q = jax.tree_util.tree_map(lambda x: x, p)
    for mod, W in new.items():
        q[mod[0]] = dict(q[mod[0]])
        q[mod[0]][mod[1]] = W
    return q


def tangent(p, delta):  # zeros everywhere except the modules in delta
    t = jax.tree_util.tree_map(jnp.zeros_like, p)
    return with_modules(t, delta)


# ---------------------------------------------------------------- train θ0 and the fine-tune
t0 = time.time()
key = jax.random.PRNGKey(SEED)
mix_cache = {}


def mix_batch(i, bs=128):
    a, b, c = make("gen", bs // 2), make("harm", bs // 4), make("risky", bs // 4)
    return jnp.concatenate([a[0], b[0], c[0]]), jnp.concatenate([a[1], b[1], c[1]])


theta0, l0 = adam_train(nll, init(key), 2500, 3e-3, mix_batch)

ft_toks, ft_mask = make("task", 256, noise=0.15)  # small, slightly noisy benign fine-tune set
R = 8


def lora_init(k):
    ks = jax.random.split(k, len(MODULES))
    out = {}
    for kk, mod in zip(ks, MODULES):
        W = get(theta0, mod)
        out[mod] = (jax.random.normal(kk, (W.shape[0], R)) * 0.02, jnp.zeros((R, W.shape[1])))
    return out


def lora_params_to_model(lp):
    return with_modules(theta0, {mod: get(theta0, mod) + A @ B * (8.0 / R) for mod, (A, B) in lp.items()})


lora_loss = lambda lp, toks, mask: nll(lora_params_to_model(lp), toks, mask)
lp_flat = {f"{a}|{b}": v for (a, b), v in lora_init(jax.random.PRNGKey(SEED + 1)).items()}
lora_loss_flat = lambda lp, toks, mask: lora_loss({tuple(k.split("|")): v for k, v in lp.items()}, toks, mask)


def ft_batch(i, bs=64):
    idx = rng.choice(256, bs, replace=False)
    return ft_toks[idx], ft_mask[idx]


lp_flat, l1 = adam_train(lora_loss_flat, lp_flat, 2000, 5e-3, ft_batch)
DELTA = {tuple(k.split("|")): A @ B * (8.0 / R) for k, (A, B) in lp_flat.items()}
theta1 = with_modules(theta0, {mod: get(theta0, mod) + DELTA[mod] for mod in MODULES})
print(f"trained θ0 (loss {l0:.3f}) and θ1 (ft loss {l1:.3f}) in {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- evaluation sets and metrics
harm_fit, harm_ev = make("harm", 512), make("harm", 512)
risky_ev = make("risky", 512)
task_fit, task_ev = make("task", 512), make("task", 512)
gen_ev = make("gen", 512)


def greedy_answers(p, toks):
    """θ's greedy response, teacher-forcing its own previous tokens (2-token responses for harm prompts)."""
    t = np.array(toks)
    for j in range(2 + K, L - 1):
        nxt = np.array(jnp.argmax(forward(p, jnp.asarray(t))[:, j], -1))
        t[:, j + 1] = nxt
    return jnp.asarray(t)


def answer_mask(toks):
    """Positions predicting the response up to and including the first EOS."""
    t = np.array(toks)
    m = np.zeros(t.shape, np.float32)
    for i in range(t.shape[0]):
        for j in range(2 + K, L - 1):
            m[i, j] = 1
            if t[i, j + 1] == EOS:
                break
    return jnp.asarray(m)


H_FIT = greedy_answers(theta0, harm_fit[0])
H_FIT_M = answer_mask(H_FIT)
H_EV = greedy_answers(theta0, harm_ev[0])
H_EV_M = answer_mask(H_EV)


@jax.jit
def drift_tokens(p, toks):
    """Per-position KL(p_θ0 || p_θ) on θ0's own answers."""
    f0, f = forward(theta0, toks), forward(p, toks)
    lp0, lp = jax.nn.log_softmax(f0), jax.nn.log_softmax(f)
    return (jnp.exp(lp0) * (lp0 - lp)).sum(-1)


def drift(p, toks=H_EV, mask=H_EV_M):
    return float((drift_tokens(p, toks) * mask).sum() / mask.sum())


def first_token_rate(p, toks, token):
    nxt = np.array(jnp.argmax(forward(p, toks)[:, 2 + K], -1))
    return float((nxt == token).mean())


def task_acc(p, toks):
    pred = np.array(jnp.argmax(forward(p, toks), -1))
    tgt = np.array(targets(toks))
    m = np.array(toks_mask_task)
    return float(((pred == tgt) * m).sum() / m.sum())


toks_mask_task = task_ev[1]


def report(p):
    ref_h = first_token_rate(p, harm_ev[0], REFUSE)
    ref_r = first_token_rate(p, risky_ev[0], REFUSE)
    return {
        "drift": drift(p),
        "task_nll": float(nll(p, *task_ev)),
        "task_acc": task_acc(p, task_ev[0]),
        "harmful_compliance": 1 - ref_h,
        "over_refusal": ref_r,
        "youden_J": ref_h - ref_r,
        "gen_nll": float(nll(p, *gen_ev)),
    }


res = {"seed": SEED, "ft_mark": FT_MARK, "theta0": report(theta0), "theta1": report(theta1)}
print("θ0", {k: round(v, 3) for k, v in res["theta0"].items()})
print("θ1", {k: round(v, 3) for k, v in res["theta1"].items()}, flush=True)

# ---------------------------------------------------------------- T1: forecast-gap decomposition
f0_fn = lambda p: forward(p, H_EV)
f0, U = jax.jvp(f0_fn, (theta0,), (tangent(theta0, DELTA),))
p0 = jax.nn.softmax(f0)
var_u = (p0 * U**2).sum(-1) - ((p0 * U).sum(-1)) ** 2
M = H_EV_M
q = float((var_u * M).sum() / M.sum())  # exact Δᵀ F_h Δ (per-token mean)


def s_lin(t):
    A = jax.nn.logsumexp(f0 + t * U, -1) - jax.nn.logsumexp(f0, -1) - t * (p0 * U).sum(-1)
    return float((A * M).sum() / M.sum()), A


ts = [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1.0]
curve = []
for t in ts:
    pt = with_modules(theta0, {mod: get(theta0, mod) + t * DELTA[mod] for mod in MODULES})
    curve.append({"t": t, "measured": drift(pt), "quadratic": 0.5 * q * t * t, "loglinear": s_lin(t)[0]})
S1, Slin1 = curve[-1]["measured"], curve[-1]["loglinear"]
# per-token: margin of θ0's answer token and the per-token gap at t = 1
tok_kl1 = np.array(drift_tokens(theta1, H_EV))[np.array(M) > 0]
tok_quad = 0.5 * np.array(var_u)[np.array(M) > 0]
f0n = np.array(f0)[np.array(M) > 0]
srt = np.sort(f0n, -1)
margin = srt[:, -1] - srt[:, -2]
big = tok_kl1 > 0.05
tok_slin1 = np.array(s_lin(1.0)[1])[np.array(M) > 0]
loc_exp = [
    float(np.log(curve[i + 1]["measured"] / curve[i]["measured"]) / np.log(ts[i + 1] / ts[i]))
    for i in range(len(ts) - 1)
]
res["T1"] = {
    "curve": curve,
    "q": q,
    "total_gap": S1 / (0.5 * q),
    "amplification": Slin1 / (0.5 * q),
    "nonlinearity": S1 / Slin1,
    "local_exponent": loc_exp,
    "median_margin": float(np.median(margin)),
    "share_of_drift_in_tokens_with_kl_gt_0.05": float(tok_kl1[big].sum() / tok_kl1.sum()),
    "spearman_tokengap_vs_margin": float(spearmanr(tok_kl1[big] / np.maximum(tok_quad[big], 1e-12), margin[big])[0])
    if big.sum() > 5
    else None,
    "spearman_loglinear_amplification_vs_margin": float(spearmanr(tok_slin1 / np.maximum(tok_quad, 1e-12), margin)[0]),
    "flipped_share": float(
        (
            np.array(jnp.argmax(forward(theta1, H_EV), -1))[np.array(M) > 0]
            != srt.argmax(-1) * 0 + np.array(jnp.argmax(f0, -1))[np.array(M) > 0]
        ).mean()
    ),
    "median_sinh_m_over_m_for_big": float(np.median(np.sinh(np.minimum(margin[big], 30)) / margin[big]))
    if big.sum()
    else None,
    "median_tokengap_for_big": float(np.median(tok_kl1[big] / np.maximum(tok_quad[big], 1e-12))) if big.sum() else None,
}
print("T1", json.dumps({k: v for k, v in res["T1"].items() if k != "curve"}, default=float), flush=True)


# ---------------------------------------------------------------- T4: diffuse vs coherent perturbations
def quad_forecast(eps):
    _, Ue = jax.jvp(f0_fn, (theta0,), (tangent(theta0, eps),))
    ve = (p0 * Ue**2).sum(-1) - ((p0 * Ue).sum(-1)) ** 2
    return 0.5 * float((ve * M).sum() / M.sum())


def perturbed(eps):
    return with_modules(theta0, {mod: get(theta0, mod) + eps[mod] for mod in eps})


T4 = {"coherent_finetune": {"measured": S1, "quadratic": 0.5 * q}}
k4 = jax.random.PRNGKey(SEED + 7)
dn = {m: jnp.linalg.norm(DELTA[m]) for m in MODULES}
sweep = []
for r in (0.01, 0.03, 0.1, 0.3, 1.0):
    coh = {m: r * DELTA[m] for m in MODULES}
    rnd = {}
    for i, mod in enumerate(MODULES):
        z = jax.random.normal(jax.random.fold_in(k4, i), DELTA[mod].shape)
        rnd[mod] = z / jnp.linalg.norm(z) * dn[mod] * r
    row = {"rel_norm": r}
    for name, eps in (("coherent", coh), ("random", rnd)):
        meas, quad = drift(perturbed(eps)), quad_forecast(eps)
        row[name] = {"measured": meas, "quadratic": quad, "ratio": meas / max(quad, 1e-12)}
    sweep.append(row)
T4["norm_sweep"] = sweep
for bits in (8, 4, 3):
    eps = {}
    for mod in MODULES:
        W = get(theta0, mod)
        step = 2 * jnp.abs(W).max() / (2**bits - 1)
        eps[mod] = jnp.round(W / step) * step - W
    T4[f"quantised_{bits}bit"] = {
        "measured": drift(perturbed(eps)),
        "quadratic": quad_forecast(eps),
        "rel_norm_vs_delta": float(
            sum(jnp.linalg.norm(eps[m]) ** 2 for m in MODULES) ** 0.5
            / sum(jnp.linalg.norm(DELTA[m]) ** 2 for m in MODULES) ** 0.5
        ),
    }
for k, v in T4.items():
    if k != "norm_sweep":
        v["measured_over_quadratic"] = v["measured"] / max(v["quadratic"], 1e-12)
res["T4"] = T4
print(
    "T4",
    json.dumps({k: round(v["measured_over_quadratic"], 2) for k, v in T4.items() if k != "norm_sweep"}),
    [(r["rel_norm"], round(r["coherent"]["ratio"], 2), round(r["random"]["ratio"], 2)) for r in sweep],
    flush=True,
)

# ---------------------------------------------------------------- T5: validation gradient predicts the task optimum
tf_toks, tf_mask = task_fit
g1, Wd = jax.jvp(lambda p: forward(p, tf_toks), (theta1,), (tangent(theta1, DELTA),))
p1 = jax.nn.softmax(g1)
y = targets(tf_toks)
gdelta = float(
    (((p1 * Wd).sum(-1) - jnp.take_along_axis(Wd, y[..., None], -1)[..., 0]) * tf_mask).sum() / tf_mask.sum()
)
dHd = float((((p1 * Wd**2).sum(-1) - (p1 * Wd).sum(-1) ** 2) * tf_mask).sum() / tf_mask.sum())
s_star_pred = 1 - gdelta / dHd
grid = np.linspace(0, 1, 41)
task_curve = [
    float(nll(with_modules(theta0, {m: get(theta0, m) + s * DELTA[m] for m in MODULES}), *task_ev)) for s in grid
]
res["T5"] = {
    "s_star_predicted": s_star_pred,
    "s_star_measured": float(grid[int(np.argmin(task_curve))]),
    "gdelta": gdelta,
    "dHd": dHd,
    "train_ft_loss": l1,
}
print("T5", res["T5"], flush=True)


# ---------------------------------------------------------------- T2: repairs
def model_s(s):
    return with_modules(theta0, {m: get(theta0, m) + float(s[i]) * DELTA[m] for i, m in enumerate(MODULES)})


wise = [{"s": float(s), **report(model_s(np.full(len(MODULES), s)))} for s in np.linspace(0, 1, 21)]

# per-module JVPs: harm at θ0 (fit split), task at θ1 (fit split)
Hf, Hm = H_FIT, H_FIT_M
fh0 = forward(theta0, Hf)
ph0 = jax.nn.softmax(fh0)
Uh = jnp.stack(
    [jax.jvp(lambda p: forward(p, Hf), (theta0,), (tangent(theta0, {m: DELTA[m]}),))[1] for m in MODULES], -1
)
gt1 = forward(theta1, tf_toks)
Wt = jnp.stack(
    [jax.jvp(lambda p: forward(p, tf_toks), (theta1,), (tangent(theta1, {m: DELTA[m]}),))[1] for m in MODULES], -1
)
EUh = (ph0[..., None] * Uh).sum(-2)


@jax.jit
def lin_obj(s, mu):
    f = fh0 + Uh @ s
    Sh = ((jax.nn.logsumexp(f, -1) - jax.nn.logsumexp(fh0, -1) - EUh @ s) * Hm).sum() / Hm.sum()
    g = gt1 + Wt @ (s - 1)
    Lt = ((jax.nn.logsumexp(g, -1) - jnp.take_along_axis(g, y[..., None], -1)[..., 0]) * tf_mask).sum() / tf_mask.sum()
    return Sh + mu * Lt


lin_grad = jax.jit(jax.grad(lin_obj))


def true_obj(s, mu):
    p = with_modules(theta0, {m: get(theta0, m) + s[i] * DELTA[m] for i, m in enumerate(MODULES)})
    Sh = (drift_tokens(p, Hf) * Hm).sum() / Hm.sum()
    return Sh + mu * nll(p, tf_toks, tf_mask)


true_vg = jax.jit(jax.value_and_grad(true_obj))


def solve(fun, grad, mu, x0=None):
    x0 = np.full(len(MODULES), 0.5) if x0 is None else x0
    r = minimize(
        lambda s: float(fun(jnp.asarray(s, jnp.float32), mu)),
        x0,
        jac=lambda s: np.asarray(grad(jnp.asarray(s, jnp.float32), mu), np.float64),
        bounds=[(0, 1)] * len(MODULES),
        method="L-BFGS-B",
    )
    return r.x


mus = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30]
convex, oracle = [], []
for mu in mus:
    s = solve(lin_obj, lin_grad, mu)
    convex.append({"mu": mu, "s": s.round(3).tolist(), "predicted_drift": None, **report(model_s(s))})
    cs = np.linspace(0, 1, 21)
    fit_vals = [float(true_vg(jnp.full(len(MODULES), c, jnp.float32), mu)[0]) for c in cs]
    c_best = cs[int(np.argmin(fit_vals))]
    s2 = solve(lambda s, mu: true_vg(s, mu)[0], lambda s, mu: true_vg(s, mu)[1], mu, x0=np.full(len(MODULES), c_best))
    oracle.append(
        {
            "mu": mu,
            "s": s2.round(3).tolist(),
            "wise_start": float(c_best),
            "fit_obj_wise": min(fit_vals),
            "fit_obj_final": float(true_vg(jnp.asarray(s2, jnp.float32), mu)[0]),
            **report(model_s(s2)),
        }
    )


# diagonal Fisher (sampled labels) of drift at θ0 and of the task at θ1
def diag_fisher(p, toks, mask, key, nsamp=2):
    def lp_sampled(params, tk, mk, k):
        f = forward(params, tk[None])[0]
        yy = jax.random.categorical(k, f)
        return (jnp.take_along_axis(jax.nn.log_softmax(f), yy[:, None], -1)[:, 0] * mk).sum()

    acc = {m: jnp.zeros_like(get(p, m)) for m in MODULES}
    keys = jax.random.split(key, toks.shape[0] * nsamp)
    gfn = jax.jit(jax.vmap(jax.grad(lp_sampled), in_axes=(None, 0, 0, 0)))
    n = 0
    for r in range(nsamp):
        for c in range(0, toks.shape[0], 128):
            g = gfn(p, toks[c : c + 128], mask[c : c + 128], keys[r * toks.shape[0] + c : r * toks.shape[0] + c + 128])
            for m in MODULES:
                acc[m] = acc[m] + (get(g, m) ** 2).sum(0)
            n += g["emb"].shape[0]
    return {m: acc[m] / n for m in MODULES}


Fh = diag_fisher(theta0, Hf, Hm, jax.random.PRNGKey(11))
Ft = diag_fisher(theta1, tf_toks, tf_mask, jax.random.PRNGKey(12))
diagf = []
for mu in [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100, 1000]:
    new = {m: get(theta0, m) + mu * Ft[m] / (Fh[m] + mu * Ft[m] + 1e-20) * DELTA[m] for m in MODULES}
    diagf.append({"mu": mu, **report(with_modules(theta0, new))})

res["T2"] = {"wise": wise, "convex_linearised": convex, "module_true_objective": oracle, "diag_fisher": diagf}


def at_budget(rows, budget):
    ok = [r for r in rows if r["task_nll"] <= budget]
    return min(ok, key=lambda r: r["drift"]) if ok else None


L0, L1 = res["theta0"]["task_nll"], res["theta1"]["task_nll"]
summary = {}
for frac in (0.9, 0.75, 0.5):
    b = L1 + (1 - frac) * (L0 - L1)
    summary[f"keep_{int(frac * 100)}pct_task_gain"] = {
        name: (
            lambda r: (
                None
                if r is None
                else {
                    k: round(r[k], 4) for k in ("drift", "task_nll", "harmful_compliance", "over_refusal", "youden_J")
                }
            )
        )(at_budget(rows, b))
        for name, rows in [
            ("wise", wise),
            ("convex_linearised", convex),
            ("module_true_objective", oracle),
            ("diag_fisher", diagf),
        ]
    }
res["T2_summary"] = summary
print("T2", json.dumps(summary, indent=1), flush=True)
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
(OUT / f"toy_seed{SEED}.json").write_text(json.dumps(res, indent=1, default=float))
print(f"done in {time.time() - t0:.0f}s")
