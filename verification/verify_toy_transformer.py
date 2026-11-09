"""End-to-end verification of the Safety Pencil maths on a real (tiny) transformer.

The toy model copies the Tiny Aya features the maths depends on: grouped-query attention
(4 query heads sharing 2 KV heads), a parallel attention+MLP block reading one mean-subtracting
LayerNorm, a gated SiLU MLP and causal masking. Everything is float64 and every expectation over
next tokens is computed exactly by enumerating the vocabulary, so identities are checked to
machine precision and approximations are measured against the exact answer.

Setup: theta_base (random) -> theta_0 ("aligned": refuses harmful prompts with a fixed 4-token
refusal, and does a copy task) -> theta_1 (fine-tuned on the copy task only, which erodes refusal).
Only the 14 linear modules are trained (3840 parameters), like LoRA targets.

Run:  python verify_toy_transformer.py      (CPU, about 25 minutes, 7 GB RAM; needs jax, numpy, scipy)
Writes verify_results.json next to this file.
"""
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sl
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

V, D, NQ, NKV, HD, I, NL, T, R = 40, 16, 4, 2, 4, 24, 2, 10, 4
REP = NQ // NKV
P = np.arange(T - R - 1, T - 1)          # logit positions that predict the response tokens
REFUSAL = np.array([0, 1, 2, 3])
SHAPES = {"q": (NQ * HD, D), "k": (NKV * HD, D), "v": (NKV * HD, D), "o": (D, NQ * HD),
          "g": (I, D), "u": (I, D), "dn": (D, I)}
RESULTS = {}


def record(name, status, **info):
    clean = {k: (float(v) if isinstance(v, (np.floating, float, np.integer)) else v) for k, v in info.items()}
    RESULTS[name] = {"status": status, **clean}
    print(f"[{status:6}] {name}: " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in clean.items()))


# ================================================================ model
def init(key):
    ks = iter(jax.random.split(key, 64))
    fixed = {"emb": jax.random.normal(next(ks), (V, D)) * 0.5,
             "pos": jax.random.normal(next(ks), (T, D)) * 0.5,
             "head": jax.random.normal(next(ks), (V, D)) / np.sqrt(D)}
    tgt = {f"{m}{l}": jax.random.normal(next(ks), s) / np.sqrt(s[1]) for l in range(NL) for m, s in SHAPES.items()}
    return fixed, tgt


def ln(x):                                   # mean-subtracting LayerNorm without affine terms, like Cohere's
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + 1e-5)


def forward(fixed, tgt, tokens, eps=None):
    """Logits (T, V) and taps {module: input activations (T, d_in)}.
    eps[module] (T, d_out) is added to each linear output, so d loss / d eps = g."""
    taps = {}
    Tn = tokens.shape[0]

    def lin(name, a):
        taps[name] = a
        y = a @ tgt[name].T
        return y if eps is None else y + eps[name]

    h = fixed["emb"][tokens] + fixed["pos"][:Tn]
    mask = jnp.tril(jnp.ones((Tn, Tn), bool))
    for l in range(NL):
        x = ln(h)
        q = lin(f"q{l}", x).reshape(Tn, NQ, HD)
        k = jnp.repeat(lin(f"k{l}", x).reshape(Tn, NKV, HD), REP, axis=1)    # query head h reads KV h // REP
        v = jnp.repeat(lin(f"v{l}", x).reshape(Tn, NKV, HD), REP, axis=1)
        s = jnp.where(mask[None], jnp.einsum("thd,shd->hts", q, k) / np.sqrt(HD), -1e30)
        att = jnp.einsum("hts,shd->thd", jax.nn.softmax(s, -1), v).reshape(Tn, NQ * HD)
        mlp = lin(f"dn{l}", jax.nn.silu(lin(f"g{l}", x)) * lin(f"u{l}", x))
        h = h + lin(f"o{l}", att) + mlp                                      # parallel block
    return ln(h) @ fixed["head"].T, taps


def logp_P(fixed, tgt, tokens, eps=None):
    return jax.nn.log_softmax(forward(fixed, tgt, tokens, eps)[0][P], -1)    # (|P|, V)


# ================================================================ data and training
rng = np.random.default_rng(0)


def harmful_prompts(n):
    return rng.integers(4, 22, (n, T - R))


def task_seqs(n):                            # copy task: the response repeats the first R prompt tokens
    pr = rng.integers(22, 40, (n, T - R))
    return np.concatenate([pr, pr[:, :R]], 1)


def ce(fixed, tgt, seqs):
    def one(tok):
        return -logp_P(fixed, tgt, tok)[jnp.arange(R), tok[P + 1]].mean()
    return jax.vmap(one)(seqs).mean()


def greedy(fixed, tgt, prompts):
    seqs = np.concatenate([prompts, np.zeros((len(prompts), R), int)], 1)
    for j in range(R):
        lg = jax.vmap(lambda s: forward(fixed, tgt, s)[0])(jnp.array(seqs))
        seqs[:, T - R + j] = np.asarray(lg[:, T - R + j - 1].argmax(-1))
    return seqs


def adam(loss, params, steps, lr):
    g_fn = jax.jit(jax.grad(loss))
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    for t in range(1, steps + 1):
        g = g_fn(params)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        params = jax.tree_util.tree_map(
            lambda p, a, b: p - lr * (a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8), params, m, v)
    return params


t0 = time.time()
fixed, tgt_base = init(jax.random.PRNGKey(0))
mix = jnp.array(np.concatenate([np.concatenate([harmful_prompts(64), np.tile(REFUSAL, (64, 1))], 1), task_seqs(16)]))
tgt0 = adam(lambda p: ce(fixed, p, mix), tgt_base, 800, 3e-3)              # "alignment"
H_fac, H_eval = harmful_prompts(32), harmful_prompts(64)
Hs_fac = jnp.array(greedy(fixed, tgt0, H_fac))                              # theta_0's own responses, teacher-forced
T_fac, T_eval, T_ft = jnp.array(task_seqs(32)), jnp.array(task_seqs(64)), jnp.array(task_seqs(128))
tgt1 = adam(lambda p: ce(fixed, p, T_ft), tgt0, 300, 2e-3)                  # benign fine-tune

flat0, unravel = ravel_pytree(tgt0)
flat1 = ravel_pytree(tgt1)[0]
DELTA = np.asarray(flat1 - flat0)
NPAR = flat0.size


def refusal_rate(tgt, prompts=H_eval):
    return float((greedy(fixed, tgt, prompts)[:, T - R:] == REFUSAL).all(1).mean())


def task_acc(tgt, seqs=T_eval):
    lg = jax.vmap(lambda s: forward(fixed, tgt, s)[0])(seqs)
    return float((lg[:, P].argmax(-1) == seqs[:, P + 1]).mean())


print(f"models built in {time.time() - t0:.0f}s; trainable parameters: {NPAR}")
print(f"refusal theta0={refusal_rate(tgt0):.2f} theta1={refusal_rate(tgt1):.2f}; "
      f"copy-task acc theta0={task_acc(tgt0):.2f} theta1={task_acc(tgt1):.2f}")

P0 = jax.vmap(lambda s: jnp.exp(logp_P(fixed, tgt0, s)))(Hs_fac)


@jax.jit
def S_flat(flat):
    lp = jax.vmap(lambda s: logp_P(fixed, unravel(flat), s))(Hs_fac)
    return (P0 * (jnp.log(P0) - lp)).sum(-1).mean()


@jax.jit
def Lt_flat(flat):
    return ce(fixed, unravel(flat), T_fac)


# ================================================================ exact curvature
def fisher_exact(flat, seqs):
    """Mean per-loss-token Fisher, exact over the vocabulary (no sampling)."""
    @jax.jit
    def per(tok):
        f = lambda w: logp_P(fixed, unravel(w), tok)
        J = jax.jacrev(f)(flat)
        return jnp.einsum("pv,pvi,pvj->ij", jnp.exp(f(flat)), J, J)
    return sum(np.asarray(per(s)) for s in seqs) / (len(seqs) * len(P))


def ggn(flat, seqs):
    @jax.jit
    def per(tok):
        f = lambda w: forward(fixed, unravel(w), tok)[0][P]
        J = jax.jacrev(f)(flat)
        p = jax.nn.softmax(f(flat), -1)
        Hz = jax.vmap(lambda q: jnp.diag(q) - jnp.outer(q, q))(p)
        return jnp.einsum("pvi,pvw,pwj->ij", J, Hz, J)
    return sum(np.asarray(per(s)) for s in seqs) / (len(seqs) * len(P))


def hessian_chunked(f, x, chunk=256):
    hvp = jax.jit(jax.vmap(lambda v: jax.jvp(jax.grad(f), (x,), (v,))[1]))
    n = x.size
    out = np.zeros((n, n))
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        E = jnp.zeros((m, n)).at[jnp.arange(m), jnp.arange(i, i + m)].set(1.0)
        out[i:i + m] = np.asarray(hvp(E))
    return 0.5 * (out + out.T)


t0 = time.time()
F_h, F_t = fisher_exact(flat0, Hs_fac), fisher_exact(flat1, T_fac)
G_t = ggn(flat1, T_fac)
Hess_S, Hess_Lt = hessian_chunked(S_flat, flat0), hessian_chunked(Lt_flat, flat1)
print(f"exact curvature in {time.time() - t0:.0f}s")

err = np.linalg.norm(Hess_S - F_h) / np.linalg.norm(F_h)
record("V1 Hessian of S at theta0 equals the sampled-label Fisher F_h (Lemma 2.1)", "PASS" if err < 1e-8 else "FAIL",
       rel_error=err, grad_norm_at_theta0=float(np.linalg.norm(jax.grad(S_flat)(flat0))))
err = np.linalg.norm(G_t - F_t) / np.linalg.norm(F_t)
record("V2 Gauss-Newton equals the model Fisher on task data (Lemma 3.1)", "PASS" if err < 1e-8 else "FAIL", rel_error=err)
record("V3 Full task Hessian vs Gauss-Newton at theta1 (size of the ignored residual term)", "REPORT",
       rel_difference=np.linalg.norm(Hess_Lt - G_t) / np.linalg.norm(Hess_Lt),
       min_eig_full_hessian=float(np.linalg.eigvalsh(Hess_Lt)[0]), task_loss_theta1=float(Lt_flat(flat1)))

# exact symmetries: shared null space of F_h and F_t (LayerNorm shift, softmax shift, V/O gauge ...)
w_sum = np.linalg.eigvalsh(F_h + F_t)
null_dim = int((w_sum < 1e-10 * w_sum.max()).sum())
record("V3b Exact null directions shared by F_h and F_t (functionally irrelevant)", "REPORT",
       null_dimension=null_dim, of_total=NPAR)

# ================================================================ Propositions 1-2 with exact matrices on the network
def restrict(Fh, Ht, rel_tol=1e-10):
    """Orthonormal basis of range(Fh + Ht): the function-relevant subspace."""
    w, U = np.linalg.eigh(Fh + Ht)
    return U[:, w > rel_tol * w.max()]


def exact_repair(Delta, Fh, Ht, mu, damping="none", gamma=0.0):
    U = restrict(Fh, Ht)
    Fr, Hr, dr = U.T @ Fh @ U, U.T @ Ht @ U, U.T @ Delta
    if damping == "euclidean":
        Hr = Hr + gamma * np.trace(Hr) / len(Hr) * np.eye(len(Hr))
    elif damping == "covariant":
        Hr = Hr + gamma * Fr
    return -U @ np.linalg.solve(Fr + mu * Hr, Fr @ dr)


quad = 0.5 * DELTA @ F_h @ DELTA
ratios = {e: float(S_flat(flat0 + e * DELTA)) / (e ** 2 * quad) for e in (1.0, 0.3, 0.1, 0.03, 0.01)}
dev = [abs(ratios[e] - 1) for e in (0.3, 0.1, 0.03, 0.01)]
record("V4 Prop 1: S(theta0 + e*Delta) / (0.5 e^2 Delta'F_h Delta) -> 1 (error shrinks like e)",
       "PASS" if dev[-1] < 0.02 and all(a > b for a, b in zip(dev, dev[1:])) else "FAIL",
       **{f"ratio_e{e}": r for e, r in ratios.items()})

rows = []
for scale in (0.01, 0.1, 1.0):
    Dl = scale * DELTA
    th = flat0 + Dl
    gl = np.asarray(jax.grad(Lt_flat)(th))
    Hl = G_t if scale == 1.0 else ggn(th, T_fac)
    Hx = Hess_Lt if scale == 1.0 else hessian_chunked(Lt_flat, th)
    S_b = float(S_flat(th))
    for damping, gamma in (("none", 0.0), ("euclidean", 1e-2), ("euclidean", 1e-1), ("covariant", 1e-1)):
        for mu in (0.03, 0.3, 3.0):
            dl = exact_repair(Dl, F_h, Hl, mu, damping, gamma)
            S_a, S_p = float(S_flat(th + dl)), 0.5 * (Dl + dl) @ F_h @ (Dl + dl)
            R_a = float(Lt_flat(th + dl) - Lt_flat(th))
            rows.append({"scale": scale, "damping": f"{damping} {gamma:g}", "mu": mu, "S_before": S_b,
                         "S_after_actual": S_a, "S_after_pred": float(S_p), "R_actual": R_a,
                         "R_pred_ggn": float(0.5 * dl @ Hl @ dl), "R_pred_fullhess_with_grad": float(0.5 * dl @ Hx @ dl + gl @ dl),
                         "step_over_update_norm": float(np.linalg.norm(dl) / np.linalg.norm(Dl))})
RESULTS["V5 repair table (exact matrices)"] = rows
sm = [r for r in rows if r["scale"] == 0.01 and r["damping"] == "none 0"]
s_err = max(abs(r["S_after_actual"] - r["S_after_pred"]) / r["S_before"] for r in sm)
r_err = max(abs(r["R_actual"] - r["R_pred_fullhess_with_grad"]) / max(abs(r["R_actual"]), 1e-12) for r in sm)
record("V5 Prop 2: predicted S and R after exact repair match the network (update x0.01, no damping)",
       "PASS" if s_err < 0.05 and r_err < 0.15 else "FAIL", S_err_rel_to_S_before=s_err, R_rel_err=r_err)
for scale in (0.1, 1.0):
    for dmp in ("none 0", "euclidean 0.01", "euclidean 0.1", "covariant 0.1"):
        r = [x for x in rows if x["scale"] == scale and x["damping"] == dmp and x["mu"] == 0.3][0]
        record(f"V5b Exact repair at update x{scale}, damping {dmp}, mu=0.3", "REPORT",
               S_before=r["S_before"], S_after_actual=r["S_after_actual"], S_after_pred=r["S_after_pred"],
               task_regret_actual=r["R_actual"], step_norm_over_update_norm=r["step_over_update_norm"])

# frontier slope on the network: finite differences of the actual S and R along mu, small update
mus = np.array([0.3, 0.31])
Dl = 0.1 * DELTA
Hl = ggn(flat0 + Dl, T_fac)
SR = []
for m in mus:
    dl = exact_repair(Dl, F_h, Hl, m)
    SR.append((0.5 * (Dl + dl) @ F_h @ (Dl + dl), 0.5 * dl @ Hl @ dl))
slope = (SR[1][0] - SR[0][0]) / (SR[1][1] - SR[0][1])
record("V6 Frontier slope dS/dR = -mu on the network's quadratic model", "PASS" if abs(slope + mus.mean()) / mus.mean() < 0.02 else "FAIL",
       slope=slope, minus_mu=-mus.mean())

# ================================================================ K-FAC: estimator, normalisation, accuracy
OFF, start = {}, 0
for k in sorted(tgt0):
    n = int(np.prod(tgt0[k].shape)); OFF[k] = slice(start, start + n); start += n


def kfac_exact(tgt, seqs, rows_mode):
    """Exact-expectation K-FAC factors. rows_mode: 'all' (every position), 'upto' (positions up to the
    last loss position), 'loss' (loss positions only). A is a mean over the rows used; G is a sum over
    the rows used divided by the number of loss tokens (K-FAC-expand normalisation)."""
    rows = {"all": np.arange(T), "upto": np.arange(P.max() + 1), "loss": P}[rows_mode]
    zeros = {k: jnp.zeros((T, tgt[k].shape[0])) for k in tgt}

    @jax.jit
    def per(tok):
        _, taps = forward(fixed, tgt, tok)
        Jg = jax.jacrev(lambda e: logp_P(fixed, tgt, tok, e))(zeros)      # {k: (|P|, V, T, d_out)}
        p = jnp.exp(logp_P(fixed, tgt, tok))
        return ({k: taps[k][rows].T @ taps[k][rows] for k in tgt},
                {k: jnp.einsum("sv,svtd,svte->de", p, Jg[k][:, :, rows], Jg[k][:, :, rows]) for k in tgt})

    A = {k: 0.0 for k in tgt}; G = {k: 0.0 for k in tgt}
    for tok in seqs:
        a, g = per(tok)
        for k in tgt:
            A[k] = A[k] + np.asarray(a[k]); G[k] = G[k] + np.asarray(g[k])
    return ({k: A[k] / (len(seqs) * len(rows)) for k in tgt}, {k: G[k] / (len(seqs) * len(P)) for k in tgt})


def tok_grads(tgt, seqs, k):
    """Exact per-position gradients of module k for every (sequence, loss position, label), each
    weighted by sqrt(p(label)), so sums of outer products are exact expectations. Shape (rows, d_out, d_in)."""
    zeros = {kk: jnp.zeros((T, tgt[kk].shape[0])) for kk in tgt}

    @jax.jit
    def per(tok):
        _, taps = forward(fixed, tgt, tok)
        Jg = jax.jacrev(lambda e: logp_P(fixed, tgt, tok, e))(zeros)[k]
        sp = jnp.sqrt(jnp.exp(logp_P(fixed, tgt, tok)))
        return jnp.einsum("sv,svtd,ti->svtdi", sp, Jg, taps[k])
    return np.concatenate([np.asarray(per(t)).reshape(-1, *tgt[k].shape) for t in seqs])


t0 = time.time()
fac = {m: (kfac_exact(tgt0, Hs_fac, m), kfac_exact(tgt1, T_fac, m)) for m in ("all", "upto", "loss")}
print(f"K-FAC factors in {time.time() - t0:.0f}s")

# V7 unbiased single-backward estimator
tok = Hs_fac[0]
zeros = {k: jnp.zeros((T, tgt0[k].shape[0])) for k in tgt0}
p_tok = jnp.exp(logp_P(fixed, tgt0, tok))


@jax.jit
def sampled_G(key):
    ys = jax.random.categorical(key, jnp.log(p_tok), axis=-1)
    g = jax.grad(lambda e: logp_P(fixed, tgt0, tok, e)[jnp.arange(R), ys].sum())(zeros)
    return {k: g[k].T @ g[k] for k in ("v0", "dn1")}


Jg = jax.jacrev(lambda e: logp_P(fixed, tgt0, tok, e))(zeros)
exactG = {k: np.asarray(jnp.einsum("sv,svtd,svte->de", p_tok, Jg[k], Jg[k])) for k in ("v0", "dn1")}
errs = []
for n in (100, 1000, 10000):
    est = jax.vmap(sampled_G)(jax.random.split(jax.random.PRNGKey(n), n))
    errs.append(max(np.linalg.norm(np.asarray(est[k]).mean(0) - exactG[k]) / np.linalg.norm(exactG[k]) for k in exactG))
record("V7 One backward pass of summed sampled labels is unbiased for the G factor",
       "PASS" if errs[2] < 0.05 and errs[2] < errs[0] / 3 else "FAIL", rel_err_n100=errs[0], rel_err_n1000=errs[1], rel_err_n10000=errs[2])

# V8 accuracy: exact block vs (a) expand without factorisation, (b) K-FAC variants, (c) eigenvalue-corrected pencil basis
acc = {}
for k in sorted(tgt0):
    s_ = OFF[k]
    Fb, d = F_h[s_, s_], DELTA[s_]
    exact_q = d @ Fb @ d
    # expand without factorisation: sum over positions of per-position outer products
    gt = tok_grads(tgt0, Hs_fac, k).reshape(-1, Fb.shape[0])
    Fexp = gt.T @ gt / (len(Hs_fac) * len(P))
    del gt
    row = {"expand_no_factorisation_err": np.linalg.norm(Fb - Fexp) / np.linalg.norm(Fb)}
    for m in ("all", "upto", "loss"):
        (Ah, Gh), _ = fac[m]
        K = np.kron(Gh[k], Ah[k])                      # row-major vec(g a^T) = g (x) a
        row[f"kfac_{m}_err"] = np.linalg.norm(Fb - K) / np.linalg.norm(Fb)
        row[f"kfac_{m}_quad_ratio"] = (d @ K @ d) / exact_q
    acc[k] = {kk: float(vv) for kk, vv in row.items()}
RESULTS["V8 per-module K-FAC accuracy (harmful side)"] = acc
med = {m: float(np.median([acc[k][f"kfac_{m}_err"] for k in acc])) for m in ("all", "upto", "loss")}
best = min(med, key=med.get)
record("V8 K-FAC normalisation variants: median relative Frobenius error vs exact block", "REPORT",
       expand_no_factorisation=float(np.median([acc[k]["expand_no_factorisation_err"] for k in acc])),
       all_positions=med["all"], up_to_last_loss=med["upto"], loss_positions_only=med["loss"], best=best)
qr = {m: [acc[k][f"kfac_{m}_quad_ratio"] for k in acc] for m in ("all", "upto", "loss")}
record("V8b K-FAC quadratic form on the actual update / exact, per module", "REPORT",
       **{f"{m}_median": float(np.median(v)) for m, v in qr.items()},
       **{f"{m}_range": f"{min(v):.2f}-{max(v):.2f}" for m, v in qr.items()})

# ================================================================ pencil construction with null-space handling and damping
def gen_eig_range(Fh, Ft, damping, gamma):
    """Generalised eigenproblem restricted to range(Fh + Ft). damping: 'covariant' (Ft + gamma*Fh),
    'euclidean' (Ft + gamma*mean(diag)*I), 'none'. Returns lam, X (columns Ft-orthonormal), damped Ft."""
    w, U = np.linalg.eigh(Fh + Ft)
    U = U[:, w > 1e-10 * w.max()]
    if damping == "covariant":
        Ftd = Ft + gamma * Fh
    elif damping == "euclidean":
        Ftd = Ft + gamma * np.trace(Ft) / len(Ft) * np.eye(len(Ft))
    else:
        Ftd = Ft
    lam, Xr = sl.eigh(U.T @ Fh @ U, U.T @ Ftd @ U)
    return np.clip(lam, 0, None), U @ Xr, Ftd


def kfac_repair(ta, tb, facs, mu, damping="covariant", gamma=1e-3, ek=None):
    (Ah, Gh), (At, Gt) = facs
    out = {}
    for k in tb:
        a, X, Atd = gen_eig_range(Ah[k], At[k], damping, gamma)
        g, Y, Gtd = gen_eig_range(Gh[k], Gt[k], damping, gamma)
        dW = np.asarray(tb[k] - ta[k])
        C = Y.T @ Gtd @ dW @ Atd @ X
        lam = np.outer(g, a) if ek is None else ek[k](X, Y)
        M = mu / (lam + mu)
        out[k] = ta[k] + dW - Y @ ((1 - M) * C) @ X.T          # subtraction form: null directions pass through
    return out


def forecast_kfac(facs, damping="covariant", gamma=1e-3):
    (Ah, Gh), (At, Gt) = facs
    total = 0.0
    for k in tgt0:
        a, X, Atd = gen_eig_range(Ah[k], At[k], damping, gamma)
        g, Y, Gtd = gen_eig_range(Gh[k], Gt[k], damping, gamma)
        C = Y.T @ Gtd @ np.asarray(tgt1[k] - tgt0[k]) @ Atd @ X
        total += 0.5 * (np.outer(g, a) * C ** 2).sum()
    return total


# eigenvalue-corrected pencil: keep the K-FAC pencil basis, re-estimate both diagonals from per-token projections
def ek_lambda(k):
    def f(X, Y):
        ph = np.einsum("jo,nok,ki->nji", Y.T, tok_grads(tgt0, Hs_fac, k), X)
        pt = np.einsum("jo,nok,ki->nji", Y.T, tok_grads(tgt1, T_fac, k), X)
        dh, dt = (ph ** 2).sum(0) / (len(Hs_fac) * len(P)), (pt ** 2).sum(0) / (len(T_fac) * len(P))
        return dh / np.maximum(dt, 1e-12 * dt.max())
    return f


EK = {k: ek_lambda(k) for k in tgt0}

# ================================================================ V9 forecast quality and repair quality vs exact
exact_forecast = quad
record("V9 Forecast 0.5 Delta'F_h Delta: K-FAC vs exact (full update)", "REPORT",
       exact=exact_forecast, kfac_all=forecast_kfac(fac["all"]), kfac_upto=forecast_kfac(fac["upto"]),
       kfac_loss=forecast_kfac(fac["loss"]), actual_S_theta1=float(S_flat(flat1)))

mu = 0.3
d_ex = exact_repair(DELTA, F_h, G_t, mu)
rows9 = {"theta0": tgt0, "theta1 (drifted)": tgt1, f"pencil exact mu={mu}": unravel(flat1 + d_ex)}
for m in ("all", "loss"):
    rows9[f"pencil K-FAC ({m}) mu={mu}"] = kfac_repair(tgt0, tgt1, fac[m], mu)
rows9[f"pencil EK-corrected mu={mu}"] = kfac_repair(tgt0, tgt1, fac["all"], mu, ek=EK)
tau = {k: tgt0[k] - tgt_base[k] for k in tgt0}
for a_ in (0.25, 0.5):
    rows9[f"ReSta alpha={a_}"] = {k: tgt1[k] + a_ * tau[k] for k in tgt0}
table = {}
for name, t_ in rows9.items():
    fl_ = ravel_pytree(t_)[0]
    table[name] = {"S_harmful_KL": float(S_flat(fl_)), "task_loss": float(Lt_flat(fl_)),
                   "refusal_rate_heldout": refusal_rate(t_), "copy_acc_heldout": task_acc(t_)}
RESULTS["V9b toy end-to-end table"] = table
print(json.dumps(table, indent=1))

sweep = []
for m in ("loss", "all"):
    for damping, gamma in (("covariant", 1e-3), ("covariant", 1e-1), ("euclidean", 1e-3), ("euclidean", 1e-1)):
        for mu_ in (0.03, 0.3, 3.0):
            t_ = kfac_repair(tgt0, tgt1, fac[m], mu_, damping, gamma)
            fl_ = ravel_pytree(t_)[0]
            sweep.append({"factors": m, "damping": f"{damping} {gamma:g}", "mu": mu_, "S": float(S_flat(fl_)),
                          "task_loss": float(Lt_flat(fl_)), "refusal": refusal_rate(t_), "copy_acc": task_acc(t_)})
RESULTS["V9c K-FAC repair sweep (toy)"] = sweep
for r in sweep:
    print(r)

# ================================================================ V10 V/O reparameterisation invariance
def rand_M(key, cond):
    q1 = np.linalg.qr(np.asarray(jax.random.normal(key, (HD, HD))))[0]
    q2 = np.linalg.qr(np.asarray(jax.random.normal(jax.random.fold_in(key, 1), (HD, HD))))[0]
    return q1 @ np.diag(np.exp(np.linspace(0, np.log(cond), HD))) @ q2


Ms = {(l, g): rand_M(jax.random.PRNGKey(100 + 10 * l + g), 10.0) for l in range(NL) for g in range(NKV)}


def reparam(tgt):
    out = {k: np.array(v) for k, v in tgt.items()}
    for (l, g), M in Ms.items():
        rs = slice(g * HD, (g + 1) * HD)
        out[f"v{l}"][rs] = M @ out[f"v{l}"][rs]
        for h in range(NQ):
            if h // REP == g:
                out[f"o{l}"][:, h * HD:(h + 1) * HD] = out[f"o{l}"][:, h * HD:(h + 1) * HD] @ np.linalg.inv(M)
    return {k: jnp.array(v) for k, v in out.items()}


allseqs = jnp.concatenate([Hs_fac, T_fac])


def logits_all(tgt):
    return np.asarray(jax.vmap(lambda s: forward(fixed, tgt, s)[0])(allseqs))


r0, r1, rb = reparam(tgt0), reparam(tgt1), reparam(tgt_base)
record("V10a Reparameterised model computes the same function", "PASS" if np.abs(logits_all(r0) - logits_all(tgt0)).max() < 1e-9 else "FAIL",
       max_logit_diff=np.abs(logits_all(r0) - logits_all(tgt0)).max())
fac_r = (kfac_exact(r0, Hs_fac, "all"), kfac_exact(r1, T_fac, "all"))
(Ah, Gh), _ = fac["all"]; (Ahr, Ghr), _ = fac_r
Mo = {l: sl.block_diag(*[Ms[(l, h // REP)] for h in range(NQ)]) for l in range(NL)}
Mv = {l: sl.block_diag(*[Ms[(l, g)] for g in range(NKV)]) for l in range(NL)}
e1 = max(np.linalg.norm(Ahr[f"o{l}"] - Mo[l] @ Ah[f"o{l}"] @ Mo[l].T) / np.linalg.norm(Ah[f"o{l}"]) for l in range(NL))
e2 = max(np.linalg.norm(Ghr[f"v{l}"] - np.linalg.inv(Mv[l]).T @ Gh[f"v{l}"] @ np.linalg.inv(Mv[l])) / np.linalg.norm(Gh[f"v{l}"])
         for l in range(NL))
record("V10b K-FAC factors transform by one-sided congruence (Prop 9.1)", "PASS" if max(e1, e2) < 1e-8 else "FAIL",
       o_input_factor_err=e1, v_output_factor_err=e2)
for damping in ("covariant", "none", "euclidean"):
    diff = np.abs(logits_all(kfac_repair(tgt0, tgt1, fac["all"], mu, damping))
                  - logits_all(kfac_repair(r0, r1, fac_r, mu, damping))).max()
    record(f"V10c Pencil K-FAC repair is invariant, damping={damping}",
           ("PASS" if diff < 1e-6 else "FAIL") if damping != "euclidean" else ("REPORT"), max_logit_diff=diff)


def safe_lora_like(ta, tb, tbase, tal):
    out = {}
    for k in tb:
        Vk = np.asarray(tal[k] - tbase[k])
        out[k] = ta[k] + (Vk @ Vk.T / np.linalg.norm(Vk)) @ np.asarray(tb[k] - ta[k])
    return out


diff = np.abs(logits_all(safe_lora_like(tgt0, tgt1, tgt_base, tgt0)) - logits_all(safe_lora_like(r0, r1, rb, r0))).max()
record("V10d Euclidean projection (Safe-LoRA-style) is NOT invariant", "PASS" if diff > 1e-3 else "FAIL", max_logit_diff=diff)
diff = np.abs(logits_all({k: tgt1[k] + 0.25 * (tgt0[k] - tgt_base[k]) for k in tgt0})
              - logits_all({k: r1[k] + 0.25 * (r0[k] - rb[k]) for k in tgt0})).max()
record("V10e ReSta (linear in the weights) IS invariant: positive control", "PASS" if diff < 1e-9 else "FAIL", max_logit_diff=diff)

# ================================================================ V11 small identities used by the implementation
rs = np.random.default_rng(1)
B = rs.standard_normal((6, 6)); A_h = B @ B.T; B = rs.standard_normal((6, 6)); A_t = B @ B.T
lam, X = sl.eigh(A_h, A_t); lam_c, Xc = sl.eigh(A_h, A_t + 0.1 * A_h)
overlap = np.abs(X.T @ A_t @ Xc)          # each covariantly damped eigenvector is parallel to one undamped one
record("V11a Covariant damping keeps eigenvectors and maps lam -> lam/(1+gamma*lam)",
       "PASS" if np.allclose(np.sort(lam_c), np.sort(lam / (1 + 0.1 * lam))) and np.allclose((overlap > 1e-8).sum(0), 1) else "FAIL")
d_in, d_out, r = 7, 5, 2
Bm, Am = rs.standard_normal((d_out, r)), rs.standard_normal((r, d_in))
Ahm = (lambda Z: Z @ Z.T)(rs.standard_normal((d_in, d_in))); Ghm = (lambda Z: Z @ Z.T)(rs.standard_normal((d_out, d_out)))
dW = 0.5 * Bm @ Am
lhs = 0.5 * np.trace(((0.5 * Bm).T @ Ghm @ (0.5 * Bm)) @ (Am @ Ahm @ Am.T))
rhs = 0.5 * dW.reshape(-1) @ np.kron(Ghm, Ahm) @ dW.reshape(-1)
record("V11b P-quad LoRA penalty equals the Kronecker quadratic form", "PASS" if np.isclose(lhs, rhs) else "FAIL", lhs=lhs, rhs=rhs)

Path(__file__).with_name("verify_results.json").write_text(json.dumps(RESULTS, indent=1, default=float))
print("wrote verify_results.json")
