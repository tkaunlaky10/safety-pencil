# ruff: noqa: F821, E402  (names below come from exec of toy.py's set-up code)
"""Pilot for safe fine-tuning (prevention): are Fisher/EWC-style safety penalties blind on confident refusals?

Reuses the toy world of toy.py (θ0 and data), then fine-tunes the same LoRA on the benign task with a penalty
computed on synthetic "harmful" prompts (no natural-language text anywhere):
  none        task loss only
  ewc         λ Σ F_h,w (θ_w − θ0_w)²          diagonal Fisher of the drift at θ0 (sampled labels)
  kl          β · KL(p_θ0 || p_θ) on θ0's answers (the exact log-partition drift)
  margin      β · mean max(0, m* − m_θ)         m = refusal logit minus the best other logit at the answer position
Prints task accuracy/NLL, harmful compliance, over-refusal, J and drift for each, plus the Fisher-vs-margin check.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "toy.py").read_text()
cut = src.index("# ---------------------------------------------------------------- T1")
sys.argv = ["toy.py", sys.argv[1] if len(sys.argv) > 1 else "0", "TASK"]
exec(compile(src[:cut], "toy_prefix", "exec"))  # builds θ0, θ1, data, metrics

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

print("θ1 (no penalty)", {k: round(v, 3) for k, v in res["theta1"].items()}, flush=True)

# a larger, cleaner benign fine-tune set, so the unpenalised fine-tune learns the task reliably
ft_toks, ft_mask = make("task", 2048, noise=0.05)


def ft_batch(i, bs=64):
    idx = rng.choice(2048, bs, replace=False)
    return ft_toks[idx], ft_mask[idx]


# Fisher of the drift at θ0 and θ0's refusal margins: the e^{-m} suppression
Fh_diag = None


def diag_fisher_drift(p, toks, mask, key, nsamp=2):
    def lp_sampled(params, tk, mk, k):
        f = forward(params, tk[None])[0]
        yy = jax.random.categorical(k, f)
        return (jnp.take_along_axis(jax.nn.log_softmax(f), yy[:, None], -1)[:, 0] * mk).sum()

    gfn = jax.jit(jax.vmap(jax.grad(lp_sampled), in_axes=(None, 0, 0, 0)))
    acc = jax.tree_util.tree_map(jnp.zeros_like, p)
    keys = jax.random.split(key, toks.shape[0] * nsamp)
    n = 0
    for r in range(nsamp):
        for c in range(0, toks.shape[0], 128):
            g = gfn(p, toks[c : c + 128], mask[c : c + 128], keys[r * toks.shape[0] + c : r * toks.shape[0] + c + 128])
            acc = jax.tree_util.tree_map(lambda a, b: a + (b**2).sum(0), acc, g)
            n += toks[c : c + 128].shape[0]
    return jax.tree_util.tree_map(lambda a: a / n, acc)


Fh = diag_fisher_drift(theta0, H_FIT, H_FIT_M, jax.random.PRNGKey(5))
f_ans = forward(theta0, H_FIT)[:, 2 + K]
srt = jnp.sort(f_ans, -1)
m0 = np.array(srt[:, -1] - srt[:, -2])
p_ref = np.array(jax.nn.softmax(f_ans)[:, REFUSE])
fisher_trace = float(sum(jnp.sum(get(Fh, m)) for m in MODULES))
print(
    f"θ0 refusal margin median {np.median(m0):.2f}, p(REFUSE) median {np.median(p_ref):.6f}, "
    f"drift-Fisher trace on the 8 modules {fisher_trace:.3e}",
    flush=True,
)

harm_pen_toks, harm_pen_mask = H_FIT, H_FIT_M  # θ0's own answers (REFUSE, EOS) on synthetic harmful prompts
m_star = float(np.median(m0))


def margin_at_answer(p, toks):
    f = forward(p, toks)[:, 2 + K]
    ref = f[:, REFUSE]
    other = jnp.max(f.at[:, REFUSE].set(-1e9), -1)
    return ref - other


def penalised_loss(kind, coef):
    def loss(lp, toks, mask):
        lpd = {tuple(k.split("|")): v for k, v in lp.items()}
        p = lora_params_to_model(lpd)
        base = nll(p, toks, mask)
        if kind == "none":
            return base
        idx = jnp.arange(128)
        ht, hm = harm_pen_toks[idx], harm_pen_mask[idx]
        if kind == "ewc":
            pen = sum(jnp.sum(get(Fh, m) * (get(p, m) - get(theta0, m)) ** 2) for m in MODULES)
        elif kind == "kl":
            pen = (drift_tokens(p, ht) * hm).sum() / hm.sum()
        else:  # margin hinge
            pen = jnp.mean(jax.nn.relu(m_star - margin_at_answer(p, ht)))
        return base + coef * pen

    return loss


rows = []
configs = [
    ("none", 0.0),
    ("ewc", 1e1),
    ("ewc", 1e2),
    ("ewc", 1e3),
    ("ewc", 1e4),
    ("kl", 0.1),
    ("kl", 1.0),
    ("margin", 0.03),
    ("margin", 0.3),
]
for kind, coef in configs:
    rng = np.random.default_rng(SEED + 100)  # same batch order for every penalty (paired)
    lp0 = {f"{a}|{b}": v for (a, b), v in lora_init(jax.random.PRNGKey(SEED + 1)).items()}
    lp, lfin = adam_train(penalised_loss(kind, coef), lp0, 1500, 3e-3, ft_batch)
    model = lora_params_to_model({tuple(k.split("|")): v for k, v in lp.items()})
    r = {"penalty": kind, "coef": coef, **report(model)}
    rows.append(r)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

summary = {
    "seed": SEED,
    "theta0": res["theta0"],
    "margin_median": float(np.median(m0)),
    "p_refuse_median": float(np.median(p_ref)),
    "fisher_trace": fisher_trace,
    "rows": rows,
}
(HERE / "results" / f"prevent_seed{SEED}.json").write_text(json.dumps(summary, indent=1, default=float))
