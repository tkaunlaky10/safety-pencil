"""How does the pencil repair fare against simpler and prior-art merges on the same toy transformer?

Reuses the model, data, theta_0/theta_1 and K-FAC helpers from verify_toy_transformer.py verbatim
(same seeds, so the same models), then traces the refusal / task frontier of five repairs:

  interp      theta_0 + t * Delta                       (uniform shrinkage: every lambda equal)
  diag-Fisher Matena & Raffel (2022) Fisher-weighted averaging with diagonal Fishers
  RegMean     input-Gram merge per linear layer (Jin et al. 2023), i.e. the K-FAC merge with G = I
  pencil-all  K-FAC pencil repair, factors over all positions        (our method)
  pencil-loss K-FAC pencil repair, factors over loss positions only  (our method)

Every row is the same objective family, min_theta  d(theta, theta_0; safety metric) + mu * d(theta, theta_1; task metric),
so the comparison isolates which curvature model is used. Evaluation uses 512 fresh held-out prompts per side. Run: python baselines_toy.py  (CPU, ~5 minutes).
Writes baselines_results.json.
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "verify_toy_transformer.py").read_text()

# 1) model, data, training and evaluation helpers: everything before the exact-curvature section
exec(SRC.split("# ================================================================ exact curvature")[0])

# 2) the helper functions we need, copied verbatim from the verification script's syntax tree
WANT = {"fisher_exact", "kfac_exact", "gen_eig_range", "kfac_repair"}
for node in ast.parse(SRC).body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        exec(compile(ast.Module([node], []), "verify_toy_transformer.py", "exec"))

t0 = time.time()
Fd_h = np.diag(fisher_exact(flat0, Hs_fac)).copy()      # exact diagonal Fishers (no sampling)
Fd_t = np.diag(fisher_exact(flat1, T_fac)).copy()
fac = {m: (kfac_exact(tgt0, Hs_fac, m), kfac_exact(tgt1, T_fac, m)) for m in ("all", "loss")}
print(f"curvature in {time.time() - t0:.0f}s")


# larger fresh evaluation sets, drawn after every training draw so the models are unchanged
H_big, T_big = harmful_prompts(512), jnp.array(task_seqs(512))


def evaluate(t_):
    fl_ = ravel_pytree(t_)[0]
    return {"S": float(S_flat(fl_)), "task_loss": float(Lt_flat(fl_)),
            "refusal": refusal_rate(t_, H_big), "copy_acc": task_acc(t_, T_big)}


def interp(t):
    return {k: tgt0[k] + t * (tgt1[k] - tgt0[k]) for k in tgt0}


def diag_fisher(mu, floor=1e-8):
    fh = np.maximum(Fd_h, floor * Fd_h.max())
    ft = np.maximum(Fd_t, floor * Fd_t.max())
    merged = (fh * np.asarray(flat0) + mu * ft * np.asarray(flat1)) / (fh + mu * ft)
    return unravel(jnp.array(merged))


def regmean(mu, factors="all"):
    """W = (W0 A_h + mu W1 A_t)(A_h + mu A_t)^+, written in subtraction form so null directions pass through."""
    (Ah, _), (At, _) = fac[factors]
    out = {}
    for k in tgt0:
        dW = np.asarray(tgt1[k] - tgt0[k])
        out[k] = tgt1[k] - dW @ Ah[k] @ np.linalg.pinv(Ah[k] + mu * At[k], rcond=1e-10, hermitian=True)
    return out


rows = [{"method": "theta0", "param": None, **evaluate(tgt0)},
        {"method": "theta1", "param": None, **evaluate(tgt1)}]
for t in (0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9):
    rows.append({"method": "interp", "param": t, **evaluate(interp(t))})
for mu in (0.001, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 1.0, 3.0):
    rows.append({"method": "diag-Fisher", "param": mu, **evaluate(diag_fisher(mu))})
    for m in ("all", "loss"):
        rows.append({"method": f"pencil-{m}", "param": mu, **evaluate(kfac_repair(tgt0, tgt1, fac[m], mu))})
for mu in (0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 30.0, 100.0, 1000.0):   # RegMean has no G factor, so its mu lives on another scale
    rows.append({"method": "RegMean", "param": mu, **evaluate(regmean(mu))})

for r in rows:
    print(f"{r['method']:12} {str(r['param']):6} S={r['S']:.3f} L={r['task_loss']:.3f} "
          f"refusal={r['refusal']:.3f} copy_acc={r['copy_acc']:.3f}")


def best_acc_at(method, min_refusal):
    ok = [r["copy_acc"] for r in rows if r["method"] == method and r["refusal"] >= min_refusal]
    return max(ok) if ok else None


summary = {m: {f"best_copy_acc_at_refusal>={q}": best_acc_at(m, q) for q in (0.5, 0.9, 0.95, 1.0)}
           for m in ("interp", "diag-Fisher", "RegMean", "pencil-all", "pencil-loss")}
print(json.dumps(summary, indent=1))
(HERE / "baselines_results.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
print("wrote baselines_results.json")
