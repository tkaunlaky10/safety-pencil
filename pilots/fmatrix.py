"""T3 pilot for E3: is the in-sample generalised eigenbasis noise, and does cross-fitting fix the repair?

One-sided pencil A_h x = α A_t x (the input side of K-FAC), estimated from n samples per side, with an
ill-conditioned task covariance as in real activations. Repair of ΔW at shrink μ:
    ΔW' = Σ_i μ/(α_i+μ) (ΔW A_t x_i) x_iᵀ,     x_i A_t-orthonormal.
Estimators: in-sample; cross-fitted (eigenvectors from half A, Rayleigh quotients from half B, swap and average);
cross-fitted + log-shrinkage toward the median (group jackknife); WiSE-FT (one uniform shrink); oracle (true
matrices). Scored on the TRUE objective J = ½tr(ΔW'A_hΔW'ᵀ) + μ/2 tr((ΔW'−ΔW)A_t(ΔW'−ΔW)ᵀ).
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

rng = np.random.default_rng(0)
p, dout, mu = 64, 32, 1.0


def rand_orth(k):
    q, r = np.linalg.qr(rng.normal(size=(k, k)))
    return q * np.sign(np.diag(r))


def make_truth(kind):
    Qt = rand_orth(p)
    At = Qt @ np.diag(np.logspace(-3, 0, p)) @ Qt.T
    if kind == "null":
        lam = np.ones(p)
    elif kind == "spiked":
        lam = np.ones(p)
        lam[:4] = [40, 15, 6, 3]
    else:  # continuous spread
        lam = np.exp(rng.normal(0, 1.0, p))
    R = rand_orth(p)
    w, Vt = np.linalg.eigh(At)
    Ath = Vt @ np.diag(np.sqrt(w)) @ Vt.T
    Ah = Ath @ R @ np.diag(lam) @ R.T @ Ath
    return Ah, At


def J(dWp, dW, Ah, At):
    d = dWp - dW
    return 0.5 * np.trace(dWp @ Ah @ dWp.T) + 0.5 * mu * np.trace(d @ At @ d.T)


def repair(dW, X, At_hat, alpha):
    C = dW @ At_hat @ X
    return (C * (mu / (alpha + mu))[None]) @ X.T


def gen_eig(Ah, At):
    ridge = 1e-10 * np.trace(At) / p
    a, X = eigh(Ah, At + ridge * np.eye(p))
    return np.maximum(a, 0), X


def sample_cov(A, n):
    w, V = np.linalg.eigh(A)
    Z = rng.normal(size=(n, p)) @ (V * np.sqrt(np.maximum(w, 0))).T
    return Z


def estimators(dW, Zh, Zt, G=8):
    n = Zh.shape[0]
    cov = lambda Z: Z.T @ Z / Z.shape[0]
    out = {}
    a, X = gen_eig(cov(Zh), cov(Zt))
    out["in_sample"] = repair(dW, X, cov(Zt), a)
    halves = [(slice(0, n // 2), slice(n // 2, n)), (slice(n // 2, n), slice(0, n // 2))]
    cf, cfs = [], []
    for A_, B_ in halves:
        aA, XA = gen_eig(cov(Zh[A_]), cov(Zt[A_]))
        ZhB, ZtB = Zh[B_], Zt[B_]
        num = ((ZhB @ XA) ** 2).mean(0)
        den = ((ZtB @ XA) ** 2).mean(0)
        lam = num / np.maximum(den, 1e-300)
        cf.append(repair(dW, XA, cov(Zt[A_]), lam))
        # group jackknife variance of log λ over G groups of half B
        gl = []
        m = ZhB.shape[0] // G
        for g in range(G):
            sl = slice(g * m, (g + 1) * m)
            gl.append(np.log(((ZhB[sl] @ XA) ** 2).mean(0) / np.maximum(((ZtB[sl] @ XA) ** 2).mean(0), 1e-300)))
        gl = np.array(gl)
        var_i = gl.var(0, ddof=1) / G
        ll = np.log(lam)
        med = np.median(ll)
        s2 = ll.var()
        w = np.clip(1 - var_i / max(s2, 1e-12), 0, 1)
        cfs.append(repair(dW, XA, cov(Zt[A_]), np.exp(med + w * (ll - med))))
    out["cross_fit"] = 0.5 * (cf[0] + cf[1])
    out["cross_fit_shrunk"] = 0.5 * (cfs[0] + cfs[1])
    kappa = np.exp(np.median(np.log(np.maximum(a, 1e-300))))
    out["wise"] = mu / (kappa + mu) * dW
    return out


results = {}
for kind in ("spiked", "continuous", "null"):
    Ah, At = make_truth(kind)
    aT, XT = gen_eig(Ah, At)
    rows = []
    for n in (96, 128, 192, 256, 512, 1024, 4096):
        acc = {}
        for _rep in range(20):
            dW = rng.normal(size=(dout, p))
            Jstar = J(repair(dW, XT, At, aT), dW, Ah, At)
            J0 = J(dW, dW, Ah, At)
            Zh, Zt = sample_cov(Ah, n), sample_cov(At, n)
            for name, dWp in estimators(dW, Zh, Zt).items():
                lost = (J(dWp, dW, Ah, At) - Jstar) / (J0 - Jstar)
                step = np.linalg.norm(dWp - dW) / np.linalg.norm(dW)
                acc.setdefault(name, {"lost": [], "step": []})
                acc[name]["lost"].append(lost)
                acc[name]["step"].append(step)
        rows.append(
            {
                "n": n,
                "y=p/n": round(p / n, 3),
                **{
                    k: {
                        "lost_median": float(np.median(v["lost"])),
                        "step_median": float(np.median(v["step"])),
                        "step_max": float(np.max(v["step"])),
                    }
                    for k, v in acc.items()
                },
            }
        )
    results[kind] = rows
    print(kind)
    for r in rows:
        print(
            f"  n={r['n']:5d} y={r['y=p/n']:.3f} | "
            + " | ".join(
                f"{k}: lost {r[k]['lost_median']:.2f} step {r[k]['step_median']:.1f}/{r[k]['step_max']:.1f}"
                for k in ("in_sample", "cross_fit", "cross_fit_shrunk", "wise")
            )
        )

# Wachter check under the null: spread of in-sample eigenvalues vs the theoretical support
Ah, At = make_truth("null")
wach = []
for n in (192, 1024):
    y = p / n
    h = np.sqrt(2 * y - y * y)
    lo, hi = ((1 - h) / (1 - y)) ** 2, ((1 + h) / (1 - y)) ** 2
    ev = np.concatenate(
        [
            gen_eig(*(lambda Zh, Zt: (Zh.T @ Zh / n, Zt.T @ Zt / n))(sample_cov(Ah, n), sample_cov(At, n)))[0]
            for _ in range(20)
        ]
    )
    wach.append(
        {"n": n, "theory": [lo, hi], "empirical_1_99pct": [float(np.percentile(ev, 1)), float(np.percentile(ev, 99))]}
    )
results["wachter_null"] = wach
print("wachter", wach)
out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)
(out / "fmatrix.json").write_text(json.dumps(results, indent=1))
