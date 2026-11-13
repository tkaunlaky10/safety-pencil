"""Reference checks for the pencil maths (numpy only, no GPU).

Every test here passed when this file was written (6 Nov 2026). The torch implementation in
src/pencil/ must match these numpy references to rtol 1e-8.

Vec convention: column-stacking, so vec(g a^T) = a (x) g and a Kronecker Fisher block for
W (d_out x d_in) is kron(A, G) with A d_in-sized and G d_out-sized.

`pencil_repair_np` is the reference for what the code must do: restrict each factor pencil to the
range of (harmful + task) factors, damp covariantly, and write the result in subtraction form.
See 01-plan/verification/VERIFICATION.md for why each of those three choices is needed.
"""

import numpy as np
import scipy.linalg as sl

rng = np.random.default_rng(0)


def spd(n, r=None):
    r = r or n
    B = rng.standard_normal((n, r))
    return B @ B.T / r + 1e-3 * np.eye(n)


def psd_with_null(n, r, null_vec):
    """Rank-r PSD matrix whose null space contains null_vec (like a LayerNorm-fed A factor)."""
    P = np.eye(n) - np.outer(null_vec, null_vec) / (null_vec @ null_vec)
    B = P @ rng.standard_normal((n, r))
    return B @ B.T / r


# ------------------------------------------------------------------ reference implementations
def gen_eig_np(F_h, F_t):
    """Solve F_h x = lam F_t x for positive definite F_t. Columns of X satisfy X.T F_t X = I."""
    lam, X = sl.eigh(F_h, F_t)
    return np.clip(lam, 0, None), X


def kron_repair_np(dW, A_h, A_t, G_h, G_t, mu):
    """Textbook form, valid only when A_t and G_t are positive definite."""
    a, X = gen_eig_np(A_h, A_t)
    g, Y = gen_eig_np(G_h, G_t)
    C = Y.T @ G_t @ dW @ A_t @ X
    M = mu / (np.outer(g, a) + mu)
    return Y @ (C * M) @ X.T


def gen_eig_range_np(F_h, F_t, damping="covariant", gamma=1e-3, rel_tol=1e-10):
    """Pencil restricted to range(F_h + F_t). damping: 'covariant' (F_t + gamma F_h, invariant),
    'euclidean' (F_t + gamma mean(diag) I, a trust region, not invariant) or 'none' (valid only
    when null(F_t) is inside null(F_h); otherwise U^T F_t U is singular, so use gamma > 0).
    Returns lam, X (columns orthonormal in the damped F_t), damped F_t."""
    w, U = np.linalg.eigh(F_h + F_t)
    U = U[:, w > rel_tol * w.max()]
    if damping == "covariant":
        F_td = F_t + gamma * F_h
    elif damping == "euclidean":
        F_td = F_t + gamma * np.trace(F_t) / len(F_t) * np.eye(len(F_t))
    else:
        F_td = F_t
    lam, Xr = sl.eigh(U.T @ F_h @ U, U.T @ F_td @ U)
    return np.clip(lam, 0, None), U @ Xr, F_td


def pencil_repair_np(dW, A_h, A_t, G_h, G_t, mu, damping="covariant", gamma=1e-3):
    """What src/pencil/repair.py must compute. Returns the repaired update dW'."""
    a, X, A_td = gen_eig_range_np(A_h, A_t, damping, gamma)
    g, Y, G_td = gen_eig_range_np(G_h, G_t, damping, gamma)
    C = Y.T @ G_td @ dW @ A_td @ X
    M = mu / (np.outer(g, a) + mu)
    return dW - Y @ ((1 - M) * C) @ X.T  # subtraction form: null directions pass through


def gen_eig_range_nu_np(F_h, F_t, gamma=1e-3, nu_rel=1e-2, rel_tol=1e-10):
    """Output side of the scaled repair (added 13 Nov): covariant damping plus nu I on range(F_h + F_t),
    G'' = G_t' + nu I with G_t' = G_t + gamma G_h and nu = nu_rel * mean(diag G_t').
    Returns gamma'', Y (Y^T G'' Y = I), G'' and nu."""
    w, U = np.linalg.eigh(F_h + F_t)
    U = U[:, w > rel_tol * w.max()]
    F_td = F_t + gamma * F_h
    nu = nu_rel * np.trace(F_td) / len(F_td)
    F_tdd = F_td + nu * np.eye(len(F_td))
    lam, Yr = sl.eigh(U.T @ F_h @ U, U.T @ F_tdd @ U)
    return np.clip(lam, 0, None), U @ Yr, F_tdd, nu


def pencil_repair_scaled_np(dW, A_h, A_t, G_h, G_t, mu, gamma=1e-3, nu_rel=1e-2, scale=True):
    """The scaled repair (13 Nov, post-freeze): covariant input side, output side G'' = G_t' + nu I, then the
    step delta = dW' - dW is scaled by t = min(1, 1/r), r = ||delta A_t^1/2||_F / ||dW A_t^1/2||_F.
    nu_rel = 0 and scale=False give pencil_repair_np (the exempt v_proj case uses nu_rel = 0, scale=False).
    Returns (dW + t delta, t, r)."""
    a, X, A_td = gen_eig_range_np(A_h, A_t, "covariant", gamma)
    g, Y, G_tdd, _ = gen_eig_range_nu_np(G_h, G_t, gamma, nu_rel)
    C = Y.T @ G_tdd @ dW @ A_td @ X
    M = mu / (np.outer(g, a) + mu)
    delta = -Y @ ((1 - M) * C) @ X.T
    r = np.sqrt(np.trace(delta @ A_t @ delta.T) / np.trace(dW @ A_t @ dW.T))
    t = min(1.0, 1.0 / r) if scale else 1.0
    return dW + t * delta, t, r


def make_problem(din=6, dout=5):
    A_h, A_t = spd(din, 3), spd(din)
    G_h, G_t = spd(dout, 2), spd(dout)
    dW = rng.standard_normal((dout, din))
    F_h, H_t = np.kron(A_h, G_h), np.kron(A_t, G_t)
    D = dW.reshape(-1, order="F")
    return A_h, A_t, G_h, G_t, dW, F_h, H_t, D


def full_repair_np(D, F_h, H_t, mu):
    return D - np.linalg.solve(F_h + mu * H_t, F_h @ D)


# ------------------------------------------------------------------ Kronecker form and propositions
def test_kron_repair_matches_full_matrix_solution():
    A_h, A_t, G_h, G_t, dW, F_h, H_t, D = make_problem()
    mu = 0.7
    full = full_repair_np(D, F_h, H_t, mu).reshape(dW.shape, order="F")
    assert np.allclose(full, kron_repair_np(dW, A_h, A_t, G_h, G_t, mu))


def test_prop1_drift_decomposition():
    _, _, _, _, _, F_h, H_t, D = make_problem()
    lam, V = sl.eigh(F_h, H_t)
    c = V.T @ H_t @ D
    assert np.isclose(0.5 * D @ F_h @ D, 0.5 * (lam * c**2).sum())
    assert np.isclose(0.5 * D @ H_t @ D, 0.5 * (c**2).sum())
    assert 0.5 * D @ F_h @ D <= lam.max() * 0.5 * D @ H_t @ D + 1e-12


def test_prop2_shrinkage_factors():
    _, _, _, _, _, F_h, H_t, D = make_problem()
    mu = 0.3
    lam, V = sl.eigh(F_h, H_t)
    c = V.T @ H_t @ D
    rep = full_repair_np(D, F_h, H_t, mu)
    c_rep = V.T @ H_t @ rep
    assert np.allclose(c_rep, mu / (lam + mu) * c)


def test_prop3_preserve_equals_repair():
    _, _, _, _, _, F_h, H_t, D = make_problem()
    mu = 0.7
    beta = 1 / mu
    preserve = np.linalg.solve(H_t + beta * F_h, H_t @ D)
    assert np.allclose(preserve, full_repair_np(D, F_h, H_t, mu))


def test_prop4_repair_commutes_with_reparameterisation():
    _, _, _, _, _, F_h, H_t, D = make_problem()
    mu, n = 0.7, D.size
    A = rng.standard_normal((n, n))  # theta = A phi
    F_p, H_p, D_p = A.T @ F_h @ A, A.T @ H_t @ A, np.linalg.solve(A, D)
    assert np.allclose(A @ full_repair_np(D_p, F_p, H_p, mu), full_repair_np(D, F_h, H_t, mu))
    assert np.allclose(sl.eigh(F_p, H_p, eigvals_only=True), sl.eigh(F_h, H_t, eigvals_only=True))


def test_euclidean_projector_is_not_invariant():
    _, _, _, _, _, F_h, _, D = make_problem()
    n = D.size
    A = rng.standard_normal((n, n))

    def project_out_top_k(F, d, k=3):
        _, U = np.linalg.eigh(F)
        U = U[:, -k:]
        return d - U @ (U.T @ d)

    F_p, D_p = A.T @ F_h @ A, np.linalg.solve(A, D)
    assert not np.allclose(A @ project_out_top_k(F_p, D_p), project_out_top_k(F_h, D))


def test_truncated_subtraction_form():
    """Keep only the top-k output-side pairs; the rest pass through unchanged."""
    A_h, A_t, G_h, G_t, dW, _, _, _ = make_problem()
    mu, k = 0.5, 2
    a, X = gen_eig_np(A_h, A_t)
    g, Y = gen_eig_np(G_h, G_t)
    keep = np.argsort(g)[-k:]
    C = Y.T @ G_t @ dW @ A_t @ X
    M = mu / (np.outer(g, a) + mu)
    sub = dW - Y[:, keep] @ ((1 - M[keep]) * C[keep]) @ X.T
    M_ref = M.copy()
    M_ref[np.setdiff1d(np.arange(len(g)), keep)] = 1.0
    assert np.allclose(sub, Y @ (C * M_ref) @ X.T)


def test_truncation_error_bound():
    """Extra safety drift from truncation <= 0.5 * eps * min(1, 2 eps / mu) * ||C_discarded||^2,
    with eps = alpha_max * gamma_(k+1)."""
    A_h, A_t, G_h, G_t, dW, _, _, _ = make_problem()
    mu, k = 0.5, 2
    a, X = gen_eig_np(A_h, A_t)
    g, Y = gen_eig_np(G_h, G_t)
    order = np.argsort(g)[::-1]
    drop = order[k:]
    C = Y.T @ G_t @ dW @ A_t @ X
    lam = np.outer(g, a)
    M = mu / (lam + mu)
    M_tr = M.copy()
    M_tr[drop] = 1.0
    S_full = 0.5 * (lam * (M * C) ** 2).sum()
    S_trunc = 0.5 * (lam * (M_tr * C) ** 2).sum()
    eps = a.max() * g[drop].max()
    bound = 0.5 * eps * min(1.0, 2 * eps / mu) * (C[drop] ** 2).sum()
    assert S_full <= S_trunc <= S_full + bound + 1e-12


def test_nullspace_projector_is_mu_to_zero_limit_of_input_side():
    din, dout = 6, 5
    B = rng.standard_normal((din, 3))
    A_h = B @ B.T / 3  # rank 3
    dW = rng.standard_normal((dout, din))
    w, U = np.linalg.eigh(A_h)
    P_null = U[:, w < 1e-9] @ U[:, w < 1e-9].T
    mu = 1e-10
    filt = dW @ U @ np.diag(mu / (np.clip(w, 0, None) + mu)) @ U.T
    assert np.allclose(filt, dW @ P_null, atol=1e-4)


def test_frontier_slope_is_minus_mu():
    lam = rng.exponential(1.0, 50)
    c = rng.standard_normal(50)
    S = lambda m: 0.5 * np.sum(lam * (m / (lam + m)) ** 2 * c**2)
    R = lambda m: 0.5 * np.sum((lam / (lam + m)) ** 2 * c**2)
    for mu in [0.1, 1.0, 5.0]:
        h = 1e-6
        slope = (S(mu + h) - S(mu - h)) / (R(mu + h) - R(mu - h))
        assert np.isclose(slope, -mu, rtol=1e-5)


def test_water_filling_bit_allocation():
    from scipy.optimize import minimize

    n = rng.integers(100, 1000, 6).astype(float)
    k = rng.exponential(1.0, 6)
    b_bar = 4.0
    res = minimize(
        lambda b: np.sum(n * k * 2.0 ** (-2 * b)),
        np.full(6, b_bar),
        constraints={"type": "eq", "fun": lambda b: np.sum(n * b) - n.sum() * b_bar},
    )
    k_bar = np.exp(np.sum(n * np.log(k)) / n.sum())
    assert np.allclose(res.x, b_bar + 0.5 * np.log2(k / k_bar), atol=1e-3)


# ------------------------------------------------------------------ added after the toy-transformer verification
def test_repair_is_fisher_weighted_merging():
    """theta0 + Delta_rep = (F_h + mu H_t)^-1 (F_h theta0 + mu H_t theta1)."""
    _, _, _, _, _, F_h, H_t, D = make_problem()
    mu = 0.4
    th0 = rng.standard_normal(D.size)
    th1 = th0 + D
    merged = np.linalg.solve(F_h + mu * H_t, F_h @ th0 + mu * H_t @ th1)
    assert np.allclose(th0 + full_repair_np(D, F_h, H_t, mu), merged)


def test_covariant_damping_identities():
    """With H' = H + gamma F: same eigenvectors, lam' = lam/(1+gamma lam), the forecast is unchanged,
    and the repair equals the undamped repair at mu/(1+mu gamma), scaled by 1/(1+mu gamma)."""
    _, _, _, _, _, F_h, H_t, D = make_problem()
    mu, gamma = 0.3, 0.2
    H_c = H_t + gamma * F_h
    lam, V = sl.eigh(F_h, H_t)
    lam_c, V_c = sl.eigh(F_h, H_c)
    assert np.allclose(np.sort(lam_c), np.sort(lam / (1 + gamma * lam)))
    c, c_c = V.T @ H_t @ D, V_c.T @ H_c @ D
    assert np.isclose((lam * c**2).sum(), (lam_c * c_c**2).sum())  # forecast unchanged
    step_c = -np.linalg.solve(F_h + mu * H_c, F_h @ D)
    mu_eff = mu / (1 + mu * gamma)
    step_eff = -np.linalg.solve(F_h + mu_eff * H_t, F_h @ D)
    assert np.allclose(step_c, step_eff / (1 + mu * gamma))


def test_step_is_bounded_in_curvature_norms_but_not_euclidean():
    """||delta||_H <= ||Delta||_H and ||Delta + delta||_F <= ||Delta||_F always, but the Euclidean
    step can be several times the update when H is ill-conditioned (why a trust region is needed)."""
    r = np.random.default_rng(3)
    n = 20
    Q = np.linalg.qr(r.standard_normal((n, n)))[0]
    H = Q @ np.diag(np.logspace(-6, 0, n)) @ Q.T
    B = r.standard_normal((n, 5))
    F = B @ B.T / 5
    D = r.standard_normal(n)
    mu = 0.3
    d = -np.linalg.solve(F + mu * H, F @ D)
    assert d @ H @ d <= D @ H @ D + 1e-12
    assert (D + d) @ F @ (D + d) <= D @ F @ D + 1e-12
    assert np.linalg.norm(d) > 2 * np.linalg.norm(D)
    # Euclidean (Levenberg-Marquardt) damping shrinks the step monotonically
    norms = [np.linalg.norm(np.linalg.solve(F + mu * (H + g * np.eye(n)), F @ D)) for g in (0.0, 1e-3, 1e-2, 1e-1)]
    assert all(x > y for x, y in zip(norms, norms[1:]))


def test_range_restricted_repair_matches_full_matrix_and_keeps_null_part():
    """Factors share an exact null vector (as LayerNorm-fed inputs do). The restricted, covariantly
    damped Kronecker repair equals the full-matrix minimiser on the range, and leaves the update's
    null-space component untouched."""
    din, dout, mu, gamma = 6, 5, 0.5, 1e-3
    u = rng.standard_normal(din)  # shared input-side null vector
    A_h, A_t = psd_with_null(din, 3, u), psd_with_null(din, 6, u)
    G_h, G_t = spd(dout, 2), spd(dout)
    dW = rng.standard_normal((dout, din))
    out = pencil_repair_np(dW, A_h, A_t, G_h, G_t, mu, "covariant", gamma)
    # full-matrix reference on range(A) (x) range(G), with the same covariant damping on each factor
    A_td, G_td = A_t + gamma * A_h, G_t + gamma * G_h
    K_h, K_t = np.kron(A_h, G_h), np.kron(A_td, G_td)
    wA, UA = np.linalg.eigh(A_h + A_t)
    UA = UA[:, wA > 1e-10 * wA.max()]
    U = np.kron(UA, np.eye(dout))
    D = dW.reshape(-1, order="F")
    step = -U @ np.linalg.solve(U.T @ (K_h + mu * K_t) @ U, U.T @ K_h @ D)
    assert np.allclose(out, (D + step).reshape(dW.shape, order="F"))
    un = u / np.linalg.norm(u)
    assert np.allclose(out @ un, dW @ un)  # null component passes through


def test_kronecker_repair_invariance_depends_on_damping():
    """One-sided gauge A -> M A M^T, dW -> dW M^-1 (the o_proj case), inputs x -> M x.
    Covariant or no damping: the repaired layer gives the same output on every input the model can
    produce (inputs live in range(A)); the raw weights may differ only along exactly-null directions.
    Euclidean damping: outputs differ."""
    din, dout, mu = 6, 5, 0.5
    u = rng.standard_normal(din)
    A_h, A_t = psd_with_null(din, 3, u), psd_with_null(din, 6, u)
    G_h, G_t = spd(dout, 2), spd(dout)
    dW = rng.standard_normal((dout, din))
    M = rng.standard_normal((din, din)) + 3 * np.eye(din)
    w, U = np.linalg.eigh(A_h + A_t)
    reachable = U[:, w > 1e-10 * w.max()]  # inputs the original model can see
    for damping, invariant in (("covariant", True), ("none", True), ("euclidean", False)):
        gamma = 1e-1 if damping != "none" else 0.0
        a = pencil_repair_np(dW, A_h, A_t, G_h, G_t, mu, damping, gamma)
        b = pencil_repair_np(dW @ np.linalg.inv(M), M @ A_h @ M.T, M @ A_t @ M.T, G_h, G_t, mu, damping, gamma)
        same_outputs = np.allclose(b @ M @ reachable, a @ reachable)  # b(Mx) == a(x) for reachable x
        assert same_outputs == invariant, damping
        if damping == "covariant":
            assert not np.allclose(b, a @ np.linalg.inv(M))  # raw weights differ along the null direction


def test_kronecker_repair_invariance_output_side():
    """The v_proj case: outputs y -> M y, so dW -> M dW and G -> M^-T G M^-1 (A unchanged), with
    a shared null vector in G (an output direction no gradient touches). Covariant damping: the
    repaired layer maps back to the original on every output direction the loss can see."""
    din, dout, mu, gamma = 6, 5, 0.5, 1e-1
    v = rng.standard_normal(dout)
    A_h, A_t = spd(din, 3), spd(din)
    G_h, G_t = psd_with_null(dout, 2, v), psd_with_null(dout, 5, v)
    dW = rng.standard_normal((dout, din))
    M = rng.standard_normal((dout, dout)) + 3 * np.eye(dout)
    Mi = np.linalg.inv(M)
    w, U = np.linalg.eigh(G_h + G_t)
    seen = U[:, w > 1e-10 * w.max()]  # output directions the loss can see
    for damping, invariant in (("covariant", True), ("euclidean", False)):
        a = pencil_repair_np(dW, A_h, A_t, G_h, G_t, mu, damping, gamma)
        b = pencil_repair_np(M @ dW, A_h, A_t, Mi.T @ G_h @ Mi, Mi.T @ G_t @ Mi, mu, damping, gamma)
        assert np.allclose(seen.T @ (Mi @ b), seen.T @ a) == invariant, damping


def test_per_head_leverage_is_a_lower_bound():
    """The top eigenvalue of any principal block of the pencil is <= the full top eigenvalue."""
    F, H = spd(12, 4), spd(12)
    top = sl.eigh(F, H, eigvals_only=True).max()
    for blk in (slice(0, 4), slice(4, 8), slice(8, 12)):
        assert sl.eigh(F[blk, blk], H[blk, blk], eigvals_only=True).max() <= top + 1e-10


# ------------------------------------------------------------------ the scaled repair (added 13 Nov, post-freeze)
def test_output_damping_adds_an_input_weighted_step_penalty():
    """1/2 tr((G' + nu I) d A' d^T) = R(d) + nu/2 ||d A'^1/2||^2, with R(d) = 1/2 tr(G' d A' d^T)."""
    A_h, A_t, G_h, G_t, _, _, _, _ = make_problem()
    gamma, nu = 1e-3, 0.07
    A_td, G_td = A_t + gamma * A_h, G_t + gamma * G_h
    d = rng.standard_normal((5, 6))
    damped = 0.5 * np.trace((G_td + nu * np.eye(5)) @ d @ A_td @ d.T)
    R = 0.5 * np.trace(G_td @ d @ A_td @ d.T)
    input_weighted = np.trace(d @ A_td @ d.T)  # ||d A'^1/2||_F^2
    assert np.isclose(damped, R + 0.5 * nu * input_weighted)


def test_scaled_repair_minimises_the_damped_objective_before_scaling():
    """Unscaled, the closed form is the full-matrix minimiser of S + mu R + mu nu/2 ||d A'^1/2||^2."""
    A_h, A_t, G_h, G_t, dW, _, _, D = make_problem()
    mu, gamma, nu_rel = 0.4, 1e-3, 1e-2
    _, _, G_tdd, _ = gen_eig_range_nu_np(G_h, G_t, gamma, nu_rel)
    K_h, K_t = np.kron(A_h, G_h), np.kron(A_t + gamma * A_h, G_tdd)
    full = np.linalg.solve(K_h + mu * K_t, mu * K_t @ D).reshape(dW.shape, order="F")
    ours, t, _ = pencil_repair_scaled_np(dW, A_h, A_t, G_h, G_t, mu, gamma, nu_rel, scale=False)
    assert t == 1.0 and np.allclose(ours, full)


def test_scaled_repair_without_nu_or_scaling_is_the_covariant_repair():
    A_h, A_t, G_h, G_t, dW, _, _, _ = make_problem()
    ours, _, _ = pencil_repair_scaled_np(dW, A_h, A_t, G_h, G_t, 0.3, nu_rel=0.0, scale=False)
    assert np.allclose(ours, pencil_repair_np(dW, A_h, A_t, G_h, G_t, 0.3))


def test_scaled_drift_is_non_increasing_in_t_on_random_pencils():
    """S(t) = 1/2 sum lam (1 - t(1 - M))^2 C^2 equals 1/2 tr(G_h (dW + t delta) A_h (.)^T) on a complete pencil
    and never rises as t goes from 0 (theta1) to 1 (the full repair)."""
    r = np.random.default_rng(5)
    ts = np.linspace(0, 1, 21)
    for _ in range(20):
        A_h, A_t = spd(6, 3), spd(6)
        G_h, G_t = spd(5, 2), spd(5)
        dW = r.standard_normal((5, 6))
        mu = 10 ** r.uniform(-3, 1)
        a, X, A_td = gen_eig_range_np(A_h, A_t, "covariant", 1e-3)
        g, Y, G_tdd, _ = gen_eig_range_nu_np(G_h, G_t, 1e-3, 1e-2)
        C = Y.T @ G_tdd @ dW @ A_td @ X
        lam = np.outer(g, a)
        M = mu / (lam + mu)
        delta = -Y @ ((1 - M) * C) @ X.T
        S_formula = [0.5 * np.sum(lam * (1 - t * (1 - M)) ** 2 * C**2) for t in ts]
        S_direct = [0.5 * np.trace(G_h @ (dW + t * delta) @ A_h @ (dW + t * delta).T) for t in ts]
        assert np.allclose(S_formula, S_direct)
        assert all(later <= earlier + 1e-12 for earlier, later in zip(S_formula, S_formula[1:]))


def test_scaling_keeps_every_input_weighted_ratio_at_most_one():
    r = np.random.default_rng(6)
    for _ in range(30):
        A_h, A_t = spd(6, 3), spd(6, 2)  # a low-rank task input factor: steps can be large on task inputs
        G_h, G_t = spd(5, 2), spd(5, 1)
        dW = r.standard_normal((5, 6))
        mu = 10 ** r.uniform(-3, 1)
        repaired, t, ratio = pencil_repair_scaled_np(dW, A_h, A_t, G_h, G_t, mu)
        delta = repaired - dW
        after = np.sqrt(np.trace(delta @ A_t @ delta.T) / np.trace(dW @ A_t @ dW.T))
        assert after <= 1 + 1e-10
        assert np.isclose(t, min(1.0, 1.0 / ratio)) and np.isclose(after, min(ratio, 1.0))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("pass", name)
