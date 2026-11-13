"""src/pencil/{geneig,repair}.py against the numpy references in test_math.py (rtol 1e-8)."""

import numpy as np
import pytest
import torch
from test_math import pencil_repair_np, pencil_repair_scaled_np, psd_with_null, spd

from pencil.geneig import gen_eig, gen_eig_topk
from pencil.repair import ModelRepair, coords, forecast, frontier, layer_basis, repair_delta, repair_layer


def to_torch(array):
    """A float64 torch tensor from a numpy array."""
    return torch.from_numpy(np.asarray(array, dtype=np.float64))


rng = np.random.default_rng(11)


def _problem(din=7, dout=5, null_in=True, null_out=False):
    u = rng.standard_normal(din)
    v = rng.standard_normal(dout)
    A_h, A_t = (psd_with_null(din, 3, u), psd_with_null(din, 7, u)) if null_in else (spd(din, 3), spd(din))
    G_h, G_t = (psd_with_null(dout, 2, v), psd_with_null(dout, 5, v)) if null_out else (spd(dout, 2), spd(dout))
    dW = rng.standard_normal((dout, din))
    return A_h, A_t, G_h, G_t, dW, u, v


@pytest.mark.parametrize("damping,gamma", [("covariant", 1e-3), ("covariant", 0.2), ("euclidean", 1e-2)])
@pytest.mark.parametrize("null_in,null_out", [(True, False), (False, True), (True, True), (False, False)])
def test_repair_matches_numpy(damping, gamma, null_in, null_out):
    A_h, A_t, G_h, G_t, dW, u, _ = _problem(null_in=null_in, null_out=null_out)
    for mu in (0.01, 0.5, 3.0):
        ref = pencil_repair_np(dW, A_h, A_t, G_h, G_t, mu, damping, gamma)
        W0 = rng.standard_normal(dW.shape)
        W_new, ratio = repair_layer(
            to_torch(W0),
            to_torch(dW),
            to_torch(A_h),
            to_torch(A_t),
            to_torch(G_h),
            to_torch(G_t),
            mu,
            damping=damping,
            gamma=gamma,
        )
        np.testing.assert_allclose(W_new.numpy() - W0, ref, rtol=1e-8, atol=1e-10)
        assert ratio == pytest.approx(np.linalg.norm(ref - dW) / np.linalg.norm(dW), rel=1e-8)


def test_null_component_passes_through():
    A_h, A_t, G_h, G_t, dW, u, _ = _problem(null_in=True)
    basis = layer_basis(to_torch(A_h), to_torch(A_t), to_torch(G_h), to_torch(G_t))
    assert basis.inp.range_dim == A_h.shape[0] - 1
    out = repair_delta(to_torch(dW), basis, 0.3).numpy()
    u_hat = u / np.linalg.norm(u)
    np.testing.assert_allclose(out @ u_hat, dW @ u_hat, rtol=1e-8, atol=1e-10)


def test_truncation_passes_discarded_pairs_through():
    """With only the top-k output pairs kept, M is 1 on the discarded rows (test_math, truncated form)."""
    A_h, A_t, G_h, G_t, dW, _, _ = _problem(null_in=False)
    k, mu = 2, 0.5
    factors = (to_torch(A_h), to_torch(A_t), to_torch(G_h), to_torch(G_t))
    full, trunc = layer_basis(*factors), layer_basis(*factors, top_k_out=k)
    C = coords(to_torch(dW), full)
    M = mu / (torch.outer(full.gamma, full.alpha) + mu)
    M[:-k] = 1.0  # gen_eig is ascending: the top-k are the last rows
    expected = dW - (full.out.X @ ((1 - M) * C) @ full.inp.X.T).numpy()
    np.testing.assert_allclose(repair_delta(to_torch(dW), trunc, mu).numpy(), expected, rtol=1e-8, atol=1e-10)


def test_forecast_equals_pencil_sum_and_ignores_task_factors():
    A_h, A_t, G_h, G_t, dW, _, _ = _problem(null_in=True, null_out=True)
    S = forecast(to_torch(dW), to_torch(A_h), to_torch(G_h))
    D = dW.reshape(-1, order="F")
    assert S == pytest.approx(0.5 * D @ np.kron(A_h, G_h) @ D, rel=1e-10)
    for gamma in (1e-3, 0.3):  # any task factors, any covariant damping
        basis = layer_basis(to_torch(A_h), to_torch(A_t), to_torch(G_h), to_torch(G_t), gamma=gamma)
        C = coords(to_torch(dW), basis)
        assert S == pytest.approx(0.5 * float((torch.outer(basis.gamma, basis.alpha) * C**2).sum()), rel=1e-7)


def test_frontier_matches_repair():
    """Predicted S(mu) equals the forecast of the repaired update (full basis, no truncation)."""
    A_h, A_t, G_h, G_t, dW, _, _ = _problem(null_in=True)
    basis = layer_basis(to_torch(A_h), to_torch(A_t), to_torch(G_h), to_torch(G_t))
    C = coords(to_torch(dW), basis)
    mus = [0.01, 0.1, 1.0]
    S, R = frontier(C, basis.alpha, basis.gamma, mus)
    for mu, s in zip(mus, S):
        assert forecast(repair_delta(to_torch(dW), basis, mu), to_torch(A_h), to_torch(G_h)) == pytest.approx(
            s, rel=1e-7
        )
    assert S == sorted(S) and R == sorted(R, reverse=True)


def test_gen_eig_normalisation_and_errors():
    A_h, A_t, *_ = _problem(null_in=True)
    pencil = gen_eig(to_torch(A_h), to_torch(A_t))
    np.testing.assert_allclose((pencil.X.T @ pencil.FX).numpy(), np.eye(pencil.X.shape[1]), atol=1e-8)
    np.testing.assert_allclose((pencil.X.T @ to_torch(A_h) @ pencil.X).numpy(), np.diag(pencil.lam.numpy()), atol=1e-8)
    assert torch.all(pencil.lam[1:] >= pencil.lam[:-1])
    with pytest.raises(ValueError):
        gen_eig(to_torch(A_h), to_torch(A_t), "covariant", 0.0)
    with pytest.raises(ValueError):
        gen_eig(to_torch(A_h), to_torch(A_t), "bogus")


def test_dtype_roundtrip_bf16():
    A_h, A_t, G_h, G_t, dW, _, _ = _problem()
    W0 = torch.randn(dW.shape, dtype=torch.bfloat16)
    W_new, _ = repair_layer(
        W0, to_torch(dW).to(torch.bfloat16), to_torch(A_h), to_torch(A_t), to_torch(G_h), to_torch(G_t), 0.5
    )
    assert W_new.dtype == torch.bfloat16 and W_new.shape == W0.shape


@pytest.mark.parametrize("damping,gamma", [("covariant", 1e-3), ("euclidean", 1e-2)])
def test_gen_eig_topk_matches_gen_eig_on_full_rank_factors(damping, gamma):
    n, k = 40, 6
    A_h, A_t = to_torch(spd(n, 10)), to_torch(spd(n))
    ref = gen_eig(A_h, A_t, damping, gamma, top_k=k)
    for eig_dtype in (torch.float64, torch.float32):
        pencil = gen_eig_topk(A_h, A_t, k, damping, gamma, jitter=0.0, eig_dtype=eig_dtype)
        tol = 1e-8 if eig_dtype == torch.float64 else 1e-4
        torch.testing.assert_close(pencil.lam, ref.lam, rtol=tol, atol=tol * float(ref.lam.max()))
        # same subspace: projectors onto the eigenvectors agree (signs are arbitrary)
        torch.testing.assert_close(pencil.X @ pencil.FX.T, ref.X @ ref.FX.T, rtol=0, atol=tol * 10)
        torch.testing.assert_close(pencil.X.T @ pencil.FX, torch.eye(k, dtype=torch.float64), rtol=0, atol=tol)


def test_gen_eig_topk_raises_its_jitter_when_fp32_noise_makes_the_task_factor_indefinite():
    """A shared null space pushed slightly negative (as fp32 accumulation does) breaks the Cholesky at the default
    jitter; the retried jitter only touches those directions, so the top pairs match the range-restricted gen_eig."""
    n, rank, k = 60, 40, 6
    generator = np.random.default_rng(7)
    basis, _ = np.linalg.qr(generator.standard_normal((n, n)))
    in_range = basis[:, :rank]  # both factors live here; the other 20 directions are null for both
    A_h = in_range @ spd(rank, 10) @ in_range.T
    A_t = in_range @ spd(rank) @ in_range.T
    null_projector = basis[:, rank:] @ basis[:, rank:].T
    A_t_noisy = A_t - 1e-9 * np.trace(A_t) / n * null_projector  # slightly negative where it should be 0

    reference = gen_eig(to_torch(A_h), to_torch(A_t), "covariant", 1e-3, top_k=k)
    with pytest.warns(UserWarning, match="needed jitter 1e-08"):
        pencil = gen_eig_topk(to_torch(A_h), to_torch(A_t_noisy), k, "covariant", 1e-3, eig_dtype=torch.float64)
    assert pencil.jitter == 1e-8 and pencil.to(torch.float32).jitter == 1e-8  # recorded for the result JSONs
    assert reference.jitter == 0.0
    # the jitter moves λ by about ε·mean(diag) / (its F_t′ eigenvalue): ~1e-6 relative near the 1/γ cap
    torch.testing.assert_close(pencil.lam, reference.lam, rtol=1e-5, atol=0)
    torch.testing.assert_close(pencil.X @ pencil.FX.T, reference.X @ reference.FX.T, rtol=0, atol=1e-5)

    hopeless = A_t - 1.0 * np.trace(A_t) / n * null_projector  # far beyond any jitter: still an error
    with pytest.raises(RuntimeError, match="not positive definite"):
        gen_eig_topk(to_torch(A_h), to_torch(hopeless), k, "covariant", 1e-3)


# ---------------------------------------------------------------- the scaled repair (13 Nov)
class _OneModuleFactors:
    """The factor-store interface ModelRepair reads (a_key, pair), for hand-made factors."""

    def __init__(self, factors_by_module: dict):
        self.factors_by_module = factors_by_module
        self.a_key = {module: module for module in factors_by_module}

    def pair(self, module, row_set):
        return self.factors_by_module[module]


UP = "model.layers.0.mlp.up_proj"  # scaled, ν on its output side
V = "model.layers.0.self_attn.v_proj"  # exempt: the V/O gauge transforms its output factor


def _scaled_repairer(module, A_h, A_t, G_h, G_t):
    harm = _OneModuleFactors({module: (to_torch(A_h), to_torch(G_h))})
    task = _OneModuleFactors({module: (to_torch(A_t), to_torch(G_t))})
    return ModelRepair(harm, task, gamma=1e-3, output_nu_rel=1e-2, scale_steps=True, basis_dtype=torch.float64)


def _low_rank_task_problem(seed):
    """Low-rank task factors, so the unscaled step is often large on task inputs and scaling acts."""
    r = np.random.default_rng(seed)
    A_h, A_t = spd(7, 3), spd(7, 2)
    G_h, G_t = spd(5, 2), spd(5, 1)
    return A_h, A_t, G_h, G_t, r.standard_normal((5, 7))


@pytest.mark.parametrize("seed", range(6))
def test_scaled_repair_matches_numpy(seed):
    A_h, A_t, G_h, G_t, dW = _low_rank_task_problem(seed)
    repairer = _scaled_repairer(UP, A_h, A_t, G_h, G_t)
    W0 = np.zeros_like(dW)
    for mu in (0.01, 0.3, 3.0):
        ref, t, ratio = pencil_repair_scaled_np(dW, A_h, A_t, G_h, G_t, mu)
        new_weights, report = repairer.repair({UP: to_torch(W0)}, {UP: to_torch(dW)}, mu)
        np.testing.assert_allclose(new_weights[UP].numpy(), ref, rtol=1e-7, atol=1e-10)
        row = report["modules"][UP]
        assert row["scale"] == pytest.approx(t, rel=1e-8)
        assert row["task_input_step_ratio_unscaled"] == pytest.approx(ratio, rel=1e-8)
        assert row["task_input_step_ratio"] <= 1 + 1e-9 and row["trust_ratio"] == row["task_input_step_ratio"]


def test_exempt_v_proj_gets_the_plain_covariant_repair_and_a_contraction_ratio():
    A_h, A_t, G_h, G_t, dW = _low_rank_task_problem(10)
    repairer = _scaled_repairer(V, A_h, A_t, G_h, G_t)
    new_weights, report = repairer.repair({V: torch.zeros(5, 7, dtype=torch.float64)}, {V: to_torch(dW)}, 0.1)
    ref = pencil_repair_np(dW, A_h, A_t, G_h, G_t, 0.1)
    np.testing.assert_allclose(new_weights[V].numpy(), ref, rtol=1e-7, atol=1e-10)
    row = report["modules"][V]
    assert row["scale"] == 1.0 and row["trust_ratio"] == row["curvature_step_ratio"] <= 1 + 1e-9


def test_scaled_predictions_equal_the_quadratic_costs_of_the_repaired_step():
    """S_after_pred = ½ tr(G_h ΔW′ A_h ΔW′ᵀ) and R_pred = ½ tr(G_t′ δ A_t′ δᵀ) on a complete pencil."""
    A_h, A_t, G_h, G_t, dW = _low_rank_task_problem(3)
    repairer = _scaled_repairer(UP, A_h, A_t, G_h, G_t)
    deltas = {UP: to_torch(dW)}
    for mu in (0.0, 0.05, 1.0):
        new_weights, report = repairer.repair({UP: torch.zeros(5, 7, dtype=torch.float64)}, deltas, mu)
        repaired = new_weights[UP].numpy()
        delta = repaired - dW
        A_td, G_td = A_t + 1e-3 * A_h, G_t + 1e-3 * G_h
        S_direct = 0.5 * np.trace(G_h @ repaired @ A_h @ repaired.T)
        # S_before − S_kept + ... cancels to rounding when μ = 0 removes everything
        assert report["S_after_pred"] == pytest.approx(S_direct, rel=1e-6, abs=1e-9 * report["S_before_pred"])
        assert report["R_pred"] == pytest.approx(0.5 * np.trace(G_td @ delta @ A_td @ delta.T), rel=1e-6)
        assert repairer.predicted(deltas, mu) == pytest.approx(
            {"S_pred": report["S_after_pred"], "R_pred": report["R_pred"]}
        )


def test_ceiling_diagnostic_flags_an_indefinite_task_factor():
    A_h, A_t, G_h, G_t, dW = _low_rank_task_problem(4)
    w, Q = np.linalg.eigh(G_t)
    G_t_indefinite = Q @ np.diag(w - 0.5 * w.max() * (np.arange(5) == 0) - 1e-3 * w.max()) @ Q.T
    healthy = _scaled_repairer(UP, A_h, A_t, G_h, G_t)
    broken = _scaled_repairer(UP, A_h, A_t, G_h, G_t_indefinite)
    for repairer in (healthy, broken):
        repairer.prepare({UP: to_torch(dW)})
    assert healthy.ceiling_report()["flagged"] == []
    assert broken.ceiling_report()["flagged"] == [f"{UP}|output"]


def test_harmful_input_ratio_matches_its_direct_computation():
    """‖δ A_h^½‖ / ‖ΔW A_h^½‖ from pencil coordinates equals the direct traces (after scaling)."""
    A_h, A_t, G_h, G_t, dW = _low_rank_task_problem(7)
    repairer = _scaled_repairer(UP, A_h, A_t, G_h, G_t)
    for mu in (0.01, 1.0):
        new_weights, report = repairer.repair({UP: torch.zeros(5, 7, dtype=torch.float64)}, {UP: to_torch(dW)}, mu)
        delta = new_weights[UP].numpy() - dW
        direct = np.sqrt(np.trace(delta @ A_h @ delta.T) / np.trace(dW @ A_h @ dW.T))
        assert report["modules"][UP]["harm_input_step_ratio"] == pytest.approx(direct, rel=1e-7)
        assert report["max_harm_input_step_ratio"] == report["modules"][UP]["harm_input_step_ratio"]
