"""src/pencil/{geneig,repair}.py against the numpy references in test_math.py (rtol 1e-8)."""

import numpy as np
import pytest
import torch
from test_math import pencil_repair_np, psd_with_null, spd

from pencil.geneig import gen_eig, gen_eig_topk
from pencil.repair import coords, forecast, frontier, layer_basis, repair_delta, repair_layer

T = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))
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
        W_new, ratio = repair_layer(T(W0), T(dW), T(A_h), T(A_t), T(G_h), T(G_t), mu, damping=damping, gamma=gamma)
        np.testing.assert_allclose(W_new.numpy() - W0, ref, rtol=1e-8, atol=1e-10)
        assert ratio == pytest.approx(np.linalg.norm(ref - dW) / np.linalg.norm(dW), rel=1e-8)


def test_null_component_passes_through():
    A_h, A_t, G_h, G_t, dW, u, _ = _problem(null_in=True)
    b = layer_basis(T(A_h), T(A_t), T(G_h), T(G_t))
    assert b.inp.range_dim == A_h.shape[0] - 1
    out = repair_delta(T(dW), b, 0.3).numpy()
    un = u / np.linalg.norm(u)
    np.testing.assert_allclose(out @ un, dW @ un, rtol=1e-8, atol=1e-10)


def test_truncation_passes_discarded_pairs_through():
    """With only the top-k output pairs kept, M is 1 on the discarded rows (test_math, truncated form)."""
    A_h, A_t, G_h, G_t, dW, _, _ = _problem(null_in=False)
    k, mu = 2, 0.5
    f = (T(A_h), T(A_t), T(G_h), T(G_t))
    full, trunc = layer_basis(*f), layer_basis(*f, top_k_out=k)
    C = coords(T(dW), full)
    M = mu / (torch.outer(full.gamma, full.alpha) + mu)
    M[:-k] = 1.0  # gen_eig is ascending: the top-k are the last rows
    expected = dW - (full.out.X @ ((1 - M) * C) @ full.inp.X.T).numpy()
    np.testing.assert_allclose(repair_delta(T(dW), trunc, mu).numpy(), expected, rtol=1e-8, atol=1e-10)


def test_forecast_equals_pencil_sum_and_ignores_task_factors():
    A_h, A_t, G_h, G_t, dW, _, _ = _problem(null_in=True, null_out=True)
    S = forecast(T(dW), T(A_h), T(G_h))
    D = dW.reshape(-1, order="F")
    assert S == pytest.approx(0.5 * D @ np.kron(A_h, G_h) @ D, rel=1e-10)
    for gamma in (1e-3, 0.3):  # any task factors, any covariant damping
        b = layer_basis(T(A_h), T(A_t), T(G_h), T(G_t), gamma=gamma)
        C = coords(T(dW), b)
        assert S == pytest.approx(0.5 * float((torch.outer(b.gamma, b.alpha) * C**2).sum()), rel=1e-7)


def test_frontier_matches_repair():
    """Predicted S(mu) equals the forecast of the repaired update (full basis, no truncation)."""
    A_h, A_t, G_h, G_t, dW, _, _ = _problem(null_in=True)
    b = layer_basis(T(A_h), T(A_t), T(G_h), T(G_t))
    C = coords(T(dW), b)
    mus = [0.01, 0.1, 1.0]
    S, R = frontier(C, b.alpha, b.gamma, mus)
    for mu, s in zip(mus, S):
        assert forecast(repair_delta(T(dW), b, mu), T(A_h), T(G_h)) == pytest.approx(s, rel=1e-7)
    assert S == sorted(S) and R == sorted(R, reverse=True)


def test_gen_eig_normalisation_and_errors():
    A_h, A_t, *_ = _problem(null_in=True)
    p = gen_eig(T(A_h), T(A_t))
    np.testing.assert_allclose((p.X.T @ p.FX).numpy(), np.eye(p.X.shape[1]), atol=1e-8)
    np.testing.assert_allclose((p.X.T @ T(A_h) @ p.X).numpy(), np.diag(p.lam.numpy()), atol=1e-8)
    assert torch.all(p.lam[1:] >= p.lam[:-1])
    with pytest.raises(ValueError):
        gen_eig(T(A_h), T(A_t), "covariant", 0.0)
    with pytest.raises(ValueError):
        gen_eig(T(A_h), T(A_t), "bogus")


def test_dtype_roundtrip_bf16():
    A_h, A_t, G_h, G_t, dW, _, _ = _problem()
    W0 = torch.randn(dW.shape, dtype=torch.bfloat16)
    W_new, _ = repair_layer(W0, T(dW).to(torch.bfloat16), T(A_h), T(A_t), T(G_h), T(G_t), 0.5)
    assert W_new.dtype == torch.bfloat16 and W_new.shape == W0.shape


@pytest.mark.parametrize("damping,gamma", [("covariant", 1e-3), ("euclidean", 1e-2)])
def test_gen_eig_topk_matches_gen_eig_on_full_rank_factors(damping, gamma):
    n, k = 40, 6
    A_h, A_t = T(spd(n, 10)), T(spd(n))
    ref = gen_eig(A_h, A_t, damping, gamma, top_k=k)
    for dt in (torch.float64, torch.float32):
        p = gen_eig_topk(A_h, A_t, k, damping, gamma, jitter=0.0, eig_dtype=dt)
        tol = 1e-8 if dt == torch.float64 else 1e-4
        torch.testing.assert_close(p.lam, ref.lam, rtol=tol, atol=tol * float(ref.lam.max()))
        # same subspace: projectors onto the eigenvectors agree (signs are arbitrary)
        torch.testing.assert_close(p.X @ p.FX.T, ref.X @ ref.FX.T, rtol=0, atol=tol * 10)
        torch.testing.assert_close(p.X.T @ p.FX, torch.eye(k, dtype=torch.float64), rtol=0, atol=tol)
