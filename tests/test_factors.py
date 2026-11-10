"""KFACCollector on a tiny random Cohere2 model (CPU, seconds). Prompt 0 step 8 checks (a)-(e)."""

import numpy as np
import pytest
import torch

from pencil.chat import collate, encode_chat, user_content_offset
from pencil.factor_store import FactorStore, LazyFactors, known_null_vectors, null_floors
from pencil.factors import KFACCollector, collect_factors, factor_bytes
from pencil.geneig import auto_rel_tol, gen_eig, noise_floor
from pencil.modules import input_factor_key, target_modules
from pencil.testing import tiny_cohere2, tiny_tokenizer

pytestmark = pytest.mark.slow

PAIRS = [
    ("How do I bake bread?", "Mix flour and water."),
    ("कैसे हो?", "मैं ठीक हूँ।"),
    ("Tell me a joke", "No."),
    ("Write a haiku about rain", "Rain on the tin roof"),
]


@pytest.fixture(scope="module")
def tokenizer():
    return tiny_tokenizer()


@pytest.fixture(scope="module")
def batch(tokenizer):
    off = user_content_offset(tokenizer)
    enc = [encode_chat(tokenizer, p, r, user_offset=off) for p, r in PAIRS]
    return collate(enc, tokenizer.pad_token_id)


def _capture(model, names):
    """Independent hooks: store each module's input and its output gradient."""
    ins, grads, handles = {}, {}, []
    mods = dict(model.named_modules())
    for n in names:

        def fwd(m, i, o, n=n):
            ins[n] = i[0].detach().clone()
            o.register_hook(lambda g, n=n: grads.__setitem__(n, g.detach().clone()))

        handles.append(mods[n].register_forward_hook(fwd))
    return ins, grads, handles


def test_matches_hand_computation_and_respects_masks(batch):
    model = tiny_cohere2()
    names = list(target_modules(model))
    with KFACCollector(model, diag=True, diag_row_set="all") as collector:
        ins, grads, handles = _capture(model, names)
        collector.collect(batch)
        for h in handles:
            h.remove()
    f = collector.finalize()
    masks = {"loss": batch["loss_mask"], "all": batch["rows_all"], "full": batch["rows_full"]}
    n_loss = int(batch["loss_mask"].sum())
    B = batch["input_ids"].shape[0]
    assert f.n_loss == n_loss and f.n_rows == {**{rs: int(m.sum()) for rs, m in masks.items()}, "full_reduce": B}
    assert not batch["rows_all"][0, : user_content_offset(tiny_tokenizer())].any()  # preamble excluded
    assert batch["rows_full"][0, 0]  # ... but not from "full"
    for n in names:
        for rs, m in masks.items():
            a, g = ins[n][m].double(), grads[n][m].double()
            A, G = (t.double() for t in f.pair(n, rs))
            A_ref, G_ref = a.T @ a / m.sum(), g.T @ g / n_loss  # fp32 accumulation vs float64 here
            torch.testing.assert_close(A, A_ref, rtol=1e-4, atol=1e-6 * float(A_ref.abs().max()))
            torch.testing.assert_close(G, G_ref, rtol=1e-4, atol=1e-6 * float(G_ref.abs().max()))
        # K-FAC-reduce: per sequence, the mean input and the summed output gradient over the "full" rows
        m = masks["full"].double()[..., None]
        a_bar = (ins[n].double() * m).sum(1) / m.sum(1)
        g_sum = (grads[n].double() * m).sum(1)
        A, G = (t.double() for t in f.pair(n, "full_reduce"))
        A_ref, G_ref = a_bar.T @ a_bar / B, g_sum.T @ g_sum / n_loss
        g_scale = float(f.pair(n, "full")[1].abs().max())  # k_proj: Σ_s ∂L/∂k_s = 0, so G_reduce ~ 0
        torch.testing.assert_close(A, A_ref, rtol=1e-4, atol=1e-6 * float(A_ref.abs().max()))
        torch.testing.assert_close(G, G_ref, rtol=1e-4, atol=1e-6 * g_scale)
        m = masks["all"].double()[..., None]  # per-sequence gradient, then squared
        D_ref = sum(((grads[n][b].double() * m[b]).T @ (ins[n][b].double() * m[b])) ** 2 for b in range(B)) / n_loss
        torch.testing.assert_close(f.D[n].double(), D_ref, rtol=1e-4, atol=1e-6 * float(D_ref.abs().max()))
    assert all(p.grad is None for p in model.parameters())  # no weight gradients


def test_parallel_block_shares_one_input_factor(batch):
    model = tiny_cohere2()
    f = collect_factors(model, [batch])
    L = len(model.model.layers)
    assert len(f.A) == 3 * L  # shared input, o_proj, down_proj per layer
    for l in range(L):
        key = f"model.layers.{l}.input"
        for leaf in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"):
            assert f.a_key[f"model.layers.{l}.{leaf}"] == key
    assert input_factor_key("model.layers.2.self_attn.o_proj", True) == "model.layers.2.self_attn.o_proj"


def test_non_parallel_block_is_caught(batch):
    """Forcing a shared input factor on a sequential block (Qwen2) must fail loudly."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=len(tiny_tokenizer()),
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    with pytest.raises(RuntimeError, match="not parallel"):
        collect_factors(model, [batch], share_input_factor=True)
    f = collect_factors(model, [batch])  # auto-detects: not shared
    assert len(set(f.a_key.values())) == len(f.a_key)


def test_layernorm_null_vector_and_range_restriction(batch):
    """The shared input factor has the null vector 1/w. In fp32 accumulation it is null only up to the
    numerical floor, which noise_floor measures; with rel_tol above it the range drops exactly that one
    direction, and gen_eig succeeds even on a rank-deficient task side."""
    model = tiny_cohere2()
    f32 = collect_factors(model, [batch])
    f64 = collect_factors(model, [batch], accum_dtype=torch.float64)
    for l, layer in enumerate(model.model.layers):
        key = f"model.layers.{l}.input"
        u = 1.0 / layer.input_layernorm.weight.detach().double()
        floor64 = noise_floor(f64.A[key]["all"], u)
        floor32 = noise_floor(f32.A[key]["all"], u)
        assert floor64 < 1e-12 and floor32 < 1e-5
        for f in (f32, f64):
            A_h, A_t = f.A[key]["all"], f.A[key]["loss"]  # "loss" has fewer rows than dims: rank-deficient
            rel_tol = auto_rel_tol([noise_floor(A_h + A_t, u)])
            w = torch.linalg.eigvalsh(f64.A[key]["all"] + f64.A[key]["loss"])
            rank = int((w > 1e-10 * w.max()).sum())  # layer 0 sees one row per distinct token
            for F_h, F_t in ((A_h, A_t), (A_t, A_h)):
                p = gen_eig(F_h, F_t, rel_tol=rel_tol)
                assert p.range_dim <= min(rank, A_h.shape[0] - 1)
                if f is f64:
                    assert p.range_dim == rank
                assert (
                    torch.isfinite(p.lam).all() and p.lam.max() <= 1e3 * 1.01
                )  # covariant cap 1/gamma (up to rounding)
                n = p.X.shape[1]
                torch.testing.assert_close(p.X.T @ p.FX, torch.eye(n, dtype=torch.float64), atol=1e-6, rtol=0)


def test_projecting_known_nulls_makes_them_exact(batch):
    """Without projection, eigenvectors of the near-zero cluster mix u into the kept range at ~1e-5 even in
    float64 (and ~1e-2 in fp32). After projecting them out the kept basis is orthogonal to u to ~1e-10, so the
    subtraction-form repair leaves ΔW u untouched."""
    model = tiny_cohere2()
    for acc in (torch.float32, torch.float64):
        f = collect_factors(model, [batch], accum_dtype=acc)
        nulls = known_null_vectors(model, f)
        assert len(nulls) == 3 * len(model.model.layers)  # shared A per layer, o and down G per layer
        projected = LazyFactors([f], nulls=nulls)
        tol = auto_rel_tol(null_floors(f, nulls).values())
        for (side, key), u in nulls.items():
            un = u / u.norm()
            p = gen_eig(projected.get(side, key, "all"), projected.get(side, key, "loss"), rel_tol=tol)
            assert float((p.X.T @ un.to(p.X)).abs().max() / p.X.abs().max()) < 1e-9, (side, key, acc)


def test_sampled_labels_are_unbiased_for_G():
    """Mean of many one-pass estimates vs the exact expectation, enumerating a 6-token vocabulary."""
    torch.manual_seed(0)
    model = tiny_cohere2(vocab_size=6, n_layers=2, hidden=16, inter=24, n_heads=2, n_kv=1, seed=3)
    names = ["model.layers.0.self_attn.v_proj", "model.layers.1.mlp.down_proj"]
    T, loss_pos, rows = 10, [5, 6, 7, 8], list(range(2, 9))
    ids = torch.randint(3, 6, (1, T))
    lm, ra = torch.zeros(1, T, dtype=torch.bool), torch.zeros(1, T, dtype=torch.bool)
    lm[0, loss_pos], ra[0, rows] = True, True

    # exact: sum over loss positions t and labels v of p_t(v) * sum_rows J J^T, per loss token
    ins, grads, handles = _capture(model, names)
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    logp = model(inputs_embeds=emb).logits.float().log_softmax(-1)[0]
    exact = {
        n: torch.zeros(model.get_submodule(n).out_features, model.get_submodule(n).out_features, dtype=torch.float64)
        for n in names
    }
    for t in loss_pos:
        for v in range(6):
            grads.clear()
            logp[t, v].backward(retain_graph=True)
            for n in names:
                g = grads[n][0, rows].double()
                exact[n] += logp[t, v].exp().item() * g.T @ g
    for h in handles:
        h.remove()
    exact = {n: e / len(loss_pos) for n, e in exact.items()}

    B, n_batches = 400, 10
    batch = {
        "input_ids": ids.repeat(B, 1),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "loss_mask": lm.repeat(B, 1),
        "rows_all": ra.repeat(B, 1),
    }
    est = {n: [] for n in names}
    for k in range(n_batches):
        f = collect_factors(model, [batch], layers=[0, 1], row_sets=("all",), seed=k)
        for n in names:
            est[n].append(f.G[n]["all"])
    for n in names:
        m = torch.stack(est[n]).mean(0)
        err = (m - exact[n]).norm() / exact[n].norm()
        assert err < 0.05, (n, float(err))
        # greedy labels instead of sampled ones would be biased: compare against a clearly wrong target
        assert (exact[n] * 2 - m).norm() / exact[n].norm() > 0.5


def test_save_load_chunks_and_pool(tmp_path, batch):
    model = tiny_cohere2()
    f1 = collect_factors(model, [batch], layers=[0, 1], seed=1, meta={"lang": "en"})
    f2 = collect_factors(model, [batch], layers=[2, 3], seed=1)
    f1.save(tmp_path / "en" / "layers00-01.safetensors")
    f2.save(tmp_path / "en" / "layers02-03.safetensors")
    store = FactorStore.from_dir(tmp_path / "en")
    assert store.meta["lang"] == "en" and store.n_loss == f1.n_loss and len(store.modules) == 7 * 4
    for n in f1.modules:
        for rs in f1.row_sets:
            for x, y in zip(f1.pair(n, rs), store.pair(n, rs)):
                torch.testing.assert_close(x, y)
    full = collect_factors(model, [batch], seed=1)
    for n in full.modules:  # chunked == all layers at once (same seed)
        torch.testing.assert_close(store.pair(n, "loss")[1], full.pair(n, "loss")[1])
    other = collect_factors(model, [batch], layers=[0, 1], seed=2)
    pooled = LazyFactors([f1, other], [3.0, 1.0])
    n = f1.modules[0]
    torch.testing.assert_close(
        pooled.get("G", n, "all"), 0.75 * f1.G[n]["all"].double() + 0.25 * other.G[n]["all"].double()
    )
    assert pooled.n_loss == f1.n_loss + other.n_loss


def test_memory_plan_matches_implementation_md():
    d = 8192
    n = factor_bytes(2048, d, 16, 4, 128, n_row_sets=1, bytes_per=1)
    assert n == 5 * 2048**2 + 2 * 512**2 + 3 * d**2
    assert np.isclose(factor_bytes(2048, d, 16, 4, 128, n_row_sets=1) / 1e9, 0.89, atol=0.01)


def test_truncation_keeps_turn_markers_and_response(tokenizer):
    from pencil.chat import prompt_suffix_len

    full = encode_chat(tokenizer, "abcdefghijklmnopqrstuvwxyz", "OK")
    short = encode_chat(tokenizer, "abcdefghijklmnopqrstuvwxyz", "OK", max_len=len(full.input_ids) - 5)
    suf = prompt_suffix_len(tokenizer)
    assert suf == len(tokenizer("<EOT><BOT>", add_special_tokens=False)["input_ids"])
    assert short.input_ids[short.response_start - suf :] == full.input_ids[full.response_start - suf :]
    assert (
        tokenizer.decode(short.input_ids[short.user_start : short.response_start]) == "abcdefghijklmnopqrstu<EOT><BOT>"
    )
    assert len(short.input_ids) == len(full.input_ids) - 5
