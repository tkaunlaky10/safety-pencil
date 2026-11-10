"""scripts/00_model_facts.py measures architecture facts by running a tiny copy of the model. These tests check
that each measurement tells the two kinds of decoder apart: Tiny Aya's parallel block (Cohere2) and a
sequential block with a V bias (Llama)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "00_model_facts.py"


@pytest.fixture(scope="module")
def model_facts():
    spec = importlib.util.spec_from_file_location("model_facts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_cohere2_config():
    from pencil.testing import tiny_cohere2

    return tiny_cohere2(n_heads=8, n_kv=2).config


def _tiny_llama_config():
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=200,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        attention_bias=True,
    )


@pytest.mark.slow
def test_parallel_block_with_plain_v_passes_every_check(model_facts):
    model = model_facts.build_tiny_model(_tiny_cohere2_config())
    input_ids = model_facts.random_input(model)

    assert model_facts.check_shared_input(model, input_ids)["shared"]
    assert model_facts.check_null_vector(model, input_ids)["orthogonal"]
    assert model_facts.check_plain_v(model, input_ids)["linear"]
    reads = model_facts.measure_kv_grouping(model, input_ids)
    assert reads == [0, 0, 0, 0, 1, 1, 1, 1]
    assert model_facts.describe_grouping(reads, n_rep=4) == "query head h reads KV head h // 4"


@pytest.mark.slow
def test_sequential_block_is_not_reported_as_parallel(model_facts):
    model = model_facts.build_tiny_model(_tiny_llama_config())
    input_ids = model_facts.random_input(model)

    assert not model_facts.check_shared_input(model, input_ids)["shared"]  # gate/up read a second norm's output
    assert not model_facts.check_null_vector(model, input_ids)["orthogonal"]  # RMSNorm does not remove the mean
    assert model.model.layers[0].self_attn.v_proj.bias is not None
    assert model_facts.check_plain_v(model, input_ids)["linear"]  # with the bias zeroed, V is still linear
    assert model_facts.measure_kv_grouping(model, input_ids) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_an_unexpected_grouping_is_reported_as_measured(model_facts):
    description = model_facts.describe_grouping([0, 1, 0, 1], n_rep=2)
    assert description == "measured: query head h reads KV head [0, 1, 0, 1][h]"
