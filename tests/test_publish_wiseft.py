"""jobs/publish_wiseft.py on a tiny synthetic adapter: scaling lora_B gives exactly (1 − α)·ΔW, the card carries
the licence, base model and warning, and a dry run makes no network calls."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pencil.delta import lora_delta

JOBS = Path(__file__).resolve().parents[1] / "jobs"
LANGS = ("en", "hi", "hinglish")


@pytest.fixture
def wiseft():
    spec = importlib.util.spec_from_file_location("jobs_publish_wiseft", JOBS / "publish_wiseft.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["jobs_publish_wiseft"] = module
    spec.loader.exec_module(module)
    return module


def _adapter(folder: Path, dtype=torch.float32, **cfg_overrides) -> Path:
    """A two-module LoRA adapter (r 2, lora_alpha 4) with random weights."""
    folder.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for module in ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"):
        prefix = f"base_model.model.{module}"
        tensors[f"{prefix}.lora_A.weight"] = torch.randn(2, 8, generator=g).to(dtype)
        tensors[f"{prefix}.lora_B.weight"] = torch.randn(6, 2, generator=g).to(dtype)
    save_file(tensors, str(folder / "adapter_model.safetensors"), {"format": "pt"})
    cfg = {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "use_dora": False, "bias": "none", "modules_to_save": None}
    (folder / "adapter_config.json").write_text(json.dumps({**cfg, **cfg_overrides}))
    return folder


def _results(folder: Path) -> Path:
    """The three test result JSONs the card reads, with made-up rates."""
    folder.mkdir()
    for index, name in enumerate(("test_theta0", "test_runA_frac100", "test_frontier_wiseft0.5_runA")):
        metrics = {}
        for lang in LANGS:
            metrics[lang] = {
                "harmful": {"unsafe": {"p": 0.05 + index / 100}, "refusal": {"p": 0.6}},
                "xstest": {"over_refusal": {"p": 0.1}},
            }
            if lang != "hinglish":
                metrics[lang]["mmlu"] = {"accuracy": {"p": 0.55}}
        result = {"metrics": metrics, "task": {"nll_per_token": 2.0 - index / 10}}
        (folder / f"{name}.json").write_text(json.dumps({"result": result}))
    return folder


@pytest.mark.parametrize("alpha", [0.5, 0.3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scaled_B_reproduces_the_scaled_update(wiseft, tmp_path, alpha, dtype):
    adapter = _adapter(tmp_path / "adapter", dtype)
    out = tmp_path / "release"
    wiseft.write_release(adapter, out, alpha, "# card")
    tolerance = 1e-6 if dtype == torch.float32 else 1e-2  # a bf16 B is rounded after scaling
    old, new = lora_delta(adapter), lora_delta(out)
    for module in old:
        assert torch.allclose(new[module], (1 - alpha) * old[module], rtol=0, atol=tolerance)
    if dtype == torch.float32:
        assert max(wiseft.verify(adapter, out, 1 - alpha).values()) <= 1e-6
    assert json.loads((out / "adapter_config.json").read_text()) == json.loads(
        (adapter / "adapter_config.json").read_text()
    )


def test_scaling_keeps_the_dtype_and_leaves_lora_A_alone(wiseft):
    state = {
        "m.lora_A.weight": torch.ones(2, 3, dtype=torch.bfloat16),
        "m.lora_B.weight": torch.ones(3, 2, dtype=torch.bfloat16),
    }
    scaled = wiseft.scale_lora_B(state, 0.5)
    assert scaled["m.lora_B.weight"].dtype == torch.bfloat16 and torch.all(scaled["m.lora_B.weight"] == 0.5)
    assert scaled["m.lora_A.weight"] is state["m.lora_A.weight"]


@pytest.mark.parametrize("override", [{"use_dora": True}, {"modules_to_save": ["lm_head"]}, {"bias": "all"}])
def test_adapters_where_scaling_B_is_not_exact_are_refused(wiseft, tmp_path, override):
    with pytest.raises(SystemExit):
        wiseft.check_plain_lora(_adapter(tmp_path / "adapter", **override))


def test_card_has_licence_base_model_warning_and_the_numbers(wiseft, tmp_path):
    numbers = wiseft.card_numbers("A", 0.5, _results(tmp_path / "results"))
    card = wiseft.wiseft_card("A", 0.5, "tkaunlaky10/safety-pencil-runA-wiseft-0.5", numbers)
    assert "license: cc-by-nc-4.0" in card and "base_model: CohereLabs/tiny-aya-global" in card
    assert "not certified" in card and "WiSE-FT" in card
    assert "test_frontier_wiseft0.5_runA.json" in card
    assert "| Harmful compliance, % | English | 5.0 | 6.0 | 7.0 |" in card


def test_dry_run_makes_no_network_calls(wiseft, tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network call in a dry run")

    for name in ("HfApi", "hf_hub_download", "snapshot_download"):
        monkeypatch.setattr(wiseft, name, no_network)
    adapter = _adapter(tmp_path / "adapter")
    results = _results(tmp_path / "results")
    out = tmp_path / "release"
    argv = ["--run", "A", "--alpha", "0.5", "--repo", "x/y", "--adapter", str(adapter), "--results", str(results)]
    assert wiseft.main([*argv, "--out", str(out), "--dry-run"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["README.md", "adapter_config.json", "adapter_model.safetensors"]
