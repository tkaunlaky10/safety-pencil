"""Publish WiSE-FT on a run as a LoRA adapter: θ = θ0 + (1 − α)ΔW, made by scaling every lora_B by (1 − α).

    python jobs/publish_wiseft.py --run A --alpha 0.5 --repo tkaunlaky10/safety-pencil-runA-wiseft-0.5 --dry-run
    python jobs/publish_wiseft.py --run A --alpha 0.5 --repo tkaunlaky10/safety-pencil-runA-wiseft-0.5

ΔW = (lora_alpha / r) · B · A is linear in B, so (1 − α)·B gives exactly (1 − α)·ΔW. That holds only for a plain
LoRA adapter, which `check_plain_lora` enforces. The script writes outputs/release/run{R}_wiseft_{α}/
(the scaled adapter, the unchanged adapter_config.json and the model card), checks every module's ΔW against
(1 − α)·ΔW_old with pencil.delta.lora_delta, and pushes a public repo only without --dry-run.

The adapter is read from outputs/ckpt/{run}/frac100 (or --adapter), else downloaded from the public
tkaunlaky10/safety-pencil-lora-run{run}. The card's numbers come from the test result JSONs in outputs/results,
fetched from the public tkaunlaky10/safety-pencil-results when missing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_cards import wiseft_card  # noqa: E402

from pencil.delta import lora_delta  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OWNER = "tkaunlaky10"
RESULTS_REPO = f"{OWNER}/safety-pencil-results"
LANGS = ("en", "hi", "hinglish")
MAX_REL_ERROR = 1e-6
# the card's columns: the result JSON behind each (outputs/results/<name>.json)
SOURCES = {
    "theta0": "test_theta0",
    "theta1": "test_run{run}_frac100",
    "wiseft": "test_frontier_wiseft{alpha:g}_run{run}",
}


def find_adapter(run: str, adapter: str | None) -> Path:
    """The run's final adapter folder: --adapter, else outputs/ckpt/{run}/frac100, else the public model repo."""
    if adapter:
        return Path(adapter)
    local = REPO_ROOT / "outputs" / "ckpt" / run / "frac100"
    if (local / "adapter_model.safetensors").exists():
        return local
    folder = snapshot_download(f"{OWNER}/safety-pencil-lora-run{run}", allow_patterns=["frac100/*"])
    return Path(folder) / "frac100"


def check_plain_lora(adapter: Path) -> None:
    """Stop unless scaling lora_B scales the whole update: no DoRA, no modules_to_save, no biases, and no
    trained tensors besides lora_A and lora_B. (rsLoRA is fine: its scaling is still linear in B.)"""
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    problems = []
    if cfg.get("use_dora"):
        problems.append("use_dora is true")
    if cfg.get("modules_to_save"):
        problems.append(f"modules_to_save = {cfg['modules_to_save']}")
    if cfg.get("bias", "none") != "none" or cfg.get("lora_bias"):
        problems.append(f"bias = {cfg.get('bias')}, lora_bias = {cfg.get('lora_bias')}")
    if cfg.get("trainable_token_indices"):
        problems.append("trainable_token_indices is set")
    with safe_open(str(adapter / "adapter_model.safetensors"), "pt") as tensors:
        keys = tensors.keys()  # a safe_open handle is not iterable itself
    other = [key for key in keys if ".lora_A." not in key and ".lora_B." not in key]
    if other:
        problems.append(f"{len(other)} tensors other than lora_A/lora_B, e.g. {other[0]}")
    if problems:
        raise SystemExit(f"{adapter}: scaling lora_B is not exact: " + "; ".join(problems))


def scale_lora_B(state_dict: dict[str, torch.Tensor], factor: float) -> dict[str, torch.Tensor]:
    """Every lora_B times `factor` (in fp32, cast back to its dtype); every other tensor unchanged."""
    scaled = {}
    for key, tensor in state_dict.items():
        if ".lora_B." in key:
            scaled[key] = (tensor.float() * factor).to(tensor.dtype)
        else:
            scaled[key] = tensor
    return scaled


def verify(old_adapter: Path, new_adapter: Path, factor: float) -> dict[str, float]:
    """{module: max|ΔW_new − factor·ΔW_old| / max|ΔW_old|}, one dense module at a time; stops above 1e-6."""
    old = lora_delta(old_adapter, lowrank=True)
    new = lora_delta(new_adapter, lowrank=True)
    if set(old) != set(new):
        raise SystemExit("the scaled adapter has different modules")
    errors = {}
    for module in old:
        dW_old = old[module].dense(torch.float64)
        dW_new = new[module].dense(torch.float64)
        errors[module] = float((dW_new - factor * dW_old).abs().max() / dW_old.abs().max())
    worst = max(errors.values())
    if worst > MAX_REL_ERROR:
        raise SystemExit(f"scaled ΔW is off by {worst:.2e} (limit {MAX_REL_ERROR:g})")
    return errors


def card_numbers(run: str, alpha: float, results_dir: Path) -> dict:
    """Per column (theta0, theta1, wiseft): the source file and its test numbers in percent, by language."""
    numbers = {}
    for column, pattern in SOURCES.items():
        name = pattern.format(run=run, alpha=alpha)
        path = results_dir / f"{name}.json"
        if not path.exists():
            path = Path(hf_hub_download(RESULTS_REPO, f"outputs/results/{name}.json", repo_type="dataset"))
        result = json.loads(path.read_text())["result"]
        numbers[column] = {"source": f"{name}.json", "task_nll": result["task"]["nll_per_token"]}
        for lang in LANGS:
            metrics = result["metrics"][lang]
            refusal = metrics["harmful"]["refusal"]["p"]
            over_refusal = metrics["xstest"]["over_refusal"]["p"]
            numbers[column][lang] = {
                "unsafe": 100 * metrics["harmful"]["unsafe"]["p"],
                "over_refusal": 100 * over_refusal,
                "refusal": 100 * refusal,
                "youden_j": 100 * (refusal - over_refusal),
            }
            if "mmlu" in metrics:
                numbers[column][lang]["mmlu"] = 100 * metrics["mmlu"]["accuracy"]["p"]
    return numbers


def write_release(adapter: Path, out: Path, alpha: float, card: str) -> None:
    """out/ with the scaled adapter, the unchanged adapter_config.json and README.md."""
    out.mkdir(parents=True, exist_ok=True)
    source = adapter / "adapter_model.safetensors"
    with safe_open(str(source), "pt") as tensors:
        metadata = tensors.metadata()
    save_file(scale_lora_B(load_file(str(source)), 1.0 - alpha), str(out / "adapter_model.safetensors"), metadata)
    shutil.copyfile(adapter / "adapter_config.json", out / "adapter_config.json")
    (out / "README.md").write_text(card)


def push(api: HfApi, repo: str, out: Path) -> None:
    """Create the model repo as public (not gated) and upload the release folder in one commit."""
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(out), commit_message="WiSE-FT adapter release")
    print(f"published https://huggingface.co/{repo}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="A")
    parser.add_argument("--alpha", type=float, required=True, help="pull toward θ0, in (0, 1)")
    parser.add_argument("--repo", required=True, help="the public model repo to create")
    parser.add_argument("--adapter", help="the run's final adapter folder (default: found automatically)")
    parser.add_argument("--results", default=str(REPO_ROOT / "outputs" / "results"), help="result JSON folder")
    parser.add_argument("--out", help="release folder (default outputs/release/run{R}_wiseft_{α})")
    parser.add_argument("--dry-run", action="store_true", help="write and check the release folder, push nothing")
    args = parser.parse_args(argv)
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1)")

    adapter = find_adapter(args.run, args.adapter)
    check_plain_lora(adapter)
    out = Path(args.out or REPO_ROOT / "outputs" / "release" / f"run{args.run}_wiseft_{args.alpha:g}")
    numbers = card_numbers(args.run, args.alpha, Path(args.results))
    write_release(adapter, out, args.alpha, wiseft_card(args.run, args.alpha, args.repo, numbers))

    errors = verify(adapter, out, 1.0 - args.alpha)
    for module, error in errors.items():
        print(f"{module}: max|ΔW_new − {1 - args.alpha:g}·ΔW_old| / max|ΔW_old| = {error:.1e}")
    print(f"{len(errors)} modules, worst {max(errors.values()):.1e} (limit {MAX_REL_ERROR:g})")
    for name in sorted(p.name for p in out.iterdir()):
        print(f"   {name}  ({(out / name).stat().st_size / 1e6:.2f} MB)")

    if args.dry_run:
        print(f"\ndry run: {out} written; nothing pushed to {args.repo}")
        return 0
    push(HfApi(), args.repo, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
