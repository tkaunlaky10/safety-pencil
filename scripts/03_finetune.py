"""LoRA fine-tune of θ0 on task T1: the drift we later forecast and repair (Prompt 3).

Keeps the adapter (ΔW = scaling · B @ A in fp32 is what the repair needs) and saves intermediate adapters at
the configured fractions of training: outputs/ckpt/{run}/frac{25,50,75,100}/. Each adapter folder also gets
the pinned tokenizer (so no adapter can bring a different chat template) and a RESEARCH_ARTIFACT.txt label.

Why not SafeTune's PlainSFTTrainer: it merges and discards the adapter. Why our own encoding rather than
SafeTune's tokenize_dataset: the factor passes, discovery and evaluation all use pencil.chat, so training
uses the identical template and response-only loss (and pads per batch, not to max_len).

    python scripts/03_finetune.py --run A
    python scripts/03_finetune.py --run D            # 2% BeaverTails contamination (existing pairs only)
    python scripts/03_finetune.py --run A --dry-run
"""
from __future__ import annotations

import argparse
import math
import random

import torch

from pencil.datasets import contamination_count, load_split, mix_contamination
from pencil.factors import _decoder, _logits
from pencil.results import write_result
from pencil.runtime import common_args, encode_pairs, load_model, load_tokenizer, make_batches, setup, task_rows
from pencil.chat import collate

LABEL = "research artifact: unrepaired, do not deploy (drifted by fine-tuning; SafeTune licence)"


def response_loss(model, batch, device, chunk: int = 512):
    """Mean token cross-entropy on response tokens. The LM head runs on those rows only, in chunks that are
    recomputed in the backward pass, so fp32 logits for 262k tokens never exist for more than `chunk` rows."""
    from torch.utils.checkpoint import checkpoint
    ids, att, lm = (batch[k].to(device) for k in ("input_ids", "attention_mask", "loss_mask"))
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = _decoder(base)(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    rows = lm[:, :-1]
    hr, tgt = h[:, :-1][rows], ids[:, 1:][rows]

    def ce_sum(hc, tc):
        return torch.nn.functional.cross_entropy(_logits(base, hc).float(), tc, reduction="sum")

    total = sum(checkpoint(ce_sum, hr[i:i + chunk], tgt[i:i + chunk], use_reentrant=False)
                for i in range(0, hr.shape[0], chunk))
    n = int(rows.sum())
    return total / max(n, 1), n


def bucketed(encoded, batch_size, seed):
    """Shuffle, then sort within windows of 32 batches by length to cut padding."""
    idx = list(range(len(encoded)))
    random.Random(seed).shuffle(idx)
    win = batch_size * 32
    out = []
    for i in range(0, len(idx), win):
        chunk = sorted(idx[i:i + win], key=lambda j: len(encoded[j].input_ids))
        out += [chunk[k:k + batch_size] for k in range(0, len(chunk), batch_size)]
    random.Random(seed + 1).shuffle(out)
    return out


@torch.no_grad()
def eval_loss(model, batches, device):
    tot, n = 0.0, 0
    for b in batches:
        loss, k = response_loss(model, b, device)
        tot += float(loss) * k
        n += k
    return tot / max(n, 1)


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--run", required=True, help="A, B, C or D (configs/experiment.yaml runs)")
    p.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    args = p.parse_args(argv)
    ctx = setup(args)
    cfg, run, lc = ctx.cfg, ctx.cfg["runs"][args.run], ctx.cfg["lora"]
    torch.manual_seed(run["seed"])
    tok = load_tokenizer(ctx)

    rows = task_rows(ctx, f"train_{run['size']}")
    if run.get("contamination", 0) > 0:
        k = contamination_count(len(rows), run["contamination"])
        contam = load_split(ctx.data_dir, "contamination", k) if k else []
        rows = mix_contamination(rows, contam, run["contamination"], run["seed"])
    enc = encode_pairs(tok, [(r["prompt"], r["answer"]) for r in rows], lc["max_len"])
    held = task_rows(ctx, "heldout", 64)
    held_b = make_batches(encode_pairs(tok, [(r["prompt"], r["answer"]) for r in held], lc["max_len"]),
                          tok.pad_token_id, lc["batch_size"])
    print(f"run {args.run}: {len(enc)} training rows ({len(rows) - len(enc)} skipped as too long)")

    from peft import LoraConfig, get_peft_model
    model = load_model(ctx.aligned, ctx)
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    if not args.no_grad_ckpt and ctx.device.startswith("cuda"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lc["lr"], weight_decay=0.0)
    ga = lc["grad_accum"]
    n_batches = math.ceil(len(enc) / lc["batch_size"])
    per_epoch = math.ceil(n_batches / ga)                # the last window of an epoch may be partial
    total = per_epoch * lc["epochs"]
    warm = max(1, round(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    saves = {max(1, round(f * total)): f for f in cfg["save_fractions"]}
    saves[total] = 1.0                                   # the final adapter is always frac100
    out_root = ctx.out("ckpt", args.run)
    log, step = {"train_loss": [], "eval_loss": {}, "saves": {}}, 0
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if ctx.device.startswith("cuda") else torch.autocast("cpu", enabled=False)

    def save(frac):
        d = out_root / f"frac{round(frac * 100)}"
        model.save_pretrained(d)
        tok.save_pretrained(d)
        (d / "RESEARCH_ARTIFACT.txt").write_text(LABEL + "\n")
        model.eval()
        with amp:
            log["eval_loss"][str(frac)] = eval_loss(model, held_b, ctx.device)
        model.train()
        log["saves"][str(frac)] = {"step": step, "path": str(d), "elapsed_s": ctx.elapsed()}
        print(f"step {step}/{total}: saved {d.name}, held-out loss {log['eval_loss'][str(frac)]:.4f}")

    for epoch in range(lc["epochs"]):
        # distinct data orders per (seed, epoch): run A's epoch 2 must not replay run B's epoch 1
        groups = bucketed(enc, lc["batch_size"], 1000 * run["seed"] + epoch)
        for i, group in enumerate(groups):
            window = min(ga, len(groups) - (i // ga) * ga)   # size of this accumulation window
            batch = collate([enc[j] for j in group], tok.pad_token_id)
            with amp:
                loss, _ = response_loss(model, batch, ctx.device)
            (loss / window).backward()
            if (i + 1) % ga and i + 1 < len(groups):
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log["train_loss"].append(loss.item())
            if step in saves:
                save(saves[step])
    log.update(total_steps=total, n_rows=len(enc), wall_clock_s=ctx.elapsed())
    path = write_result(f"finetune_{args.run}", log, config={"run": run, "lora": lc}, seed=run["seed"],
                        checkpoint=str(out_root), out_dir=ctx.out("results"))
    print(f"done in {ctx.elapsed():.0f}s -> {path}")


if __name__ == "__main__":
    main()
