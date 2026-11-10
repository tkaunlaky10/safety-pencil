"""LoRA fine-tune of θ0 on task T1: the drift we later forecast and repair (Prompt 3).

Saves the adapter (the repair needs ΔW = scaling · B @ A in fp32) at each configured fraction of training,
in outputs/ckpt/{run}/frac{25,50,75,100}/, with the pinned tokenizer and a RESEARCH_ARTIFACT.txt label.

Not SafeTune's PlainSFTTrainer, because it merges and discards the adapter. Encoding goes through
pencil.chat, like the factor passes and evaluation, so the template and the response-only loss match.

    python scripts/03_finetune.py --run A
    python scripts/03_finetune.py --run D            # 2% existing BeaverTails pairs mixed in
    python scripts/03_finetune.py --run A --dry-run
"""

from __future__ import annotations

import argparse
import math
import random

import torch
from torch.utils.checkpoint import checkpoint

from pencil.chat import collate
from pencil.datasets import contamination_count, load_split, mix_contamination
from pencil.delta import save_checkpoint
from pencil.modules import decoder, lm_logits
from pencil.results import write_result
from pencil.runtime import common_args, encode_pairs, load_model, load_tokenizer, make_batches, setup, task_rows

LABEL = "research artifact: unrepaired, do not deploy (drifted by fine-tuning; SafeTune licence)"


def response_loss(model, batch, device, chunk: int = 512):
    """(mean cross-entropy over response tokens, their count). The LM head runs on response rows only, in
    chunks recomputed during backward, so fp32 logits over the 262k vocabulary exist for `chunk` rows at a time."""
    ids, att, loss_mask = (batch[k].to(device) for k in ("input_ids", "attention_mask", "loss_mask"))
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = decoder(base)(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    rows = loss_mask[:, :-1]  # position t predicts token t + 1
    h, targets = hidden[:, :-1][rows], ids[:, 1:][rows]

    def ce_sum(h_chunk, t_chunk):
        return torch.nn.functional.cross_entropy(lm_logits(base, h_chunk).float(), t_chunk, reduction="sum")

    total = sum(
        checkpoint(ce_sum, h[i : i + chunk], targets[i : i + chunk], use_reentrant=False)
        for i in range(0, h.shape[0], chunk)
    )
    n = int(rows.sum())
    return total / max(n, 1), n


def length_bucketed_batches(encoded, batch_size: int, seed: int) -> list[list[int]]:
    """Shuffle, sort by length within windows of 32 batches to cut padding, then shuffle the batches."""
    order = list(range(len(encoded)))
    random.Random(seed).shuffle(order)
    window = batch_size * 32
    batches = []
    for i in range(0, len(order), window):
        chunk = sorted(order[i : i + window], key=lambda j: len(encoded[j].input_ids))
        batches += [chunk[k : k + batch_size] for k in range(0, len(chunk), batch_size)]
    random.Random(seed + 1).shuffle(batches)
    return batches


@torch.no_grad()
def eval_loss(model, batches, device) -> float:
    total, n = 0.0, 0
    for b in batches:
        loss, k = response_loss(model, b, device)
        total, n = total + float(loss) * k, n + k
    return total / max(n, 1)


def training_rows(ctx, run: dict) -> list[dict]:
    rows = task_rows(ctx, f"train_{run['size']}")
    if run.get("contamination", 0) > 0:
        k = contamination_count(len(rows), run["contamination"])
        rows = mix_contamination(
            rows, load_split(ctx.data_dir, "contamination", k) if k else [], run["contamination"], run["seed"]
        )
    return rows


def lora_model(ctx, lc: dict, grad_ckpt: bool):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        load_model(ctx.aligned, ctx),
        LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    if grad_ckpt and ctx.device.startswith("cuda"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model.train()


def warmup_linear_decay(optimizer, total_steps: int, warmup_frac: float = 0.03):
    warm = max(1, round(warmup_frac * total_steps))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / warm, max(0.0, (total_steps - s) / max(1, total_steps - warm)))
    )


def main(argv=None):
    p = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--run", required=True, help="A, B, C or D (configs/experiment.yaml runs)")
    p.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    args = p.parse_args(argv)
    ctx = setup(args)
    run, lc = ctx.cfg["runs"][args.run], ctx.cfg["lora"]
    torch.manual_seed(run["seed"])
    tok = load_tokenizer(ctx)

    rows = training_rows(ctx, run)
    train = encode_pairs(tok, [(r["prompt"], r["answer"]) for r in rows], lc["max_len"])
    heldout = make_batches(
        encode_pairs(tok, [(r["prompt"], r["answer"]) for r in task_rows(ctx, "heldout", 64)], lc["max_len"]),
        tok.pad_token_id,
        lc["batch_size"],
    )
    print(f"run {args.run}: {len(train)} training rows ({len(rows) - len(train)} skipped as too long)")

    model = lora_model(ctx, lc, grad_ckpt=not args.no_grad_ckpt)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lc["lr"], weight_decay=0.0)
    accum = lc["grad_accum"]
    steps_per_epoch = math.ceil(math.ceil(len(train) / lc["batch_size"]) / accum)  # last window may be partial
    total = steps_per_epoch * lc["epochs"]
    scheduler = warmup_linear_decay(optimizer, total)
    save_at = {max(1, round(f * total)): f for f in ctx.cfg["save_fractions"]} | {total: 1.0}
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if ctx.device.startswith("cuda")
        else torch.autocast("cpu", enabled=False)
    )
    log, step = {"train_loss": [], "eval_loss": {}, "saves": {}}, 0

    for epoch in range(lc["epochs"]):
        # a distinct order per (seed, epoch), so run A's epoch 2 doesn't replay run B's epoch 1
        batches = length_bucketed_batches(train, lc["batch_size"], 1000 * run["seed"] + epoch)
        for i, group in enumerate(batches):
            window = min(accum, len(batches) - (i // accum) * accum)
            with amp:
                loss, _ = response_loss(model, collate([train[j] for j in group], tok.pad_token_id), ctx.device)
            (loss / window).backward()
            if (i + 1) % accum and i + 1 < len(batches):
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            log["train_loss"].append(loss.item())
            if step in save_at:
                frac = save_at[step]
                path = save_checkpoint(model, tok, ctx.out("ckpt", args.run, f"frac{round(frac * 100)}"), LABEL)
                model.eval()
                with amp:
                    log["eval_loss"][str(frac)] = eval_loss(model, heldout, ctx.device)
                model.train()
                log["saves"][str(frac)] = {"step": step, "path": str(path), "elapsed_s": ctx.elapsed()}
                print(f"step {step}/{total}: saved {path.name}, held-out loss {log['eval_loss'][str(frac)]:.4f}")

    log.update(total_steps=total, n_rows=len(train), wall_clock_s=ctx.elapsed())
    path = write_result(
        f"finetune_{args.run}",
        log,
        config={"run": run, "lora": lc},
        seed=run["seed"],
        checkpoint=str(ctx.out("ckpt", args.run)),
        out_dir=ctx.out("results"),
    )
    print(f"done in {ctx.elapsed():.0f}s -> {path}")


if __name__ == "__main__":
    main()
