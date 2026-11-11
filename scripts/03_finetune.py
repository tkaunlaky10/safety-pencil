"""LoRA fine-tune of θ0 on task T1: the drift we later forecast and repair (Prompt 3).

Saves the adapter (the repair needs ΔW = scaling · B @ A in fp32) at each configured fraction of training,
in outputs/ckpt/{run}/frac{25,50,75,100}/, with the pinned tokenizer and a RESEARCH_ARTIFACT.txt label.

Not SafeTune's PlainSFTTrainer, because it merges and discards the adapter. Encoding goes through
pencil.chat, like the factor passes and evaluation, so the template and the response-only loss match.

    python scripts/03_finetune.py --run A
    python scripts/03_finetune.py --run D            # 2% existing BeaverTails pairs mixed in
    python scripts/03_finetune.py --run A --dry-run
    python scripts/03_finetune.py --run A --max-steps 20   # timing run: 20 optimizer steps, nothing saved
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch
from torch.utils.checkpoint import checkpoint

from pencil.chat import collate
from pencil.datasets import contamination_count, load_split, mix_contamination
from pencil.delta import save_checkpoint
from pencil.modules import decoder, lm_logits
from pencil.results import write_result
from pencil.runtime import common_args, encode_pairs, load_model, load_tokenizer, make_batches, setup, task_rows

LABEL = "research artifact: unrepaired, do not deploy (drifted by fine-tuning; SafeTune licence)"

LM_HEAD_CHUNK = 512  # response rows per LM-head chunk in the loss
BUCKET_WINDOW_BATCHES = 32  # length bucketing sorts rows within windows of this many batches
N_HELDOUT_ROWS = 64  # held-out task rows scored at every save
MAX_GRAD_NORM = 1.0


def _summed_cross_entropy(base_model, hidden_chunk, target_chunk):
    """Cross-entropy of the fp32 next-token logits for these hidden rows, summed over the rows."""
    logits = lm_logits(base_model, hidden_chunk).float()
    return torch.nn.functional.cross_entropy(logits, target_chunk, reduction="sum")


def response_loss(model, batch, device, chunk_rows: int = LM_HEAD_CHUNK):
    """(mean cross-entropy over response tokens, their count). The LM head runs on response rows only, in
    chunks recomputed during backward, so fp32 logits over the 262k vocabulary exist for `chunk_rows` rows at a time."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model  # unwrap PEFT
    outputs = decoder(base_model)(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    predicts_response = loss_mask[:, :-1]  # position t predicts token t + 1
    response_hidden = outputs.last_hidden_state[:, :-1][predicts_response]
    targets = input_ids[:, 1:][predicts_response]
    total = 0
    for start in range(0, response_hidden.shape[0], chunk_rows):
        end = start + chunk_rows
        chunk_loss = checkpoint(
            _summed_cross_entropy, base_model, response_hidden[start:end], targets[start:end], use_reentrant=False
        )
        total = total + chunk_loss
    n_tokens = int(predicts_response.sum())
    return total / max(n_tokens, 1), n_tokens


def length_bucketed_batches(encoded, batch_size: int, seed: int) -> list[list[int]]:
    """Batches of row indices: shuffle, sort by length within windows of 32 batches to cut padding, then
    shuffle the batches."""
    order = list(range(len(encoded)))
    random.Random(seed).shuffle(order)
    lengths = [len(item.input_ids) for item in encoded]
    window = batch_size * BUCKET_WINDOW_BATCHES
    batches = []
    for window_start in range(0, len(order), window):
        window_rows = order[window_start : window_start + window]
        window_rows = sorted(window_rows, key=lambda index: lengths[index])
        for batch_start in range(0, len(window_rows), batch_size):
            batches.append(window_rows[batch_start : batch_start + batch_size])
    random.Random(seed + 1).shuffle(batches)
    return batches


@torch.no_grad()
def eval_loss(model, batches, device) -> float:
    """Mean cross-entropy per response token over all `batches`."""
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        loss, n_tokens = response_loss(model, batch, device)
        total_loss += float(loss) * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def training_rows(ctx, run: dict) -> list[dict]:
    """The run's task training rows, with existing BeaverTails pairs mixed in if the run has contamination."""
    rows = task_rows(ctx, f"train_{run['size']}")
    fraction = run.get("contamination", 0)
    if fraction > 0:
        n_contaminated = contamination_count(len(rows), fraction)
        contamination = []
        if n_contaminated:
            contamination = load_split(ctx.data_dir, "contamination", n_contaminated)
        rows = mix_contamination(rows, contamination, fraction, run["seed"])
    return rows


def lora_model(ctx, lora_cfg: dict, grad_ckpt: bool):
    """θ0 wrapped with fresh LoRA adapters, in training mode; gradient checkpointing on CUDA if `grad_ckpt`."""
    from peft import LoraConfig, get_peft_model

    base_model = load_model(ctx.aligned, ctx)
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    if grad_ckpt and ctx.device.startswith("cuda"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model.train()


def warmup_linear_decay(optimizer, total_steps: int, warmup_frac: float = 0.03):
    """Learning-rate schedule: linear warm-up over the first `warmup_frac` of steps, then linear decay to 0."""
    warmup_steps = max(1, round(warmup_frac * total_steps))
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_factor(step: int) -> float:
        warmup = (step + 1) / warmup_steps
        decay = max(0.0, (total_steps - step) / decay_steps)
        return min(warmup, decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def save_steps(total_steps: int, fractions: list[float]) -> dict[int, float]:
    """Optimizer step -> fraction of training at which to save the adapter; the last step always saves 1.0."""
    steps = {}
    for fraction in fractions:
        steps[max(1, round(fraction * total_steps))] = fraction
    steps[total_steps] = 1.0
    return steps


def autocast_context(device: str):
    """bfloat16 autocast on CUDA; a disabled autocast (plain float32) on CPU."""
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def encode_rows(tokenizer, rows: list[dict], max_len: int) -> list:
    """Template each row's (prompt, answer); rows whose answer alone exceeds max_len are skipped."""
    pairs = [(row["prompt"], row["answer"]) for row in rows]
    return encode_pairs(tokenizer, pairs, max_len)


def heldout_batches(ctx, tokenizer, lora_cfg: dict) -> list[dict]:
    """The first N_HELDOUT_ROWS held-out task rows, encoded and batched for the eval loss."""
    encoded = encode_rows(tokenizer, task_rows(ctx, "heldout", N_HELDOUT_ROWS), lora_cfg["max_len"])
    return make_batches(encoded, tokenizer.pad_token_id, lora_cfg["batch_size"])


def save_and_evaluate(ctx, run_id: str, fraction: float, model, tokenizer, heldout, amp):
    """Save the adapter at this fraction of training and score it on the held-out batches. Returns (path, loss)."""
    folder = ctx.out("ckpt", run_id, f"frac{round(fraction * 100)}")
    path = save_checkpoint(model, tokenizer, folder, LABEL)
    model.eval()
    with amp:
        heldout_loss = eval_loss(model, heldout, ctx.device)
    model.train()
    return path, heldout_loss


def optimizer_step(params, optimizer, scheduler) -> None:
    """Clip the gradient norm, step the optimizer and the learning-rate schedule, and clear the gradients."""
    torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def train(
    ctx,
    run_id: str,
    seed: int,
    lora_cfg: dict,
    model,
    tokenizer,
    train_set,
    heldout,
    max_steps: int | None = None,
) -> dict:
    """AdamW with gradient accumulation, clipping and warm-up/decay, saving at each configured fraction of
    training. Returns the log: train loss per step, held-out loss and checkpoint per save, totals.

    With `max_steps`, training stops after that many optimizer steps and saves nothing: a timing run, whose
    seconds per step (`s_per_step`) times `total_steps` estimates the full run."""
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lora_cfg["lr"], weight_decay=0.0)
    accum = lora_cfg["grad_accum"]
    n_batches = math.ceil(len(train_set) / lora_cfg["batch_size"])
    steps_per_epoch = math.ceil(n_batches / accum)  # the last accumulation window may be partial
    total_steps = steps_per_epoch * lora_cfg["epochs"]
    scheduler = warmup_linear_decay(optimizer, total_steps)
    save_at = save_steps(total_steps, ctx.cfg["save_fractions"])
    amp = autocast_context(ctx.device)
    log = {"train_loss": [], "eval_loss": {}, "saves": {}}
    step = 0
    window_loss = 0.0  # mean loss over the micro-batches of the current optimizer step
    train_start = time.time()

    for epoch in range(lora_cfg["epochs"]):
        # a distinct order per (seed, epoch), so run A's epoch 2 doesn't replay run B's epoch 1
        batches = length_bucketed_batches(train_set, lora_cfg["batch_size"], 1000 * seed + epoch)
        for i, group in enumerate(batches):
            window = min(accum, len(batches) - (i // accum) * accum)  # batches in this optimizer step
            batch = collate([train_set[j] for j in group], tokenizer.pad_token_id)
            with amp:
                loss, _ = response_loss(model, batch, ctx.device)
            (loss / window).backward()
            window_loss += loss.item() / window
            window_complete = (i + 1) % accum == 0 or i + 1 == len(batches)
            if not window_complete:
                continue
            optimizer_step(params, optimizer, scheduler)
            step += 1
            log["train_loss"].append(window_loss)
            window_loss = 0.0
            if _reached(step, max_steps):
                break
            if step not in save_at:
                continue
            fraction = save_at[step]
            path, heldout_loss = save_and_evaluate(ctx, run_id, fraction, model, tokenizer, heldout, amp)
            log["eval_loss"][str(fraction)] = heldout_loss
            log["saves"][str(fraction)] = {"step": step, "path": str(path), "elapsed_s": ctx.elapsed()}
            print(f"step {step}/{total_steps}: saved {path.name}, held-out loss {heldout_loss:.4f}")
        if _reached(step, max_steps):
            break

    train_s = time.time() - train_start
    log.update(
        total_steps=total_steps,
        steps_done=step,
        train_s=train_s,
        s_per_step=train_s / max(step, 1),
        n_rows=len(train_set),
        wall_clock_s=ctx.elapsed(),
    )
    return log


def _reached(step: int, max_steps: int | None) -> bool:
    """True once a timing run (--max-steps) has taken its steps; never for a full run."""
    return max_steps is not None and step >= max_steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(parser)
    parser.add_argument("--run", required=True, help="A, B, C or D (configs/experiment.yaml runs)")
    parser.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="timing run: stop after this many optimizer steps, save nothing"
    )
    args = parser.parse_args(argv)
    ctx = setup(args)
    run = ctx.cfg["runs"][args.run]
    lora_cfg = ctx.cfg["lora"]
    torch.manual_seed(run["seed"])
    tokenizer = load_tokenizer(ctx)

    rows = training_rows(ctx, run)
    train_set = encode_rows(tokenizer, rows, lora_cfg["max_len"])
    heldout = heldout_batches(ctx, tokenizer, lora_cfg)
    n_skipped = len(rows) - len(train_set)
    print(f"run {args.run}: {len(train_set)} training rows ({n_skipped} skipped as too long)")

    model = lora_model(ctx, lora_cfg, grad_ckpt=not args.no_grad_ckpt)
    log = train(ctx, args.run, run["seed"], lora_cfg, model, tokenizer, train_set, heldout, args.max_steps)
    result_name = f"finetune_{args.run}"
    if args.max_steps is not None:
        result_name = f"finetune_{args.run}_timing{args.max_steps}"
        estimate_h = log["s_per_step"] * log["total_steps"] / 3600
        print(f"{log['s_per_step']:.1f} s per step; the full run's {log['total_steps']} steps ≈ {estimate_h:.2f} h")
    path = write_result(
        result_name,
        log,
        config={"run": run, "lora": lora_cfg},
        seed=run["seed"],
        checkpoint=str(ctx.out("ckpt", args.run)),
        out_dir=ctx.out("results"),
    )
    print(f"done in {ctx.elapsed():.0f}s -> {path}")


if __name__ == "__main__":
    main()
