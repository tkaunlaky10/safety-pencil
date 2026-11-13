#!/usr/bin/env python
"""Save a WiSE-FT checkpoint without factors (13 Nov): W0 + (1 − α)·ΔW, α pulling toward θ0, with ΔW from the run's
final LoRA adapter in fp32, merged by pencil.baselines.wise_ft (the same path 07_baselines.py measured on dev).

Used by the wiseft_test stage for the post-hoc frontier match (WiSE-FT α = 0.5, PREREGISTRATION.md deviation log),
which needs a test-tier evaluation of a setting the scaled stage did not save.

    python scripts/11_wiseft_checkpoint.py --run A --alpha 0.5
    python scripts/11_wiseft_checkpoint.py --run A --alpha 0.5 --dry-run
"""

from __future__ import annotations

import argparse
import sys

from pencil.baselines import wise_ft
from pencil.delta import lora_delta, save_checkpoint
from pencil.runtime import adapter_dir, common_args, load_model, load_tokenizer, setup

LABEL = "research artifact: baseline wiseft:{alpha:g} on run {run} (post-hoc frontier match)"


def checkpoint_name(run: str, alpha: float) -> str:
    """The folder name under outputs/ckpt/baselines, e.g. 'runA_wiseft_0.5'."""
    return f"run{run}_wiseft_{alpha:g}"


def main(argv=None) -> int:
    parser = common_args(
        argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    )
    parser.add_argument("--run", default="A")
    parser.add_argument("--alpha", type=float, required=True, help="pull toward θ0, in [0, 1]")
    args = parser.parse_args(argv)
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    ctx = setup(args)
    model = load_model(ctx.aligned, ctx)
    deltas = lora_delta(adapter_dir(ctx, args.run, 100), lowrank=True)
    wise_ft(model, deltas, args.alpha)
    folder = ctx.out("ckpt", "baselines", checkpoint_name(args.run, args.alpha))
    path = save_checkpoint(model, load_tokenizer(ctx), folder, LABEL.format(alpha=args.alpha, run=args.run))
    print(path)  # the last line of output: the stage evaluates exactly this folder
    return 0


if __name__ == "__main__":
    sys.exit(main())
