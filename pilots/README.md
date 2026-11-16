# Pilots (CPU, no GPU credits)

Small experiments behind WRITEUP §8 and `docs/WITH_MORE_CREDITS.md` §1.5. They test mechanisms on controlled models; they are not results on Tiny Aya.

| File | What it tests |
| --- | --- |
| `toy.py` | A 2-layer transformer trained from scratch on a synthetic language: prompts with a trigger token must be refused; a LoRA fine-tune on an unrelated benign task erodes that. T1: the forecast-gap decomposition (E1). T2: repairs (WiSE-FT, diagonal Fisher, module weights on the linearised and on the exact objective; E2). T4: perturbation size and direction against the quadratic forecast (E6). T5: held-out gradient prediction of the best interpolation point (E5). |
| `prevent.py` | Safe fine-tuning: the same fine-tune with an EWC-style Fisher penalty, an exact KL penalty, or a refusal-margin penalty (E9). |
| `fmatrix.py` | The pencil as a two-sample F-matrix problem: in-sample vs cross-fitted vs shrunk generalised eigenvalues, scored on the true objective (E3); Wachter's law under the null. |

No natural-language or harmful text is used anywhere: the "harmful" prompts are synthetic token patterns.

```bash
pip install "jax[cpu]" scipy numpy
python pilots/toy.py 0        # seed 0; about 10 min on 2 CPU cores; writes pilots/results/
python pilots/prevent.py 0
python pilots/fmatrix.py
```

Results for seeds 0–2 are in `pilots/results/`.
