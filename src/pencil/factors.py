"""Collecting K-FAC factors with sampled-label Fisher passes on a Hugging Face causal LM.

For each hooked linear layer W (d_out × d_in), one forward/backward pass per batch adds

    A += Σ_rows a aᵀ      (a: the layer's input)       finally divided by the number of rows
    G += Σ_rows g gᵀ      (g: gradient at its output)  finally divided by the number of loss tokens

(the "expand" normalisation in CLAUDE.md). Labels are sampled from the model's own softmax at every loss
position, so one backward pass of their summed log-probs gives an unbiased G (VERIFICATION.md V7).

Row sets, all filled in the same pass (masks from pencil.chat.collate):
  loss         positions whose logits are scored (they predict a response token)
  all          the user prompt and the response, without the fixed system preamble
  full         every position, preamble included
  full_reduce  K-FAC-reduce over the full rows: one (mean input, summed gradient) pair per sequence

Rules (CLAUDE.md 3): weights frozen and `inputs_embeds` differentiable, so no weight gradients are stored;
no gradient checkpointing; small batches; hook a chunk of layers per pass if the factors don't fit.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .factor_store import Factors
from .modules import decoder, input_factor_key, is_parallel_block, lm_logits, target_modules

ROW_SETS = ("loss", "all", "full", "full_reduce")
REDUCE_ROW_SETS = ("full_reduce",)
ROW_MASK = {"loss": "loss_mask", "all": "rows_all", "full": "rows_full", "full_reduce": "rows_full"}


def sampled_label_backward(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> int:
    """Forward with differentiable input embeddings, sample one label per loss position from the model's
    softmax, and back-propagate the sum of those labels' log-probs. Returns the number of loss tokens."""
    emb = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    hidden = decoder(model)(inputs_embeds=emb, attention_mask=attention_mask, use_cache=False).last_hidden_state
    logp = lm_logits(model, hidden[loss_mask]).float().log_softmax(-1)
    with torch.no_grad():
        labels = torch.multinomial(logp.exp(), 1, generator=generator)
    logp.gather(-1, labels).sum().backward()
    return int(loss_mask.sum())


def _rows(x: torch.Tensor, mask: torch.Tensor, row_set: str, dtype, reduce_mean: bool) -> torch.Tensor:
    """The rows of x (batch × time × d) a row set uses. Reduce sets give one row per sequence: the mean
    over its rows for inputs, the sum for gradients."""
    if row_set not in REDUCE_ROW_SETS:
        return x[mask].to(dtype)
    summed = (x.to(dtype) * mask[..., None]).sum(1)
    return summed / mask.sum(1, keepdim=True).clamp_min(1) if reduce_mean else summed


class KFACCollector:
    """Hooks the decoder's linear layers and accumulates K-FAC factors over `collect` calls:

        with KFACCollector(model, layers=range(6)) as kc:
            for batch in batches:              # dicts from pencil.chat.collate
                kc.collect(batch)
        factors = kc.finalize()

    In a parallel block (Tiny Aya), q/k/v/gate/up share one input factor; the first batch checks that they
    really read the same tensor. `diag=True` also accumulates the per-sequence diagonal Fisher
    Σ_seq (Σ_t g_t a_tᵀ)² / n_loss (the diagonal Fisher merge baseline) on `diag_row_set`.
    Accumulators are fp32 by default; float64 is exact but slow on consumer GPUs."""

    def __init__(
        self,
        model: nn.Module,
        layers: Iterable[int] | None = None,
        *,
        row_sets=ROW_SETS,
        share_input_factor: bool | None = None,
        diag: bool = False,
        diag_row_set: str = "all",
        seed: int = 0,
        accum_dtype: torch.dtype = torch.float32,
    ):
        if not set(row_sets) <= set(ROW_SETS):
            raise ValueError(f"row_sets must be drawn from {ROW_SETS}")
        if diag and diag_row_set in REDUCE_ROW_SETS:
            raise ValueError("the diagonal Fisher needs a per-row set (loss, all or full)")
        self.model = model
        self.linears = target_modules(model, layers)
        self.row_sets = tuple(row_sets)
        self.shared = is_parallel_block(model) if share_input_factor is None else share_input_factor
        self.a_key = {n: input_factor_key(n, self.shared) for n in self.linears}
        self.owner = {}  # the module that accumulates each input factor
        for n, k in self.a_key.items():
            self.owner.setdefault(k, n)
        self.diag, self.diag_row_set = diag, diag_row_set
        self.seed, self.accum_dtype = seed, accum_dtype
        self.A = {k: {} for k in self.owner}
        self.G = {n: {} for n in self.linears}
        self.D = {}
        self.n_rows = dict.fromkeys(self.row_sets, 0)
        self.n_loss = self.n_seqs = 0
        self._generator = None
        self._handles = []
        self._masks = None  # set only while collect() runs, so hooks ignore other forwards
        self._diag_inputs = {}
        self._first_batch = True
        self._shared_inputs = {}

    def __enter__(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._handles = [m.register_forward_hook(self._forward_hook(n)) for n, m in self.linears.items()]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    @staticmethod
    def _add(store: dict, key, M: torch.Tensor) -> None:
        store[key] = M if key not in store else store[key] + M

    def _forward_hook(self, name):
        def hook(module, inputs, output):
            if self._masks is None:
                return
            a = inputs[0].detach()
            key = self.a_key[name]
            if self.owner[key] == name:
                for rs in self.row_sets:
                    rows = _rows(a, self._masks[rs], rs, self.accum_dtype, reduce_mean=True)
                    self._add(self.A[key], rs, rows.T @ rows)
                if self._first_batch:
                    self._shared_inputs[key] = a
            elif self._first_batch and not torch.equal(self._shared_inputs.get(key, torch.empty(0)), a):
                raise RuntimeError(
                    f"{name} does not read the same input as {self.owner[key]}: the block is not "
                    "parallel, pass share_input_factor=False"
                )
            if self.diag:
                self._diag_inputs[name] = a.float() * self._masks[self.diag_row_set][..., None]
            if output.requires_grad:
                output.register_hook(self._grad_hook(name))

        return hook

    def _grad_hook(self, name):
        def hook(g):
            g = g.detach()
            for rs in self.row_sets:
                rows = _rows(g, self._masks[rs], rs, self.accum_dtype, reduce_mean=False)
                self._add(self.G[name], rs, rows.T @ rows)
            if self.diag:  # per-sequence gradient, squared: cross-position terms included
                a = self._diag_inputs.pop(name)
                gm = g.float() * self._masks[self.diag_row_set][..., None]
                self._add(self.D, name, sum((gm[b].T @ a[b]) ** 2 for b in range(gm.shape[0])))

        return hook

    def collect(self, batch: dict[str, torch.Tensor]) -> int:
        """One sampled-label pass on a batch from pencil.chat.collate. Returns its loss-token count."""
        dev = next(self.model.parameters()).device
        ids, att = batch["input_ids"].to(dev), batch["attention_mask"].to(dev).bool()
        needed = {*self.row_sets, "loss", *([self.diag_row_set] if self.diag else [])}
        masks = {rs: batch[ROW_MASK[rs]].to(dev).bool() & att for rs in needed}
        if self._generator is None:
            self._generator = torch.Generator(device=dev).manual_seed(self.seed)
        self._masks = masks
        try:
            n = sampled_label_backward(self.model, ids, att.long(), masks["loss"], self._generator)
        finally:
            self._masks = None
            self._diag_inputs.clear()
            self._shared_inputs.clear()
        self._first_batch = False
        for rs in self.row_sets:  # reduce sets normalise A per sequence, the others per row
            self.n_rows[rs] += int(masks[rs].any(1).sum()) if rs in REDUCE_ROW_SETS else int(masks[rs].sum())
        self.n_loss += n
        self.n_seqs += ids.shape[0]
        return n

    def finalize(self, meta: dict | None = None) -> Factors:
        """Normalise in place and move each factor to the CPU as soon as it is done, so GPU memory never
        exceeds the accumulators. Factors keep accum_dtype; consumers upcast to float64."""
        if self.n_loss == 0:
            raise RuntimeError("no loss tokens were seen")
        A = {k: {rs: v.pop(rs).div_(self.n_rows[rs]).cpu() for rs in list(v)} for k, v in self.A.items()}
        G = {k: {rs: v.pop(rs).div_(self.n_loss).cpu() for rs in list(v)} for k, v in self.G.items()}
        D = {k: self.D.pop(k).div_(self.n_loss).cpu() for k in list(self.D)}
        info = {
            "shared_input_factor": self.shared,
            "seed": self.seed,
            "accum_dtype": str(self.accum_dtype),
            **(meta or {}),
        }
        return Factors(
            A=A,
            G=G,
            a_key=dict(self.a_key),
            n_rows=dict(self.n_rows),
            n_loss=self.n_loss,
            n_seqs=self.n_seqs,
            D=D,
            meta=info,
        )


def collect_factors(model: nn.Module, batches: Iterable[dict], meta: dict | None = None, **kw) -> Factors:
    """One collector over all batches; kw go to KFACCollector."""
    with KFACCollector(model, **kw) as kc:
        for b in batches:
            kc.collect(b)
    return kc.finalize(meta)


def factor_bytes(
    hidden: int,
    inter: int,
    n_heads: int,
    n_kv: int,
    head_dim: int,
    *,
    shared: bool = True,
    n_row_sets: int = 2,
    bytes_per: int = 4,
) -> int:
    """Memory of one layer's factors for one data side over all row sets (the plan in IMPLEMENTATION.md)."""
    q, kv = n_heads * head_dim, n_kv * head_dim
    a = (1 if shared else 5) * hidden**2 + q**2 + inter**2  # shared in, o_proj in, down in
    g = q**2 + 2 * kv**2 + hidden**2 + 2 * inter**2 + hidden**2  # q, k, v, o, gate, up, down
    return (a + g) * n_row_sets * bytes_per
