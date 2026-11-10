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
from functools import partial

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
    embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    body = decoder(model)
    hidden = body(inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False).last_hidden_state
    log_probs = lm_logits(model, hidden[loss_mask]).float().log_softmax(-1)
    with torch.no_grad():
        labels = torch.multinomial(log_probs.exp(), 1, generator=generator)
    log_probs.gather(-1, labels).sum().backward()
    return int(loss_mask.sum())


def _input_rows(a: torch.Tensor, mask: torch.Tensor, row_set: str, dtype: torch.dtype) -> torch.Tensor:
    """The rows of a layer input a (batch × time × d) that a row set uses, as (rows × d). A reduce set gives
    one row per sequence: the mean input over the sequence's masked positions."""
    if row_set not in REDUCE_ROW_SETS:
        return a[mask].to(dtype)
    rows_per_sequence = mask.sum(1, keepdim=True).clamp_min(1)
    return _per_sequence_sum(a, mask, dtype) / rows_per_sequence


def _gradient_rows(g: torch.Tensor, mask: torch.Tensor, row_set: str, dtype: torch.dtype) -> torch.Tensor:
    """The rows of an output gradient g (batch × time × d) that a row set uses, as (rows × d). A reduce set
    gives one row per sequence: the summed gradient over the sequence's masked positions."""
    if row_set not in REDUCE_ROW_SETS:
        return g[mask].to(dtype)
    return _per_sequence_sum(g, mask, dtype)


def _per_sequence_sum(x: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Σ over the masked positions of each sequence: (batch × time × d) -> (batch × d)."""
    return (x.to(dtype) * mask[..., None]).sum(1)


def _row_count(mask: torch.Tensor, row_set: str) -> int:
    """How many rows a batch adds to a row set's A normalisation: one per non-empty sequence for a reduce
    set, otherwise one per masked position."""
    if row_set in REDUCE_ROW_SETS:
        return int(mask.any(1).sum())
    return int(mask.sum())


def _accumulate(store: dict, key, M: torch.Tensor) -> None:
    """store[key] += M, starting the entry if it is new."""
    if key in store:
        store[key] = store[key] + M
    else:
        store[key] = M


class KFACCollector:
    """Hooks the decoder's linear layers and accumulates K-FAC factors over `collect` calls:

        with KFACCollector(model, layers=range(6)) as kc:
            for batch in batches:              # dicts from pencil.chat.collate
                kc.collect(batch)
        factors = kc.finalize()

    Data flow per collect(batch):
      1. collect() builds the batch's row-set masks and stores them in `_masks` for the hooks.
      2. Forward hook (one per linear layer): adds a aᵀ for each row set to A[input key][row set]. In a
         parallel block (Tiny Aya) q/k/v/gate/up share one input factor, so only its owner module (the first
         one) accumulates it; on the first batch the others check that they really read the same tensor.
         The hook also registers a gradient hook on the layer's output.
      3. Gradient hook (during backward): adds g gᵀ for each row set to G[module][row set].
      4. collect() clears the per-batch state and adds the batch's row and loss-token counts.
    finalize() divides by the counts and returns a Factors object.

    `diag=True` also accumulates the per-sequence diagonal Fisher Σ_seq (Σ_t g_t a_tᵀ)² / n_loss (the
    diagonal Fisher merge baseline) on `diag_row_set`: the forward hook keeps the masked input until the
    gradient hook pairs it with the gradient. Accumulators are fp32 by default; float64 is exact but slow on
    consumer GPUs."""

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
        if share_input_factor is None:
            share_input_factor = is_parallel_block(model)
        self.share_input_factor = share_input_factor
        self.a_key = {module: input_factor_key(module, share_input_factor) for module in self.linears}
        self.input_owner = {}  # input factor key -> the one module whose hook accumulates it
        for module, input_key in self.a_key.items():
            self.input_owner.setdefault(input_key, module)
        self.diag = diag
        self.diag_row_set = diag_row_set
        self.seed = seed
        self.accum_dtype = accum_dtype

        # Accumulators (unnormalised until finalize) and the counts that normalise them.
        self.A = {input_key: {} for input_key in self.input_owner}  # A[input key][row set]
        self.G = {module: {} for module in self.linears}  # G[module][row set]
        self.D = {}  # D[module]: the diagonal Fisher, d_out × d_in
        self.n_rows = dict.fromkeys(self.row_sets, 0)
        self.n_loss = 0
        self.n_seqs = 0

        self._generator = None  # label sampler, created on the model's device by the first collect()
        self._handles = []  # forward hooks, registered by __enter__
        self._masks = None  # row-set masks, set only while collect() runs, so hooks ignore other forwards
        self._diag_inputs = {}  # module -> masked input, from its forward hook until its gradient hook
        self._first_batch = True
        self._shared_inputs = {}  # first batch only: input key -> the owner's input, to check the others

    def __enter__(self):
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._handles = []
        for module_name, linear in self.linears.items():
            self._handles.append(linear.register_forward_hook(partial(self._on_forward, module_name)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _on_forward(self, module_name: str, linear: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        """Forward hook: accumulate the input factor, keep what the gradient hook needs, hook the output."""
        if self._masks is None:
            return
        a = inputs[0].detach()
        input_key = self.a_key[module_name]
        if self.input_owner[input_key] == module_name:
            for row_set in self.row_sets:
                rows = _input_rows(a, self._masks[row_set], row_set, self.accum_dtype)
                _accumulate(self.A[input_key], row_set, rows.T @ rows)
            if self._first_batch:
                self._shared_inputs[input_key] = a
        elif self._first_batch:
            self._check_reads_shared_input(module_name, input_key, a)
        if self.diag:
            self._diag_inputs[module_name] = a.float() * self._masks[self.diag_row_set][..., None]
        if output.requires_grad:
            output.register_hook(partial(self._on_output_grad, module_name))

    def _check_reads_shared_input(self, module_name: str, input_key: str, a: torch.Tensor) -> None:
        """Raise if a module assigned to a shared input factor reads a different tensor from its owner."""
        owner_input = self._shared_inputs.get(input_key, torch.empty(0))
        if not torch.equal(owner_input, a):
            raise RuntimeError(
                f"{module_name} does not read the same input as {self.input_owner[input_key]}: the block is not "
                "parallel, pass share_input_factor=False"
            )

    def _on_output_grad(self, module_name: str, g: torch.Tensor) -> None:
        """Gradient hook on a layer's output: accumulate G and, with diag, the diagonal Fisher."""
        g = g.detach()
        for row_set in self.row_sets:
            rows = _gradient_rows(g, self._masks[row_set], row_set, self.accum_dtype)
            _accumulate(self.G[module_name], row_set, rows.T @ rows)
        if self.diag:
            # The gradient of W for each sequence, Σ_t g_t a_tᵀ, squared: cross-position terms included.
            a_masked = self._diag_inputs.pop(module_name)
            g_masked = g.float() * self._masks[self.diag_row_set][..., None]
            n_sequences = g_masked.shape[0]
            squared = sum((g_masked[s].T @ a_masked[s]) ** 2 for s in range(n_sequences))
            _accumulate(self.D, module_name, squared)

    def _row_masks(self, batch: dict, attention: torch.Tensor, device) -> dict[str, torch.Tensor]:
        """{row set: boolean mask} for every row set the hooks or the loss need, restricted to real tokens."""
        needed = set(self.row_sets)
        needed.add("loss")
        if self.diag:
            needed.add(self.diag_row_set)
        return {row_set: batch[ROW_MASK[row_set]].to(device).bool() & attention for row_set in needed}

    def collect(self, batch: dict[str, torch.Tensor]) -> int:
        """One sampled-label pass on a batch from pencil.chat.collate. Returns its loss-token count."""
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device).bool()
        masks = self._row_masks(batch, attention, device)
        if self._generator is None:
            self._generator = torch.Generator(device=device).manual_seed(self.seed)

        self._masks = masks
        try:
            n_loss = sampled_label_backward(self.model, input_ids, attention.long(), masks["loss"], self._generator)
        finally:
            self._masks = None
            self._diag_inputs.clear()
            self._shared_inputs.clear()
        self._first_batch = False

        for row_set in self.row_sets:
            self.n_rows[row_set] += _row_count(masks[row_set], row_set)
        self.n_loss += n_loss
        self.n_seqs += input_ids.shape[0]
        return n_loss

    def finalize(self, meta: dict | None = None) -> Factors:
        """Normalise the factors (A by its row count, G and D by the loss-token count) and return them.

        Works in place and moves each factor to the CPU as soon as it is done, so GPU memory never exceeds
        the accumulators. Factors keep accum_dtype; consumers upcast to float64."""
        if self.n_loss == 0:
            raise RuntimeError("no loss tokens were seen")
        A = {}
        for input_key, by_row_set in self.A.items():
            A[input_key] = {}
            for row_set in list(by_row_set):
                A[input_key][row_set] = by_row_set.pop(row_set).div_(self.n_rows[row_set]).cpu()
        G = {}
        for module, by_row_set in self.G.items():
            G[module] = {}
            for row_set in list(by_row_set):
                G[module][row_set] = by_row_set.pop(row_set).div_(self.n_loss).cpu()
        D = {}
        for module in list(self.D):
            D[module] = self.D.pop(module).div_(self.n_loss).cpu()

        info = {
            "shared_input_factor": self.share_input_factor,
            "seed": self.seed,
            "accum_dtype": str(self.accum_dtype),
        }
        if meta is not None:
            info.update(meta)
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


def collect_factors(
    model: nn.Module, batches: Iterable[dict], meta: dict | None = None, **collector_options
) -> Factors:
    """Factors from one KFACCollector run over all batches; `collector_options` go to KFACCollector."""
    with KFACCollector(model, **collector_options) as collector:
        for batch in batches:
            collector.collect(batch)
    return collector.finalize(meta)


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
    q_width = n_heads * head_dim
    kv_width = n_kv * head_dim
    n_copies_of_shared_input = 1 if shared else 5
    # input factors: the shared input (q/k/v/gate/up), o_proj input, down_proj input
    input_entries = n_copies_of_shared_input * hidden**2 + q_width**2 + inter**2
    # output factors: q, k, v, o, gate, up, down
    output_entries = q_width**2 + 2 * kv_width**2 + hidden**2 + 2 * inter**2 + hidden**2
    return (input_entries + output_entries) * n_row_sets * bytes_per
