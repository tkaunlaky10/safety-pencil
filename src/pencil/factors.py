"""K-FAC factors from sampled-label Fisher passes on a Hugging Face causal LM.

For every target linear layer W (d_out × d_in) we accumulate, in one forward/backward pass per batch:

    A = Σ_rows a aᵀ / n_rows        (a: the layer's input at a row)
    G = Σ_rows g gᵀ / n_loss        (g: gradient of the summed sampled-label log-probs w.r.t. the output)

which is the K-FAC "expand" normalisation in CLAUDE.md. Two row sets are filled in the same pass:
"loss" (positions whose logits are scored, i.e. predict a response token) and "all" (every user-prompt
and response position, without the fixed system preamble). Labels are sampled from the model's own
softmax at each loss position, so a single backward pass of their summed log-probs is an unbiased
estimate of G (VERIFICATION.md V7).

Tiny Aya's decoder block is parallel (attention and MLP read the same `input_layernorm` output), so
q, k, v, gate and up share one A factor per layer, stored under the key "<prefix>layers.<L>.input".
o_proj and down_proj have their own. The collector checks on the first batch that the shared inputs
really are identical.

Conventions (CLAUDE.md rule 3): every parameter frozen, `inputs_embeds.requires_grad_(True)` so no weight
gradients are stored, no gradient checkpointing, small batches; hook a chunk of layers per pass if the
MLP-width factors don't fit.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

ROW_SETS = ("loss", "all", "full", "full_reduce")
REDUCE = ("full_reduce",)          # K-FAC-reduce: one (mean a, summed g) pair per sequence
MASK_OF = {"loss": "loss_mask", "all": "rows_all", "full": "rows_full", "full_reduce": "rows_full"}
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
SHARED_INPUT = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
PARALLEL_MODEL_TYPES = ("cohere", "cohere2")
_NAME = re.compile(r"^(?P<prefix>.*?)layers\.(?P<layer>\d+)\.(?P<block>self_attn|mlp)\.(?P<leaf>\w+_proj)$")


# ---------------------------------------------------------------- module bookkeeping
def parse_name(name: str) -> tuple[str, int, str]:
    """'model.layers.3.self_attn.q_proj' -> ('model.', 3, 'q_proj')."""
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"not a decoder linear module: {name}")
    return m["prefix"], int(m["layer"]), m["leaf"]


def target_modules(model: nn.Module, layers: Iterable[int] | None = None) -> dict[str, nn.Linear]:
    """Every q/k/v/o/gate/up/down nn.Linear in the decoder layers (optionally only `layers`)."""
    keep = None if layers is None else set(int(x) for x in layers)
    out = {}
    for name, mod in model.named_modules():
        m = _NAME.match(name)
        if m and m["leaf"] in ATTN + MLP and isinstance(mod, nn.Linear):
            if keep is None or int(m["layer"]) in keep:
                out[name] = mod
    if not out:
        raise ValueError("no target nn.Linear modules found. Merge any PEFT adapter into the weights first "
                         "(pencil.delta.merge_deltas), the collector hooks plain nn.Linear layers only")
    return out


def is_parallel_block(model: nn.Module) -> bool:
    return getattr(getattr(model, "config", None), "model_type", None) in PARALLEL_MODEL_TYPES


def a_key_for(name: str, shared: bool) -> str:
    prefix, layer, leaf = parse_name(name)
    return f"{prefix}layers.{layer}.input" if shared and leaf in SHARED_INPUT else name


def _decoder(model: nn.Module) -> nn.Module:
    dec = getattr(model, "model", None)
    return dec if dec is not None else model.get_decoder()


def _logits(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    logits = model.get_output_embeddings()(hidden)
    scale = getattr(model, "logit_scale", None)        # Cohere multiplies logits by logit_scale
    return logits * scale if scale is not None else logits


# ---------------------------------------------------------------- the sampled-label pass
def sampled_label_backward(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                           loss_mask: torch.Tensor, generator: torch.Generator | None = None) -> int:
    """One forward with frozen weights and differentiable input embeddings, sample one label per loss
    position from softmax(logits) without gradient, and back-propagate the sum of their log-probs.
    The LM head runs only on loss positions (the vocabulary is 262k for Tiny Aya). Returns n_loss."""
    emb = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    hidden = _decoder(model)(inputs_embeds=emb, attention_mask=attention_mask, use_cache=False).last_hidden_state
    logp = _logits(model, hidden[loss_mask]).float().log_softmax(-1)          # (n_loss, V)
    with torch.no_grad():
        y = torch.multinomial(logp.exp(), 1, generator=generator)
    logp.gather(-1, y).sum().backward()
    return int(loss_mask.sum())


# ---------------------------------------------------------------- results container
@dataclass
class Factors:
    """Normalised factors. A[a_key][row_set], G[module][row_set]; a_key[module] says which A a module uses."""
    A: dict[str, dict[str, torch.Tensor]]
    G: dict[str, dict[str, torch.Tensor]]
    a_key: dict[str, str]
    n_rows: dict[str, int]
    n_loss: int
    n_seqs: int
    D: dict[str, torch.Tensor] = field(default_factory=dict)     # optional diagonal Fisher per module
    meta: dict = field(default_factory=dict)

    @property
    def modules(self) -> list[str]:
        return sorted(self.G)

    @property
    def row_sets(self) -> list[str]:
        return sorted(self.n_rows)

    def pair(self, module: str, row_set: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(A, G) for one module and row set."""
        return self.A[self.a_key[module]][row_set], self.G[module][row_set]

    def get(self, kind: str, key: str, row_set: str | None = None) -> torch.Tensor:
        """One stored tensor: kind "A" (key = an A key), "G" (key = a module) or "D" (key = a module)."""
        if kind == "D":
            return self.D[key]
        return (self.A if kind == "A" else self.G)[key][row_set]

    def dim(self, kind: str, key: str) -> int:
        return self.get(kind, key, self.row_sets[0]).shape[0]

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file
        flat = {}
        for k, by_rs in self.A.items():
            for rs, t in by_rs.items():
                flat[f"A::{k}::{rs}"] = t.contiguous().cpu()
        for k, by_rs in self.G.items():
            for rs, t in by_rs.items():
                flat[f"G::{k}::{rs}"] = t.contiguous().cpu()
        for k, t in self.D.items():
            flat[f"D::{k}"] = t.contiguous().cpu()
        meta = {"a_key": self.a_key, "n_rows": self.n_rows, "n_loss": self.n_loss, "n_seqs": self.n_seqs,
                "meta": self.meta}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(flat, str(path), metadata={"pencil": json.dumps(meta)})

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "Factors":
        from safetensors import safe_open
        A, G, D = {}, {}, {}
        with safe_open(str(path), framework="pt", device=str(device)) as f:
            meta = json.loads(f.metadata()["pencil"])
            for key in f.keys():
                kind, rest = key.split("::", 1)
                if kind == "D":
                    D[rest] = f.get_tensor(key)
                    continue
                name, rs = rest.rsplit("::", 1)
                (A if kind == "A" else G).setdefault(name, {})[rs] = f.get_tensor(key)
        return cls(A=A, G=G, a_key=meta["a_key"], n_rows=meta["n_rows"], n_loss=meta["n_loss"],
                   n_seqs=meta["n_seqs"], D=D, meta=meta.get("meta", {}))

    @classmethod
    def load_many(cls, paths: Iterable[str | Path], device="cpu") -> "Factors":
        """Merge per-layer-chunk files of the same pass into one object."""
        parts = [cls.load(p, device) for p in paths]
        out = parts[0]
        for p in parts[1:]:
            if p.n_loss != out.n_loss or p.n_rows != out.n_rows:
                raise ValueError("chunks come from different passes (token counts differ)")
            out.A.update(p.A)
            out.G.update(p.G)
            out.a_key.update(p.a_key)
            out.D.update(p.D)
        return out


def pool(factors: list[Factors], weights: list[float] | None = None) -> Factors:
    """Weighted average of normalised factors (e.g. across languages). Weights default to equal and
    are renormalised to sum to 1. Use token counts as weights for a token-weighted pool."""
    w = torch.tensor(weights if weights is not None else [1.0] * len(factors), dtype=torch.float64)
    w = (w / w.sum()).tolist()
    f0 = factors[0]
    for f in factors[1:]:
        if f.a_key != f0.a_key or set(f.n_rows) != set(f0.n_rows):
            raise ValueError("cannot pool factors with different modules or row sets")

    def avg(get):
        return sum(wi * get(f) for wi, f in zip(w, factors))

    A = {k: {rs: avg(lambda f: f.A[k][rs]) for rs in f0.A[k]} for k in f0.A}
    G = {k: {rs: avg(lambda f: f.G[k][rs]) for rs in f0.G[k]} for k in f0.G}
    D = {k: avg(lambda f: f.D[k]) for k in f0.D} if all(f.D for f in factors) else {}
    return Factors(A=A, G=G, a_key=dict(f0.a_key), n_rows={rs: sum(f.n_rows[rs] for f in factors) for rs in f0.n_rows},
                   n_loss=sum(f.n_loss for f in factors), n_seqs=sum(f.n_seqs for f in factors), D=D,
                   meta={"pooled_from": [f.meta for f in factors], "weights": w})


# ---------------------------------------------------------------- the collector
class KFACCollector:
    """Hooks the target linear layers of `model` and accumulates K-FAC factors over `collect` calls.

        with KFACCollector(model, layers=range(0, 6)) as kc:
            for batch in batches:              # dicts from pencil.chat.collate
                kc.collect(batch)
        factors = kc.finalize()

    `diag=True` also accumulates the diagonal of the per-sequence Fisher, Σ_seq (Σ_t g_t a_tᵀ)² / n_loss, per
    module on the `diag_row_set` rows (Matena & Raffel's estimator with sampled labels; same size as the
    weights, so use layer chunks).

    Factors are accumulated in `accum_dtype` (fp32 by default), so their exact null directions are only
    null up to ~1e-7 of the top eigenvalue: use geneig.noise_floor / auto_rel_tol before range restriction.
    """

    def __init__(self, model: nn.Module, layers: Iterable[int] | None = None, *, row_sets=ROW_SETS,
                 share_input_factor: bool | None = None, diag: bool = False, diag_row_set: str = "all",
                 factor_device: str | torch.device | None = None, seed: int = 0,
                 accum_dtype: torch.dtype = torch.float32):
        self.model = model
        self.modules = target_modules(model, layers)
        self.row_sets = tuple(row_sets)
        if not set(self.row_sets) <= set(ROW_SETS):
            raise ValueError(f"row_sets must be drawn from {ROW_SETS}")
        self.shared = is_parallel_block(model) if share_input_factor is None else share_input_factor
        self.a_key = {n: a_key_for(n, self.shared) for n in self.modules}
        # one module per A key does the accumulation; the others in a shared group are checked against it
        self.owner = {}
        for n, k in self.a_key.items():
            self.owner.setdefault(k, n)
        self.diag = diag
        if diag and diag_row_set in REDUCE:
            raise ValueError("the diagonal Fisher needs a per-row set (loss, all or full)")
        self.diag_row_set = diag_row_set
        dev = next(model.parameters()).device
        self.fdev = torch.device(factor_device) if factor_device is not None else dev
        self.A = {k: {rs: None for rs in self.row_sets} for k in self.owner}
        self.G = {n: {rs: None for rs in self.row_sets} for n in self.modules}
        self.D = {n: None for n in self.modules} if diag else {}
        self.n_rows = {rs: 0 for rs in self.row_sets}
        self.n_loss = 0
        self.n_seqs = 0
        self.seed = seed
        self.accum_dtype = accum_dtype          # float64 is exact but slow on consumer GPUs (L4 FP64 ~1/64 rate)
        self._gen = None
        self._handles = []
        self._masks: dict[str, torch.Tensor] | None = None
        self._diag_inputs: dict[str, torch.Tensor] = {}
        self._check_shared = self.shared
        self._shared_seen: dict[str, torch.Tensor] = {}

    # -- hooks
    def __enter__(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        for name, mod in self.modules.items():
            self._handles.append(mod.register_forward_hook(self._forward_hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _forward_hook(self, name):
        def hook(mod, inputs, output):
            if self._masks is None:
                return
            a = inputs[0].detach()
            key = self.a_key[name]
            if self.owner[key] == name:
                for rs in self.row_sets:
                    m = self._masks[rs]
                    if rs in REDUCE:
                        am = (a.to(self.accum_dtype) * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
                    else:
                        am = a[m].to(self.accum_dtype)
                    self._add(self.A[key], rs, am.T @ am)
                if self._check_shared:
                    self._shared_seen[key] = a
            elif self._check_shared:
                ref = self._shared_seen.get(key)
                if ref is None or not torch.equal(ref, a):
                    raise RuntimeError(f"{name} does not read the same input as {self.owner[key]}; the block is "
                                       "not parallel, pass share_input_factor=False")
            if self.diag:
                m = self._masks[self.diag_row_set]
                self._diag_inputs[name] = a.float() * m[..., None]
            if output.requires_grad:
                output.register_hook(self._grad_hook(name))
        return hook

    def _grad_hook(self, name):
        def hook(g):
            g = g.detach()
            for rs in self.row_sets:
                m = self._masks[rs]
                gm = (g.to(self.accum_dtype) * m[..., None]).sum(1) if rs in REDUCE else g[m].to(self.accum_dtype)
                self._add(self.G[name], rs, gm.T @ gm)
            if self.diag:
                # true per-sequence diagonal: (Σ_t g_t a_tᵀ)² summed over sequences, cross-position terms included
                am = self._diag_inputs.pop(name)
                gm = g.float() * self._masks[self.diag_row_set][..., None]
                d = sum((gm[b].T @ am[b]) ** 2 for b in range(gm.shape[0]))
                self.D[name] = d.to(self.fdev) if self.D[name] is None else self.D[name] + d.to(self.fdev)
        return hook

    def _add(self, store, rs, M):
        M = M.to(self.fdev)
        store[rs] = M if store[rs] is None else store[rs] + M

    # -- one batch
    def collect(self, batch: dict[str, torch.Tensor]) -> int:
        """Run one sampled-label pass on a batch from pencil.chat.collate. Returns the loss-token count."""
        dev = next(self.model.parameters()).device
        ids = batch["input_ids"].to(dev)
        att = batch["attention_mask"].to(dev)
        need = set(self.row_sets) | {"loss"} | ({self.diag_row_set} if self.diag else set())
        masks = {rs: batch[MASK_OF[rs]].to(dev).bool() & att.bool() for rs in need}
        if self._gen is None:
            self._gen = torch.Generator(device=dev).manual_seed(self.seed)
        self._masks = masks
        try:
            n = sampled_label_backward(self.model, ids, att, masks["loss"], self._gen)
        finally:
            self._masks = None
            self._diag_inputs.clear()
            self._shared_seen.clear()
        self._check_shared = False
        for rs in self.row_sets:                 # reduce sets normalise A per sequence, the others per row
            self.n_rows[rs] += int(masks[rs].any(1).sum()) if rs in REDUCE else int(masks[rs].sum())
        self.n_loss += n
        self.n_seqs += ids.shape[0]
        return n

    def finalize(self, meta: dict | None = None) -> Factors:
        """Normalise in place and move each factor to the CPU as it is done, so peak GPU memory stays at the
        accumulators' size. Factors keep accum_dtype (fp32 by default; consumers upcast to float64)."""
        if self.n_loss == 0:
            raise RuntimeError("no loss tokens were seen")

        def done(t, n):
            out = t.div_(n).cpu()
            return out

        A = {k: {} for k in self.A}
        for k, v in self.A.items():
            for rs in list(v):
                A[k][rs] = done(v.pop(rs), self.n_rows[rs])
        G = {k: {} for k in self.G}
        for k, v in self.G.items():
            for rs in list(v):
                G[k][rs] = done(v.pop(rs), self.n_loss)
        D = {k: done(self.D[k], self.n_loss) for k in list(self.D)} if self.diag else {}
        self.D = {}
        info = {"shared_input_factor": self.shared, "seed": self.seed, "accum_dtype": str(self.accum_dtype)}
        if meta:
            info.update(meta)
        return Factors(A=A, G=G, a_key=dict(self.a_key), n_rows=dict(self.n_rows), n_loss=self.n_loss,
                       n_seqs=self.n_seqs, D=D, meta=info)


def collect_factors(model: nn.Module, batches: Iterable[dict], **kw) -> Factors:
    """Convenience wrapper: one collector over all batches. kw go to KFACCollector (layers, row_sets,
    share_input_factor, diag, factor_device, seed) except `meta`, which goes to finalize."""
    meta = kw.pop("meta", None)
    with KFACCollector(model, **kw) as kc:
        for b in batches:
            kc.collect(b)
    return kc.finalize(meta)


def layer_chunks(n_layers: int, per_pass: int) -> list[list[int]]:
    return [list(range(i, min(i + per_pass, n_layers))) for i in range(0, n_layers, per_pass)]


def factor_bytes(hidden: int, inter: int, n_heads: int, n_kv: int, head_dim: int, *, shared: bool = True,
                 n_row_sets: int = 2, bytes_per: int = 4) -> int:
    """Memory of one layer's factors for one data side, all row sets (the plan in IMPLEMENTATION.md)."""
    q, kv = n_heads * head_dim, n_kv * head_dim
    a = (1 if shared else 5) * hidden ** 2 + q ** 2 + inter ** 2                 # shared in, o_proj in, down in
    g = q ** 2 + 2 * kv ** 2 + hidden ** 2 + 2 * inter ** 2 + hidden ** 2        # q, k, v, o, gate, up, down
    return (a + g) * n_row_sets * bytes_per


# ---------------------------------------------------------------- exact null directions we know analytically
RESIDUAL_WRITERS = ("o_proj", "down_proj")


def known_null_vectors(model: nn.Module, f) -> dict[tuple[str, str], torch.Tensor]:
    """Directions that are null in exact arithmetic for a Cohere-style model (mean-subtracting LayerNorm
    everywhere, no bias):

    - ("A", "<prefix>layers.L.input"): 1/w, w = that layer's input_layernorm weight. The shared q/k/v/gate/up
      input is w ∘ z with Σz = 0.
    - ("G", "<...>.o_proj" / "<...>.down_proj"): the all-ones vector. These write into the residual stream,
      and every path from it to the loss passes through a mean-subtracting LayerNorm, so ∂L/∂h ⊥ 1.

    Only for model types in PARALLEL_MODEL_TYPES (others return {})."""
    if not is_parallel_block(model):
        return {}
    layers = (_decoder(model)).layers
    out = {}
    for name in f.modules:
        _, l, leaf = parse_name(name)
        key = f.a_key[name]
        if key != name and ("A", key) not in out:
            out[("A", key)] = 1.0 / layers[l].input_layernorm.weight.detach().double().cpu()
        if leaf in RESIDUAL_WRITERS:
            out[("G", name)] = torch.ones(model.get_submodule(name).out_features, dtype=torch.float64)
    return out


def null_floors(f, nulls: dict) -> dict[str, float]:
    """Rayleigh quotient of each known null vector relative to its factor's top eigenvalue, per row set:
    the factors' numerical floor (≈0 in exact arithmetic, ~1e-8..1e-6 for bf16 models with fp32 sums)."""
    out = {}
    for (side, key), u in nulls.items():
        for rs in f.row_sets:
            F = f.get(side, key, rs).double()
            un = (u / u.norm()).to(F)
            top = torch.linalg.eigvalsh(0.5 * (F + F.T)).max()
            out[f"{side}::{key}::{rs}"] = float((un @ F @ un).abs() / top) if top > 0 else 0.0
    return out


def project_out(f: Factors, nulls: dict) -> Factors:
    """Copy of `f` with each known null vector projected out of its factor (F ← P F P, P = I − ûûᵀ), so it is
    exactly null and range restriction removes it regardless of the numerical floor."""
    A = {k: dict(v) for k, v in f.A.items()}
    G = {k: dict(v) for k, v in f.G.items()}
    for (side, key), u in nulls.items():
        store = A if side == "A" else G
        for rs, F in store[key].items():
            store[key][rs] = _project(F, u)
    return Factors(A=A, G=G, a_key=dict(f.a_key), n_rows=dict(f.n_rows), n_loss=f.n_loss, n_seqs=f.n_seqs,
                   D=f.D, meta={**f.meta, "projected_nulls": len(nulls)})


def _project(F: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """P F P with P = I − ûûᵀ, in float64."""
    F = F.double()
    un = (u / u.norm()).to(F)
    Fu = F @ un
    # P F P = F − û(Fû)ᵀ − (Fû)ûᵀ + (ûᵀFû) ûûᵀ, without forming P
    return F - torch.outer(un, Fu) - torch.outer(Fu, un) + (un @ Fu) * torch.outer(un, un)


# ---------------------------------------------------------------- streaming access for real-scale runs
_LAYER = re.compile(r"layers\.(\d+)\.")


def _layer_of(key: str) -> int:
    m = _LAYER.search(key)
    return int(m.group(1)) if m else -1


class FactorStore:
    """Index over saved chunk files of one pass. Tensors are read on demand (safetensors memory-maps them),
    so nothing is loaded until a module asks for it. Same read interface as Factors: pair, get, a_key,
    modules, row_sets, n_rows, n_loss."""

    def __init__(self, files):
        from safetensors import safe_open
        self.files = [Path(f) for f in files]
        if not self.files:
            raise ValueError("no factor files")
        self.index: dict[str, Path] = {}
        metas = []
        for f in self.files:
            with safe_open(str(f), framework="pt") as h:
                metas.append(json.loads(h.metadata()["pencil"]))
                for k in h.keys():
                    self.index[k] = f
        m0 = metas[0]
        for m in metas[1:]:
            if m["n_loss"] != m0["n_loss"] or m["n_rows"] != m0["n_rows"]:
                raise ValueError("chunk files come from different passes (token counts differ)")
        self.a_key = {k: v for m in metas for k, v in m["a_key"].items()}
        self.n_rows, self.n_loss, self.n_seqs = m0["n_rows"], m0["n_loss"], m0["n_seqs"]
        self.meta = m0.get("meta", {})
        self.D = {k.split("::", 1)[1]: None for k in self.index if k.startswith("D::")}   # names only

    @property
    def modules(self) -> list[str]:
        return sorted(self.a_key)

    @property
    def row_sets(self) -> list[str]:
        return sorted(self.n_rows)

    def get(self, kind: str, key: str, row_set: str | None = None) -> torch.Tensor:
        from safetensors import safe_open
        k = f"D::{key}" if kind == "D" else f"{kind}::{key}::{row_set}"
        with safe_open(str(self.index[k]), framework="pt") as h:
            return h.get_tensor(k)

    def pair(self, module: str, row_set: str):
        return self.get("A", self.a_key[module], row_set), self.get("G", module, row_set)

    def dim(self, kind: str, key: str) -> int:
        from safetensors import safe_open
        k = f"{kind}::{key}::{self.row_sets[0]}"
        with safe_open(str(self.index[k]), framework="pt") as h:
            return h.get_slice(k).get_shape()[0]


class LazyFactors:
    """Factors computed on demand, one layer at a time, from one or more sources (FactorStore or Factors):
    a weighted pool (e.g. over languages), then known null vectors projected out, then any transforms
    (the V/O gauge, AlignMerge's Euclidean projection). Results are float64 and cached for the current
    layer only, so process modules in layer order (ModelRepair does)."""

    def __init__(self, sources, weights=None, *, nulls=None, transforms=(), meta=None):
        self.sources = list(sources)
        w = torch.tensor(weights if weights is not None else [1.0] * len(self.sources), dtype=torch.float64)
        self.weights = (w / w.sum()).tolist()
        s0 = self.sources[0]
        for s in self.sources[1:]:
            if set(s.a_key) != set(s0.a_key) or set(s.row_sets) != set(s0.row_sets):
                raise ValueError("cannot pool factors with different modules or row sets")
        self.a_key = dict(s0.a_key)
        self.n_rows = {rs: sum(s.n_rows[rs] for s in self.sources) for rs in s0.row_sets}
        self.n_loss = sum(s.n_loss for s in self.sources)
        self.n_seqs = sum(s.n_seqs for s in self.sources)
        self.nulls = dict(nulls or {})
        self.transforms = list(transforms)
        self.meta = meta or {}
        self.D = s0.D if all(s.D for s in self.sources) else {}
        self._cache: dict = {}
        self._layer = None

    def _derive(self, **kw) -> "LazyFactors":
        out = LazyFactors(self.sources, self.weights, nulls=kw.get("nulls", self.nulls),
                          transforms=kw.get("transforms", self.transforms), meta=self.meta)
        return out

    def with_nulls(self, nulls) -> "LazyFactors":
        return self._derive(nulls=nulls)

    def with_transform(self, fn) -> "LazyFactors":
        return self._derive(transforms=self.transforms + [fn])

    @property
    def modules(self) -> list[str]:
        return sorted(self.a_key)

    @property
    def row_sets(self) -> list[str]:
        return sorted(self.n_rows)

    def get(self, kind: str, key: str, row_set: str | None = None) -> torch.Tensor:
        ck = (kind, key, row_set)
        if ck in self._cache:
            return self._cache[ck]
        layer = _layer_of(key)
        if layer != self._layer:
            self._cache.clear()
            self._layer = layer
        F = sum(w * s.get(kind, key, row_set).double() for w, s in zip(self.weights, self.sources))
        if kind != "D" and (kind, key) in self.nulls:
            F = _project(F, self.nulls[(kind, key)])
        for fn in self.transforms:
            F = fn(kind, key, F)
        self._cache[ck] = F
        return F

    def pair(self, module: str, row_set: str):
        return self.get("A", self.a_key[module], row_set), self.get("G", module, row_set)

    def dim(self, kind: str, key: str) -> int:
        return self.sources[0].dim(kind, key)

    def materialize(self, modules=None) -> Factors:
        """An in-memory Factors with every (or the given) module's factors; for small models and tests."""
        mods = self.modules if modules is None else modules
        A, G = {}, {}
        for n in sorted(mods, key=lambda x: (_layer_of(x), x)):
            for rs in self.row_sets:
                a, g = self.pair(n, rs)
                A.setdefault(self.a_key[n], {})[rs] = a
                G.setdefault(n, {})[rs] = g
        return Factors(A=A, G=G, a_key={n: self.a_key[n] for n in mods}, n_rows=dict(self.n_rows),
                       n_loss=self.n_loss, n_seqs=self.n_seqs, meta=dict(self.meta))
