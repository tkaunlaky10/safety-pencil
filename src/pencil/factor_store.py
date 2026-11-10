"""Holding, saving and reading K-FAC factors.

- `Factors`: one pass's normalised factors in memory, saved as one safetensors file per layer chunk.
- `FactorStore`: an index over a pass's chunk files that reads each tensor only when asked for it.
- `LazyFactors`: the view the repair uses. It pools several sources (e.g. languages), projects out the
  known null vectors and applies transforms (the V/O gauge, AlignMerge's projection), one layer at a time.

All three answer `pair(module, row_set) -> (A, G)`, `get(kind, key, row_set)`, `a_key`, `modules` and
`row_sets`, so the repair code doesn't care which one it is given.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .modules import RESIDUAL_WRITERS, decoder, is_parallel_block, layer_of, parse_name


def _tensor_name(kind: str, name: str, row_set: str | None) -> str:
    """Tensor name inside a factor file: "A::<a_key>::<row_set>", "G::<module>::<row_set>" or "D::<module>"."""
    if kind == "D":
        return f"D::{name}"
    return f"{kind}::{name}::{row_set}"


@dataclass
class Factors:
    """A[a_key][row_set] and G[module][row_set]; a_key[module] says which input factor a module uses.
    A is normalised by its row count, G by the number of loss tokens. D is the optional diagonal Fisher."""

    A: dict[str, dict[str, torch.Tensor]]
    G: dict[str, dict[str, torch.Tensor]]
    a_key: dict[str, str]
    n_rows: dict[str, int]
    n_loss: int
    n_seqs: int
    D: dict[str, torch.Tensor] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def modules(self) -> list[str]:
        return sorted(self.G)

    @property
    def row_sets(self) -> list[str]:
        return sorted(self.n_rows)

    @property
    def has_diag(self) -> bool:
        return bool(self.D)

    def get(self, kind: str, name: str, row_set: str | None = None) -> torch.Tensor:
        """One factor: kind "A" (name is an a_key), "G" (name is a module) or "D" (no row set)."""
        if kind == "D":
            return self.D[name]
        if kind == "A":
            return self.A[name][row_set]
        return self.G[name][row_set]

    def pair(self, module: str, row_set: str) -> tuple[torch.Tensor, torch.Tensor]:
        """(A, G) for a module."""
        return self.A[self.a_key[module]][row_set], self.G[module][row_set]

    def save(self, path: str | Path) -> None:
        """Write every factor to one safetensors file, with the counts and a_key in its metadata."""
        from safetensors.torch import save_file

        info = {
            "a_key": self.a_key,
            "n_rows": self.n_rows,
            "n_loss": self.n_loss,
            "n_seqs": self.n_seqs,
            "meta": self.meta,
        }
        tensors = {}
        for name, tensor in self._named_tensors().items():
            tensors[name] = tensor.contiguous().cpu()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata={"pencil": json.dumps(info)})

    def _named_tensors(self) -> dict[str, torch.Tensor]:
        """{tensor name in the file: tensor} for every A, G and D factor."""
        tensors = {}
        for input_key, by_row_set in self.A.items():
            for row_set, tensor in by_row_set.items():
                tensors[_tensor_name("A", input_key, row_set)] = tensor
        for module, by_row_set in self.G.items():
            for row_set, tensor in by_row_set.items():
                tensors[_tensor_name("G", module, row_set)] = tensor
        for module, tensor in self.D.items():
            tensors[_tensor_name("D", module, None)] = tensor
        return tensors


class FactorStore:
    """All chunk files of one factor pass, indexed by tensor name; tensors are read on demand."""

    def __init__(self, files):
        self.files = [Path(file) for file in files]
        if not self.files:
            raise ValueError("no factor files")
        self.index: dict[str, Path] = {}  # tensor name -> the chunk file that holds it
        headers = []
        for path in self.files:
            header, tensor_names = _read_header(path)
            headers.append(header)
            for tensor_name in tensor_names:
                self.index[tensor_name] = path

        first = headers[0]
        for header in headers:
            if header["n_loss"] != first["n_loss"] or header["n_rows"] != first["n_rows"]:
                raise ValueError("chunk files come from different passes (token counts differ)")
        self.a_key = {}
        for header in headers:
            self.a_key.update(header["a_key"])
        self.n_rows = first["n_rows"]
        self.n_loss = first["n_loss"]
        self.n_seqs = first["n_seqs"]
        self.meta = first.get("meta", {})
        self.has_diag = any(tensor_name.startswith("D::") for tensor_name in self.index)

    @classmethod
    def from_dir(cls, folder: str | Path) -> FactorStore:
        """The store for every layers*.safetensors file in a folder."""
        files = sorted(Path(folder).glob("layers*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no factor files in {folder}; run scripts/04_factors.py first")
        return cls(files)

    @property
    def modules(self) -> list[str]:
        return sorted(self.a_key)

    @property
    def row_sets(self) -> list[str]:
        return sorted(self.n_rows)

    def get(self, kind: str, name: str, row_set: str | None = None) -> torch.Tensor:
        """Read one factor from its chunk file (see Factors.get for the arguments)."""
        from safetensors import safe_open

        tensor_name = _tensor_name(kind, name, row_set)
        with safe_open(str(self.index[tensor_name]), framework="pt") as handle:
            return handle.get_tensor(tensor_name)

    def pair(self, module: str, row_set: str):
        """(A, G) for a module."""
        return self.get("A", self.a_key[module], row_set), self.get("G", module, row_set)


def _read_header(path: Path) -> tuple[dict, list[str]]:
    """(the pass information saved by Factors.save, the tensor names) of one chunk file."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        header = json.loads(handle.metadata()["pencil"])
        tensor_names = list(handle.keys())
    return header, tensor_names


class LazyFactors:
    """A weighted pool of sources (FactorStore or Factors) with known null vectors projected out and
    optional transforms, computed on demand in float64.

    get(kind, name, row_set) builds a factor in three steps:
        1. pool:       F = Σ_s w_s F_s over the sources (weights normalised to sum to 1)
        2. project:    F ← P F P with P = I − ûûᵀ, if (kind, name) has a known null vector u
        3. transform:  F ← fn(kind, name, F) for each transform, in order
    Results are cached only for the most recent layer (the whole model's factors don't fit in memory), so
    read modules in layer order.

    `with_nulls` and `with_transform` return new views over the same sources, with an empty cache."""

    def __init__(self, sources, weights=None, *, nulls=None, transforms=(), meta=None):
        self.sources = list(sources)
        if weights is None:
            weights = [1.0] * len(self.sources)
        raw_weights = torch.tensor(weights, dtype=torch.float64)
        self.weights = (raw_weights / raw_weights.sum()).tolist()
        _check_poolable(self.sources)
        first = self.sources[0]
        self.a_key = dict(first.a_key)
        self.n_rows = {}
        for row_set in first.row_sets:
            self.n_rows[row_set] = sum(source.n_rows[row_set] for source in self.sources)
        self.n_loss = sum(source.n_loss for source in self.sources)
        self.nulls = {} if nulls is None else dict(nulls)  # (kind, name) -> null vector
        self.transforms = list(transforms)
        self.meta = meta if meta else {}
        self.has_diag = all(source.has_diag for source in self.sources)
        self._cache: dict = {}  # (kind, name, row_set) -> factor, for _cached_layer only
        self._cached_layer = None

    def with_nulls(self, nulls: dict) -> LazyFactors:
        """The same pool with these known null vectors projected out."""
        return LazyFactors(self.sources, self.weights, nulls=nulls, transforms=self.transforms, meta=self.meta)

    def with_transform(self, fn) -> LazyFactors:
        """The same pool with one more transform fn(kind, name, F) -> F′, applied after pooling and null
        projection."""
        transforms = [*self.transforms, fn]
        return LazyFactors(self.sources, self.weights, nulls=self.nulls, transforms=transforms, meta=self.meta)

    @property
    def modules(self) -> list[str]:
        return sorted(self.a_key)

    @property
    def row_sets(self) -> list[str]:
        return sorted(self.n_rows)

    def get(self, kind: str, name: str, row_set: str | None = None) -> torch.Tensor:
        """One factor, pooled, projected and transformed (see the class docstring), in float64."""
        cache_key = (kind, name, row_set)
        if cache_key in self._cache:
            return self._cache[cache_key]
        layer = layer_of(name)
        if layer != self._cached_layer:
            self._cache.clear()
            self._cached_layer = layer

        F = self._pooled(kind, name, row_set)
        if (kind, name) in self.nulls:
            F = project_out(F, self.nulls[(kind, name)])
        for transform in self.transforms:
            F = transform(kind, name, F)
        self._cache[cache_key] = F
        return F

    def _pooled(self, kind: str, name: str, row_set: str | None) -> torch.Tensor:
        """Σ_s w_s F_s over the sources, in float64."""
        pooled = 0
        for weight, source in zip(self.weights, self.sources):
            pooled = pooled + weight * source.get(kind, name, row_set).double()
        return pooled

    def pair(self, module: str, row_set: str):
        """(A, G) for a module."""
        return self.get("A", self.a_key[module], row_set), self.get("G", module, row_set)


def _check_poolable(sources: list) -> None:
    """Raise unless every source has the same modules and row sets as the first."""
    first = sources[0]
    for source in sources:
        if set(source.a_key) != set(first.a_key) or set(source.row_sets) != set(first.row_sets):
            raise ValueError("cannot pool factors with different modules or row sets")


def as_lazy(factors) -> LazyFactors:
    """`factors` as a LazyFactors view (unchanged if it already is one)."""
    if isinstance(factors, LazyFactors):
        return factors
    return LazyFactors([factors])


# ---------------------------------------------------------------- exact null directions known analytically
def known_null_vectors(model: nn.Module, factors) -> dict[tuple[str, str], torch.Tensor]:
    """{(kind, name): u} for the directions that are null in exact arithmetic in a Cohere-style model
    (parallel block, mean-subtracting LayerNorm, no biases):

    - ("A", "<...>layers.L.input"): u = 1/w, with w that layer's input_layernorm weight. The norm's output is
      w ∘ z with Σz = 0, so every shared input a satisfies aᵀ(1/w) = 0.
    - ("G", "<...>.o_proj") and ("G", "<...>.down_proj"): u = all ones. These write into the residual stream,
      and every path from there to the loss passes a mean-subtracting LayerNorm, so ∂L/∂h ⊥ 1.

    In fp32 factors these are null only to ~1e-8, so project them out (LazyFactors.with_nulls)."""
    if not is_parallel_block(model):
        return {}
    layers = decoder(model).layers
    nulls = {}
    for module in factors.modules:
        _, layer, module_leaf = parse_name(module)
        input_key = factors.a_key[module]
        if input_key != module:  # a shared input factor
            norm_weight = layers[layer].input_layernorm.weight
            nulls[("A", input_key)] = 1.0 / norm_weight.detach().double().cpu()
        if module_leaf in RESIDUAL_WRITERS:
            width = model.get_submodule(module).out_features
            nulls[("G", module)] = torch.ones(width, dtype=torch.float64)
    return nulls


def null_floors(factors, nulls: dict) -> dict[str, float]:
    """{"<kind>::<name>::<row set>": floor}: each known null vector's Rayleigh quotient over its factor's top
    eigenvalue, per row set. The factors' numerical floor (0 in exact arithmetic); geneig.auto_rel_tol turns
    it into a range threshold."""
    from .geneig import noise_floor

    floors = {}
    for (kind, name), u in nulls.items():
        for row_set in factors.row_sets:
            floors[f"{kind}::{name}::{row_set}"] = noise_floor(factors.get(kind, name, row_set), u)
    return floors


def project_out(F: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """P F P with P = I − ûûᵀ, in float64, without forming P:
    P F P = F − û(Fû)ᵀ − (Fû)ûᵀ + (ûᵀFû) ûûᵀ."""
    F = F.double()
    u_hat = (u / u.norm()).to(F)
    Fu = F @ u_hat
    return F - torch.outer(u_hat, Fu) - torch.outer(Fu, u_hat) + (u_hat @ Fu) * torch.outer(u_hat, u_hat)


def prepare_for_repair(
    model: nn.Module, harm_factors, task_factors, *, base_rel_tol: float = 1e-10, floor_factor: float = 10.0
):
    """Both sides as LazyFactors with the known null vectors projected out, and the range threshold for those
    factors: max(base_rel_tol, floor_factor × the largest measured floor, over both sides).
    Returns (harm_factors, task_factors, rel_tol, floors)."""
    from .geneig import auto_rel_tol

    harm = as_lazy(harm_factors)
    task = as_lazy(task_factors)
    nulls = known_null_vectors(model, harm)
    floors = {}
    for name, floor in null_floors(harm, nulls).items():
        floors[f"harm::{name}"] = floor
    for name, floor in null_floors(task, nulls).items():
        floors[f"task::{name}"] = floor
    rel_tol = auto_rel_tol(floors.values(), base_rel_tol, floor_factor)
    return harm.with_nulls(nulls), task.with_nulls(nulls), rel_tol, floors
