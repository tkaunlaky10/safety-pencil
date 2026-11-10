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


def _key(kind: str, name: str, row_set: str | None) -> str:
    """Tensor name inside a factor file: "A::<a_key>::<row_set>", "G::<module>::<row_set>" or "D::<module>"."""
    return f"D::{name}" if kind == "D" else f"{kind}::{name}::{row_set}"


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
        if kind == "D":
            return self.D[name]
        return (self.A if kind == "A" else self.G)[name][row_set]

    def pair(self, module: str, row_set: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.A[self.a_key[module]][row_set], self.G[module][row_set]

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        tensors = {_key("A", k, rs): t for k, by_rs in self.A.items() for rs, t in by_rs.items()}
        tensors |= {_key("G", k, rs): t for k, by_rs in self.G.items() for rs, t in by_rs.items()}
        tensors |= {_key("D", k, None): t for k, t in self.D.items()}
        info = {
            "a_key": self.a_key,
            "n_rows": self.n_rows,
            "n_loss": self.n_loss,
            "n_seqs": self.n_seqs,
            "meta": self.meta,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: t.contiguous().cpu() for k, t in tensors.items()}, str(path), metadata={"pencil": json.dumps(info)}
        )


class FactorStore:
    """All chunk files of one factor pass, indexed by tensor name; tensors are read on demand."""

    def __init__(self, files):
        from safetensors import safe_open

        self.files = [Path(f) for f in files]
        if not self.files:
            raise ValueError("no factor files")
        self.index: dict[str, Path] = {}
        infos = []
        for f in self.files:
            with safe_open(str(f), framework="pt") as h:
                infos.append(json.loads(h.metadata()["pencil"]))
                self.index |= dict.fromkeys(h.keys(), f)
        first = infos[0]
        if any(i["n_loss"] != first["n_loss"] or i["n_rows"] != first["n_rows"] for i in infos):
            raise ValueError("chunk files come from different passes (token counts differ)")
        self.a_key = {k: v for i in infos for k, v in i["a_key"].items()}
        self.n_rows, self.n_loss, self.n_seqs = first["n_rows"], first["n_loss"], first["n_seqs"]
        self.meta = first.get("meta", {})
        self.has_diag = any(k.startswith("D::") for k in self.index)

    @classmethod
    def from_dir(cls, folder: str | Path) -> FactorStore:
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
        from safetensors import safe_open

        k = _key(kind, name, row_set)
        with safe_open(str(self.index[k]), framework="pt") as h:
            return h.get_tensor(k)

    def pair(self, module: str, row_set: str):
        return self.get("A", self.a_key[module], row_set), self.get("G", module, row_set)


class LazyFactors:
    """Pool of sources (FactorStore or Factors) with known null vectors projected out and optional transforms,
    computed on demand in float64. Only the current layer is cached, so read modules in layer order."""

    def __init__(self, sources, weights=None, *, nulls=None, transforms=(), meta=None):
        self.sources = list(sources)
        w = torch.tensor(weights if weights is not None else [1.0] * len(self.sources), dtype=torch.float64)
        self.weights = (w / w.sum()).tolist()
        first = self.sources[0]
        if any(set(s.a_key) != set(first.a_key) or set(s.row_sets) != set(first.row_sets) for s in self.sources):
            raise ValueError("cannot pool factors with different modules or row sets")
        self.a_key = dict(first.a_key)
        self.n_rows = {rs: sum(s.n_rows[rs] for s in self.sources) for rs in first.row_sets}
        self.n_loss = sum(s.n_loss for s in self.sources)
        self.nulls = dict(nulls or {})
        self.transforms = list(transforms)
        self.meta = meta or {}
        self.has_diag = all(s.has_diag for s in self.sources)
        self._cache: dict = {}
        self._cached_layer = None

    def with_nulls(self, nulls: dict) -> LazyFactors:
        return LazyFactors(self.sources, self.weights, nulls=nulls, transforms=self.transforms, meta=self.meta)

    def with_transform(self, fn) -> LazyFactors:
        """fn(kind, name, F) -> F', applied after pooling and null projection."""
        return LazyFactors(
            self.sources, self.weights, nulls=self.nulls, transforms=[*self.transforms, fn], meta=self.meta
        )

    @property
    def modules(self) -> list[str]:
        return sorted(self.a_key)

    @property
    def row_sets(self) -> list[str]:
        return sorted(self.n_rows)

    def get(self, kind: str, name: str, row_set: str | None = None) -> torch.Tensor:
        cache_key = (kind, name, row_set)
        if cache_key not in self._cache:
            if layer_of(name) != self._cached_layer:
                self._cache.clear()
                self._cached_layer = layer_of(name)
            F = sum(w * s.get(kind, name, row_set).double() for w, s in zip(self.weights, self.sources))
            if (kind, name) in self.nulls:
                F = project_out(F, self.nulls[(kind, name)])
            for fn in self.transforms:
                F = fn(kind, name, F)
            self._cache[cache_key] = F
        return self._cache[cache_key]

    def pair(self, module: str, row_set: str):
        return self.get("A", self.a_key[module], row_set), self.get("G", module, row_set)


def as_lazy(f) -> LazyFactors:
    return f if isinstance(f, LazyFactors) else LazyFactors([f])


# ---------------------------------------------------------------- exact null directions known analytically
def known_null_vectors(model: nn.Module, f) -> dict[tuple[str, str], torch.Tensor]:
    """Directions that are null in exact arithmetic for a Cohere-style model (mean-subtracting LayerNorm,
    no biases):
    - ("A", "<...>layers.L.input"): 1/w, w = that layer's input_layernorm weight (its output is w ∘ z, Σz = 0);
    - ("G", "<...>.o_proj" and "<...>.down_proj"): all ones. They write into the residual stream, and every
      path from there to the loss passes a mean-subtracting LayerNorm, so ∂L/∂h ⊥ 1.
    In fp32 factors these are null only to ~1e-8, so project them out (LazyFactors.with_nulls)."""
    if not is_parallel_block(model):
        return {}
    layers = decoder(model).layers
    out = {}
    for name in f.modules:
        _, layer, lf = parse_name(name)
        key = f.a_key[name]
        if key != name:
            out[("A", key)] = 1.0 / layers[layer].input_layernorm.weight.detach().double().cpu()
        if lf in RESIDUAL_WRITERS:
            out[("G", name)] = torch.ones(model.get_submodule(name).out_features, dtype=torch.float64)
    return out


def null_floors(f, nulls: dict) -> dict[str, float]:
    """Rayleigh quotient of each known null vector over its factor's top eigenvalue, per row set: the
    factors' numerical floor (0 in exact arithmetic). geneig.auto_rel_tol turns it into a range threshold."""
    from .geneig import noise_floor

    return {
        f"{kind}::{name}::{rs}": noise_floor(f.get(kind, name, rs), u)
        for (kind, name), u in nulls.items()
        for rs in f.row_sets
    }


def project_out(F: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """P F P with P = I − ûûᵀ, in float64, without forming P."""
    F = F.double()
    u_hat = (u / u.norm()).to(F)
    Fu = F @ u_hat
    return F - torch.outer(u_hat, Fu) - torch.outer(Fu, u_hat) + (u_hat @ Fu) * torch.outer(u_hat, u_hat)


def prepare_for_repair(model: nn.Module, fac_h, fac_t, *, base_rel_tol: float = 1e-10, floor_factor: float = 10.0):
    """Both sides as LazyFactors with the known null vectors projected out, and the range threshold for
    those factors: max(base_rel_tol, floor_factor × the largest measured floor).
    Returns (fac_h, fac_t, rel_tol, floors)."""
    from .geneig import auto_rel_tol

    fh, ft = as_lazy(fac_h), as_lazy(fac_t)
    nulls = known_null_vectors(model, fh)
    floors = {f"harm::{k}": v for k, v in null_floors(fh, nulls).items()}
    floors |= {f"task::{k}": v for k, v in null_floors(ft, nulls).items()}
    rel_tol = auto_rel_tol(floors.values(), base_rel_tol, floor_factor)
    return fh.with_nulls(nulls), ft.with_nulls(nulls), rel_tol, floors
