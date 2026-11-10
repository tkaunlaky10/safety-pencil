"""The pencil repair as a SafeTune recover method, so it runs through the same harness as every baseline.

    from pencil.safetune_plugin import register
    register()                                  # adds alias "pencil" -> PencilTrainer
    PencilTrainer(drifted, aligned_model=aligned, adapter_path=..., harm_factors=[...], task_factors=[...],
                  mu=0.1).apply()

SafeTune recover trainers take the merged drifted model, change it in place and return it
(`safetune/runner/recover/_base.py`). ΔW is rebuilt from the fp32 adapter when `adapter_path` (or `deltas`)
is given; differencing merged bf16 weights is only a fallback, with a warning (CLAUDE.md rule 1).
"""
from __future__ import annotations

import warnings

import torch

from .delta import lora_delta
from .factors import FactorStore, Factors, LazyFactors, known_null_vectors, null_floors, project_out
from .geneig import auto_rel_tol
from .repair import ModelRepair

try:
    from safetune.runner.recover._base import _RecoverBase
except ImportError:                              # SafeTune not installed: keep the class importable for docs/tests
    class _RecoverBase:                          # type: ignore[no-redef]
        PILLAR = "recover"
        METHOD = ""

        def __init__(self, model=None, **kwargs):
            self.model = model


def _load(f):
    """Factors / LazyFactors pass through; a file, a list of chunk files or a folder becomes a streaming view."""
    if f is None or isinstance(f, (Factors, LazyFactors)):
        return f
    from pathlib import Path
    files = [Path(x) for x in f] if isinstance(f, (list, tuple)) else (
        sorted(Path(f).glob("layers*.safetensors")) if Path(f).is_dir() else [Path(f)])
    return LazyFactors([FactorStore(files)])


class PencilTrainer(_RecoverBase):
    METHOD = "PencilTrainer"

    def __init__(self, model=None, *, aligned_model=None, adapter_path=None, deltas=None, harm_factors=None,
                 task_factors=None, mu: float = 0.1, row_set="full", damping: str = "covariant",
                 gamma: float = 1e-3, top_k_mlp: int | None = 512, rel_tol: float = 1e-10,
                 rel_tol_other: float = 1e-10,
                 rel_tol_floor_factor: float | None = 10.0, project_known_nulls: bool = True,
                 modules: list[str] | None = None, device: str = "cpu", **kwargs):
        super().__init__(model, **kwargs)
        self.aligned_model = aligned_model
        self.adapter_path, self.deltas = adapter_path, deltas
        self.harm_factors, self.task_factors = harm_factors, task_factors
        self.mu, self.row_set, self.damping, self.gamma = mu, row_set, damping, gamma
        self.top_k_mlp, self.rel_tol, self.rel_tol_floor_factor = top_k_mlp, rel_tol, rel_tol_floor_factor
        self.rel_tol_other = rel_tol_other
        self.project_known_nulls, self.modules, self.device = project_known_nulls, modules, device
        self.report: dict = {}

    def _deltas(self, model, aligned_sd, names_hint):
        if self.deltas is not None:
            return self.deltas
        if self.adapter_path is not None:
            return lora_delta(self.adapter_path, lowrank=True)
        warnings.warn("PencilTrainer: no adapter given, taking ΔW from merged weights (bf16 rounding enters ΔW)")
        sd = model.state_dict()
        return {n: sd[n + ".weight"].float() - aligned_sd[n + ".weight"].float() for n in names_hint}

    def apply(self, *, mu: float | None = None, **kwargs):
        if self.aligned_model is None:
            raise ValueError("PencilTrainer needs aligned_model (θ0)")
        fh, ft = _load(self.harm_factors), _load(self.task_factors)
        if fh is None or ft is None:
            raise ValueError("PencilTrainer needs harm_factors and task_factors")
        model = self.model
        aligned_sd = self.aligned_model.state_dict()
        deltas = self._deltas(model, aligned_sd, fh.modules)
        deltas = {n: d for n, d in deltas.items() if n in fh.a_key}
        rel_tol = self.rel_tol
        if self.project_known_nulls:
            nulls = known_null_vectors(self.aligned_model, fh)
            floors = list(null_floors(fh, nulls).values()) + list(null_floors(ft, nulls).values())
            if self.rel_tol_floor_factor:
                rel_tol = auto_rel_tol(floors, self.rel_tol, self.rel_tol_floor_factor)
            proj = lambda x: x.with_nulls(nulls) if isinstance(x, LazyFactors) else project_out(x, nulls)
            fh, ft = proj(fh), proj(ft)
        rep = ModelRepair(fh, ft, row_set=self.row_set, damping=self.damping, gamma=self.gamma,
                          top_k_mlp=self.top_k_mlp, rel_tol=rel_tol, rel_tol_other=self.rel_tol_other,
                          device=self.device)
        W0 = {n: aligned_sd[n + ".weight"] for n in deltas}
        new, report = rep.repair(W0, deltas, self.mu if mu is None else mu, modules=self.modules)
        mods = dict(model.named_modules())
        with torch.no_grad():
            for n, W in new.items():
                mods[n].weight.copy_(W.to(mods[n].weight))
        report["rel_tol"] = rel_tol
        self.report = report
        return model


def register() -> None:
    """Make "pencil" a SafeTune recover alias (CLI lookup is getattr(safetune.runner.recover, name))."""
    import safetune.runner.recover as RR
    from safetune.runner._registry import register_recover
    RR.PencilTrainer = PencilTrainer
    register_recover("pencil", "PencilTrainer")
