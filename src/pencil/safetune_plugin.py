"""The pencil repair as a SafeTune recover method, so it runs through the same harness as every baseline.

    from pencil.safetune_plugin import register
    register()                                     # adds the alias "pencil" -> PencilTrainer
    PencilTrainer(drifted, aligned_model=aligned, adapter_path=..., harm_factors=..., task_factors=...,
                  mu=0.1).apply()

Like every SafeTune recover trainer it changes the merged drifted model in place and returns it. ΔW comes
from the fp32 adapter (`adapter_path` or `deltas`); differencing merged bf16 weights is only a fallback, with
a warning (CLAUDE.md rule 1). Factors can be Factors/LazyFactors objects or factor folders.
"""

from __future__ import annotations

import warnings

from .delta import lora_delta, set_weights
from .factor_store import FactorStore, prepare_for_repair
from .repair import ModelRepair

try:
    from safetune.runner.recover._base import _RecoverBase
except ImportError:  # keeps this module importable without SafeTune

    class _RecoverBase:  # type: ignore[no-redef]
        def __init__(self, model=None, **kwargs):
            self.model = model


def _factors(f):
    return FactorStore.from_dir(f) if isinstance(f, str) or hasattr(f, "is_dir") else f


class PencilTrainer(_RecoverBase):
    METHOD = "PencilTrainer"

    def __init__(
        self,
        model=None,
        *,
        aligned_model=None,
        harm_factors=None,
        task_factors=None,
        adapter_path=None,
        deltas=None,
        mu: float = 0.1,
        modules: list[str] | None = None,
        **repair_kw,
    ):
        """repair_kw go to ModelRepair (row_set, damping, gamma, top_k_mlp, rel_tol_other, ...)."""
        super().__init__(model)
        self.aligned_model, self.harm_factors, self.task_factors = aligned_model, harm_factors, task_factors
        self.adapter_path, self.deltas, self.mu, self.modules = adapter_path, deltas, mu, modules
        self.repair_kw = {"row_set": "full", "top_k_mlp": 512, "rel_tol_other": 1e-10, **repair_kw}
        self.report: dict = {}

    def _deltas(self, names):
        if self.deltas is not None:
            return self.deltas
        if self.adapter_path is not None:
            return lora_delta(self.adapter_path, lowrank=True)
        warnings.warn(
            "PencilTrainer: no adapter given, taking ΔW from merged weights (bf16 rounding enters ΔW)", stacklevel=2
        )
        drifted, aligned = self.model.state_dict(), self.aligned_model.state_dict()
        return {n: drifted[n + ".weight"].float() - aligned[n + ".weight"].float() for n in names}

    def apply(self, *, mu: float | None = None, **kwargs):
        if self.aligned_model is None or self.harm_factors is None or self.task_factors is None:
            raise ValueError("PencilTrainer needs aligned_model, harm_factors and task_factors")
        fh, ft, rel_tol, _ = prepare_for_repair(
            self.aligned_model, _factors(self.harm_factors), _factors(self.task_factors)
        )
        deltas = {n: d for n, d in self._deltas(fh.modules).items() if n in fh.a_key}
        aligned = self.aligned_model.state_dict()
        W0 = {n: aligned[n + ".weight"] for n in deltas}
        repairer = ModelRepair(fh, ft, rel_tol=rel_tol, **self.repair_kw)
        new, self.report = repairer.repair(W0, deltas, self.mu if mu is None else mu, modules=self.modules)
        self.report["rel_tol"] = rel_tol
        set_weights(self.model, new)
        return self.model


def register() -> None:
    """Make "pencil" a SafeTune recover alias (the CLI looks classes up on safetune.runner.recover)."""
    import safetune.runner.recover as recover
    from safetune.runner._registry import register_recover

    recover.PencilTrainer = PencilTrainer
    register_recover("pencil", "PencilTrainer")
