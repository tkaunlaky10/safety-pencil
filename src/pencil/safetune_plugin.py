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


# ModelRepair settings used unless the caller overrides them
DEFAULT_REPAIR_KW = {"row_set": "full", "top_k_mlp": 512, "rel_tol_other": 1e-10}


def _as_factors(factors_or_folder):
    """A factor folder (a str or Path) opened as a FactorStore; factor objects are returned unchanged."""
    is_folder = isinstance(factors_or_folder, str) or hasattr(factors_or_folder, "is_dir")
    if is_folder:
        return FactorStore.from_dir(factors_or_folder)
    return factors_or_folder


class PencilTrainer(_RecoverBase):
    """SafeTune recover trainer for the pencil repair: `apply()` replaces each repaired module's weight in the
    drifted model by W0 + ΔW′ and keeps the repair report in `self.report`."""

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
        self.aligned_model = aligned_model
        self.harm_factors = harm_factors
        self.task_factors = task_factors
        self.adapter_path = adapter_path
        self.deltas = deltas
        self.mu = mu
        self.modules = modules
        self.repair_kw = {**DEFAULT_REPAIR_KW, **repair_kw}
        self.report: dict = {}

    def _deltas(self, names):
        """ΔW per module: the given deltas, else the adapter's, else (with a warning) drifted minus aligned."""
        if self.deltas is not None:
            return self.deltas
        if self.adapter_path is not None:
            return lora_delta(self.adapter_path, lowrank=True)
        warnings.warn(
            "PencilTrainer: no adapter given, taking ΔW from merged weights (bf16 rounding enters ΔW)", stacklevel=2
        )
        drifted = self.model.state_dict()
        aligned = self.aligned_model.state_dict()
        return {name: drifted[name + ".weight"].float() - aligned[name + ".weight"].float() for name in names}

    def apply(self, *, mu: float | None = None, **kwargs):
        """Repair the drifted model in place at μ (default: the constructor's) and return it. Extra keyword
        arguments from SafeTune's harness are accepted and ignored."""
        if self.aligned_model is None or self.harm_factors is None or self.task_factors is None:
            raise ValueError("PencilTrainer needs aligned_model, harm_factors and task_factors")
        harm, task, rel_tol, _ = prepare_for_repair(
            self.aligned_model, _as_factors(self.harm_factors), _as_factors(self.task_factors)
        )
        all_deltas = self._deltas(harm.modules)
        deltas = {module: delta for module, delta in all_deltas.items() if module in harm.a_key}
        aligned = self.aligned_model.state_dict()
        W0 = {module: aligned[module + ".weight"] for module in deltas}

        if mu is None:
            mu = self.mu
        repairer = ModelRepair(harm, task, rel_tol=rel_tol, **self.repair_kw)
        new_weights, self.report = repairer.repair(W0, deltas, mu, modules=self.modules)
        self.report["rel_tol"] = rel_tol
        set_weights(self.model, new_weights)
        return self.model


def register() -> None:
    """Make "pencil" a SafeTune recover alias (the CLI looks classes up on safetune.runner.recover)."""
    import safetune.runner.recover as recover
    from safetune.runner._registry import register_recover

    recover.PencilTrainer = PencilTrainer
    register_recover("pencil", "PencilTrainer")
