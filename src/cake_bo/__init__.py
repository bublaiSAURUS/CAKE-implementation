"""CAKE: LLM-guided kernel evolution for Bayesian optimization."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .baker import BAKER, BAKERResult, KernelRanking
    from .evolution import BASE_KERNELS, OPERATORS, CAKE, KernelResult
    from .optimizer import CAKEOptimizer, IterationRecord, OptimizationResult

_EXPORTS = {
    "BAKER": (".baker", "BAKER"),
    "BAKERResult": (".baker", "BAKERResult"),
    "KernelRanking": (".baker", "KernelRanking"),
    "BASE_KERNELS": (".evolution", "BASE_KERNELS"),
    "CAKE": (".evolution", "CAKE"),
    "KernelResult": (".evolution", "KernelResult"),
    "OPERATORS": (".evolution", "OPERATORS"),
    "CAKEOptimizer": (".optimizer", "CAKEOptimizer"),
    "IterationRecord": (".optimizer", "IterationRecord"),
    "OptimizationResult": (".optimizer", "OptimizationResult"),
}


def __getattr__(name: str) -> Any:
    """Load public objects lazily so parser utilities stay lightweight."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
