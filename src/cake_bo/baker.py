"""BIC-Acquisition Kernel Ranking (BAKER)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf

from .evolution import KernelResult

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class KernelRanking:
    """BAKER metrics for one successfully optimized kernel."""

    kernel: str
    candidate: torch.Tensor
    bic: float
    bic_weight: float
    acquisition_value: float
    normalized_acquisition: float
    score: float


@dataclass(frozen=True)
class BAKERResult:
    """The selected query and complete ranking for one BO iteration."""

    candidate: torch.Tensor
    kernel: str
    score: float
    rankings: tuple[KernelRanking, ...]


class BAKER:
    """Rank kernel-specific EI candidates using BIC model weights.

    Expected-improvement values are normalized by the largest successful EI,
    placing them in ``[0, 1]`` while preserving their ratios. If every EI is
    zero, all normalized values are set to one and selection falls back to BIC.
    """

    def __init__(
        self,
        num_restarts: int | None = None,
        raw_samples: int | None = None,
    ) -> None:
        if num_restarts is not None and num_restarts < 1:
            raise ValueError("num_restarts must be positive")
        if raw_samples is not None and raw_samples < 1:
            raise ValueError("raw_samples must be positive")
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def select(
        self,
        population: Mapping[str, KernelResult | None],
        train_y: torch.Tensor,
        bounds: torch.Tensor,
    ) -> BAKERResult:
        """Optimize EI for every GP and return the highest weighted candidate."""
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("bounds must have shape (2, d)")
        if not torch.all(bounds[0] < bounds[1]):
            raise ValueError("Each lower bound must be smaller than its upper bound")
        if train_y.numel() == 0:
            raise ValueError("train_y must not be empty")
        evaluated = [
            (expression, result)
            for expression, result in population.items()
            if result is not None
        ]
        if not evaluated:
            raise ValueError("BAKER requires an evaluated kernel population")

        bics = torch.tensor(
            [result.bic for _, result in evaluated],
            dtype=torch.float64,
            device=bounds.device,
        )
        bic_weights = torch.softmax(-bics, dim=0)
        dimension = bounds.shape[1]
        restarts = self.num_restarts or 20 * dimension
        samples = self.raw_samples or 50 * dimension
        successful: list[tuple[str, KernelResult, torch.Tensor, float, float]] = []

        for index, (expression, result) in enumerate(evaluated):
            acquisition = ExpectedImprovement(
                model=result.model,
                best_f=train_y.max(),
                maximize=True,
            )
            try:
                candidate, value = optimize_acqf(
                    acq_function=acquisition,
                    bounds=bounds,
                    q=1,
                    num_restarts=restarts,
                    raw_samples=samples,
                    retry_on_optimization_warning=True,
                )
            except (RuntimeError, ValueError) as error:
                LOGGER.warning("BAKER could not optimize %s: %s", expression, error)
                continue
            successful.append(
                (
                    expression,
                    result,
                    candidate.detach(),
                    float(value.detach().reshape(-1)[0]),
                    float(bic_weights[index].item()),
                )
            )

        if not successful:
            raise RuntimeError("BAKER failed to optimize every kernel candidate")
        max_acquisition = max(item[3] for item in successful)
        rankings: list[KernelRanking] = []
        for expression, result, candidate, acquisition_value, weight in successful:
            normalized = (
                acquisition_value / max_acquisition if max_acquisition > 0.0 else 1.0
            )
            rankings.append(
                KernelRanking(
                    kernel=expression,
                    candidate=candidate,
                    bic=result.bic,
                    bic_weight=weight,
                    acquisition_value=acquisition_value,
                    normalized_acquisition=normalized,
                    score=weight * normalized,
                )
            )
        rankings.sort(key=lambda item: item.score, reverse=True)
        winner = rankings[0]
        return BAKERResult(
            candidate=winner.candidate,
            kernel=winner.kernel,
            score=winner.score,
            rankings=tuple(rankings),
        )


__all__ = ["BAKER", "BAKERResult", "KernelRanking"]
