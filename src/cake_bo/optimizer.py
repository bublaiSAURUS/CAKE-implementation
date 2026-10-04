"""Complete CAKE + BAKER Bayesian-optimization loop."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .baker import BAKER, BAKERResult
from .evolution import CAKE

LOGGER = logging.getLogger(__name__)
Objective = Callable[[torch.Tensor], torch.Tensor | float]


@dataclass(frozen=True)
class IterationRecord:
    """Trace information from one sequential BO evaluation."""

    iteration: int
    candidate: torch.Tensor
    observation: float
    kernel: str
    baker_score: float
    best_value: float
    baker_result: BAKERResult


@dataclass(frozen=True)
class OptimizationResult:
    """Observations and trace returned by the full optimizer."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    best_x: torch.Tensor
    best_y: float
    history: tuple[IterationRecord, ...]


class CAKEOptimizer:
    """Maximize a black-box objective with CAKE and BAKER.

    ``budget`` is the total number of objective evaluations, including the
    initial design. Minimization problems can be handled by negating their
    objective values.
    """

    def __init__(
        self,
        cake: CAKE,
        baker: BAKER | None = None,
        random_seed: int | None = None,
    ) -> None:
        self.cake = cake
        self.baker = baker or BAKER()
        self.random_seed = random_seed

    @staticmethod
    def _validate_bounds(bounds: torch.Tensor) -> None:
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("bounds must have shape (2, d)")
        if bounds.shape[1] < 1:
            raise ValueError("bounds must contain at least one dimension")
        if not torch.isfinite(bounds).all():
            raise ValueError("bounds must be finite")
        if not torch.all(bounds[0] < bounds[1]):
            raise ValueError("Each lower bound must be smaller than its upper bound")

    @staticmethod
    def _evaluate(objective: Objective, point: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(
            objective(point.reshape(1, -1)),
            dtype=point.dtype,
            device=point.device,
        ).reshape(-1)
        if value.numel() != 1:
            raise ValueError("The objective must return one scalar per queried point")
        if not torch.isfinite(value[0]):
            raise ValueError("The objective returned a non-finite value")
        return value[0]

    def _initial_design(
        self,
        objective: Objective,
        bounds: torch.Tensor,
        num_initial: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unit_points = torch.quasirandom.SobolEngine(
            dimension=bounds.shape[1],
            scramble=True,
            seed=self.random_seed,
        ).draw(num_initial)
        unit_points = unit_points.to(device=bounds.device, dtype=bounds.dtype)
        points = bounds[0] + (bounds[1] - bounds[0]) * unit_points
        values = torch.stack([self._evaluate(objective, point) for point in points])
        return points, values

    def optimize(
        self,
        objective: Objective,
        bounds: torch.Tensor,
        budget: int,
        *,
        num_initial: int = 5,
        initial_x: torch.Tensor | None = None,
        initial_y: torch.Tensor | None = None,
        callback: Callable[[IterationRecord], None] | None = None,
        reset_population: bool = True,
    ) -> OptimizationResult:
        """Run the complete sequential CAKE + BAKER procedure."""
        bounds = torch.as_tensor(
            bounds, dtype=torch.double, device=self.cake.device
        )
        self._validate_bounds(bounds)
        if budget < 1:
            raise ValueError("budget must be positive")
        if num_initial < 2:
            raise ValueError("num_initial must be at least 2")
        if initial_y is not None and initial_x is None:
            raise ValueError("initial_y cannot be supplied without initial_x")

        if initial_x is None:
            if num_initial > budget:
                raise ValueError("num_initial cannot exceed budget")
            train_x, train_y = self._initial_design(
                objective, bounds, num_initial
            )
        else:
            train_x = torch.as_tensor(
                initial_x, dtype=bounds.dtype, device=bounds.device
            )
            if train_x.ndim != 2 or train_x.shape[1] != bounds.shape[1]:
                raise ValueError("initial_x must have shape (n, bounds.shape[1])")
            if train_x.shape[0] < 2:
                raise ValueError("At least two initial observations are required")
            if train_x.shape[0] > budget:
                raise ValueError("The initial design cannot exceed budget")
            if torch.any(train_x < bounds[0]) or torch.any(train_x > bounds[1]):
                raise ValueError("initial_x contains points outside bounds")
            if initial_y is None:
                train_y = torch.stack(
                    [self._evaluate(objective, point) for point in train_x]
                )
            else:
                train_y = torch.as_tensor(
                    initial_y, dtype=bounds.dtype, device=bounds.device
                ).reshape(-1)
                if train_y.shape[0] != train_x.shape[0]:
                    raise ValueError("initial_x and initial_y must have equal length")
                if not torch.isfinite(train_y).all():
                    raise ValueError("initial_y must be finite")

        if reset_population:
            self.cake.reset_population()
        history: list[IterationRecord] = []
        while train_x.shape[0] < budget:
            iteration = len(history) + 1
            self.cake.evolve(train_x, train_y)
            baker_result = self.baker.select(
                self.cake.population, train_y, bounds
            )
            candidate = baker_result.candidate.reshape(1, -1)
            observation = self._evaluate(objective, candidate[0])
            train_x = torch.cat((train_x, candidate), dim=0)
            train_y = torch.cat((train_y, observation.reshape(1)), dim=0)
            record = IterationRecord(
                iteration=iteration,
                candidate=candidate.detach().clone(),
                observation=float(observation.item()),
                kernel=baker_result.kernel,
                baker_score=baker_result.score,
                best_value=float(train_y.max().item()),
                baker_result=baker_result,
            )
            history.append(record)
            LOGGER.info(
                "Iteration %d/%d: y=%.6g, best=%.6g, kernel=%s",
                train_x.shape[0],
                budget,
                record.observation,
                record.best_value,
                record.kernel,
            )
            if callback is not None:
                callback(record)

        best_index = int(torch.argmax(train_y).item())
        return OptimizationResult(
            train_x=train_x,
            train_y=train_y,
            best_x=train_x[best_index].detach().clone(),
            best_y=float(train_y[best_index].item()),
            history=tuple(history),
        )


__all__ = [
    "CAKEOptimizer",
    "IterationRecord",
    "Objective",
    "OptimizationResult",
]
