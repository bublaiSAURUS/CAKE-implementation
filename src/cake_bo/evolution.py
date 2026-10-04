"""LLM-guided Context-Aware Kernel Evolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from .kernels import parse_expression
from .llm import LLMClient, OpenAIClient
from .models import fit_gp_model

LOGGER = logging.getLogger(__name__)
BASE_KERNELS = ("SE", "PER", "LIN", "RQ", "M3", "M5")
OPERATORS = ("+", "*")

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in Gaussian processes. Analyze the observations below and
propose kernel structures that capture patterns in the data.

Observations:
{observations}

Allowed base kernels: {base_kernels}
Allowed operators: {operators}

Only use the listed kernels, +, *, and parentheses. Lower BIC is better.
When asked for a candidate, respond in exactly this format:
Kernel: <kernel expression>
Analysis: <brief explanation>
"""

CROSSOVER_PROMPT_TEMPLATE = """\
Create one child from these parent kernels:
- {parent_kernel1} (BIC: {fitness1:.6g})
- {parent_kernel2} (BIC: {fitness2:.6g})

Combine or reuse their components with the allowed operators {operators}.
"""

MUTATION_PROMPT_TEMPLATE = """\
Mutate this kernel by replacing one base-kernel component:
- {kernel} (BIC: {fitness:.6g})

Choose replacements only from {base_kernels}.
"""


@dataclass
class KernelResult:
    """A fitted candidate in the kernel population."""

    bic: float
    model: object
    likelihood: object


class CAKE:
    """Evolve compositional GP kernels using an LLM as genetic operator.

    Defaults follow the paper: five crossovers, population size ten, and a
    mutation probability of 0.7.
    """

    def __init__(
        self,
        num_crossover: int = 5,
        num_population: int = 10,
        mutation_prob: float = 0.7,
        model_name: str = "gpt-4o-mini",
        device: str | torch.device = "cpu",
        llm_client: LLMClient | None = None,
        random_seed: int | None = None,
    ) -> None:
        if num_crossover < 0:
            raise ValueError("num_crossover must be non-negative")
        if num_population < 2:
            raise ValueError("num_population must be at least 2")
        if not 0.0 <= mutation_prob <= 1.0:
            raise ValueError("mutation_prob must be between 0 and 1")
        self.num_crossover = num_crossover
        self.num_population = num_population
        self.mutation_prob = mutation_prob
        self.base_kernels = list(BASE_KERNELS)
        self.operators = list(OPERATORS)
        self.device = torch.device(device)
        self.llm = llm_client or OpenAIClient(model_name=model_name)
        self.rng = np.random.default_rng(random_seed)
        self.train_x: torch.Tensor | None = None
        self.train_y: torch.Tensor | None = None
        self.system_prompt = ""
        self.reset_population()

    def reset_population(self) -> None:
        """Restore the initial population of six base kernels."""
        self.population: dict[str, KernelResult | None] = {
            kernel: None for kernel in self.base_kernels
        }
        self.population_prob = torch.full(
            (len(self.population),), 1.0 / len(self.population), dtype=torch.float64
        )

    @staticmethod
    def parse_response(response: str) -> tuple[str, str]:
        """Extract and validate a kernel expression from an LLM response."""
        if not response or not response.strip():
            raise ValueError("The LLM returned an empty response")
        fields: dict[str, str] = {}
        current: str | None = None
        for raw_line in response.strip().replace("```", "").splitlines():
            line = raw_line.strip()
            lowered = line.lower()
            if lowered.startswith("kernel:"):
                current = "kernel"
                fields[current] = line.split(":", 1)[1].strip().strip("`")
            elif lowered.startswith("analysis:"):
                current = "analysis"
                fields[current] = line.split(":", 1)[1].strip()
            elif current and line:
                fields[current] = f"{fields[current]} {line}".strip()
        expression = fields.get("kernel", "")
        if not expression:
            raise ValueError("The LLM response does not contain a 'Kernel:' field")
        return str(parse_expression(expression)), fields.get("analysis", "")

    def update_data(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Validate observations and refresh the LLM context."""
        if train_x.ndim != 2:
            raise ValueError("train_x must have shape (n, d)")
        if train_y.ndim == 2 and train_y.shape[-1] == 1:
            train_y = train_y.squeeze(-1)
        if train_y.ndim != 1:
            raise ValueError("train_y must have shape (n,) or (n, 1)")
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train_x and train_y must contain the same number of rows")
        if train_x.shape[0] < 2:
            raise ValueError("At least two observations are required")
        if not torch.isfinite(train_x).all() or not torch.isfinite(train_y).all():
            raise ValueError("Training observations must be finite")
        self.train_x = train_x.to(self.device)
        self.train_y = train_y.to(self.device)
        observations = "\n".join(
            f"x = {x}, y = {y}"
            for x, y in zip(self.train_x.tolist(), self.train_y.tolist(), strict=True)
        )
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            observations=observations,
            base_kernels=", ".join(self.base_kernels),
            operators=", ".join(self.operators),
        )

    def _require_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train_x is None or self.train_y is None:
            raise RuntimeError("Call update_data() or evolve() before fitting kernels")
        return self.train_x, self.train_y

    def _fit(self, expression: str) -> KernelResult:
        train_x, train_y = self._require_data()
        model, likelihood, bic = fit_gp_model(
            train_x, train_y, kernel=expression, device=self.device
        )
        return KernelResult(bic=bic, model=model, likelihood=likelihood)

    def _refresh_probabilities(self) -> None:
        results = list(self.population.values())
        if any(result is None for result in results):
            raise RuntimeError("All population members must be evaluated first")
        bics = torch.tensor(
            [result.bic for result in results if result is not None],
            dtype=torch.float64,
        )
        scale = bics.std(unbiased=False)
        standardized = (
            (bics - bics.mean()) / scale
            if scale > 0
            else torch.zeros_like(bics)
        )
        self.population_prob = torch.softmax(-standardized, dim=0)

    def evaluate_population(self) -> None:
        """Refit every current candidate against the latest observations."""
        for expression in list(self.population):
            self.population[expression] = self._fit(expression)
        self._refresh_probabilities()

    def get_best_kernel(self) -> str:
        """Return the expression with the lowest BIC."""
        evaluated = {
            expression: result
            for expression, result in self.population.items()
            if result is not None
        }
        if not evaluated:
            raise RuntimeError("The population has not been evaluated")
        return min(evaluated, key=lambda expression: evaluated[expression].bic)

    def _crossover(self) -> None:
        mating_pool = list(self.population)
        probabilities = self.population_prob.cpu().numpy()
        for _ in range(self.num_crossover):
            selected = self.rng.choice(
                mating_pool, size=2, replace=False, p=probabilities
            )
            parent1, parent2 = str(selected[0]), str(selected[1])
            first = self.population[parent1]
            second = self.population[parent2]
            assert first is not None and second is not None
            fallback = f"{parent1} {self.rng.choice(self.operators)} {parent2}"
            try:
                response = self.llm.generate(
                    CROSSOVER_PROMPT_TEMPLATE.format(
                        parent_kernel1=parent1,
                        parent_kernel2=parent2,
                        fitness1=first.bic,
                        fitness2=second.bic,
                        operators=", ".join(self.operators),
                    ),
                    self.system_prompt,
                )
                candidate, rationale = self.parse_response(response)
                if rationale:
                    LOGGER.info("Crossover rationale: %s", rationale)
            except Exception as error:
                # LLMClient is provider-agnostic, so adapters may expose
                # different exception hierarchies. The deterministic fallback
                # keeps this generation usable and the failure is logged.
                LOGGER.warning("LLM crossover failed; using %s (%s)", fallback, error)
                candidate = fallback
            try:
                self.population[candidate] = self._fit(candidate)
            except (RuntimeError, ValueError) as error:
                LOGGER.warning(
                    "Discarding crossover candidate %s: %s", candidate, error
                )

    def _mutate(self) -> None:
        if self.rng.random() >= self.mutation_prob:
            return
        best = self.get_best_kernel()
        result = self.population[best]
        assert result is not None
        try:
            response = self.llm.generate(
                MUTATION_PROMPT_TEMPLATE.format(
                    kernel=best,
                    fitness=result.bic,
                    base_kernels=", ".join(self.base_kernels),
                ),
                self.system_prompt,
            )
            candidate, rationale = self.parse_response(response)
        except Exception as error:
            LOGGER.warning("LLM mutation skipped: %s", error)
            return
        if rationale:
            LOGGER.info("Mutation rationale: %s", rationale)
        try:
            self.population[candidate] = self._fit(candidate)
        except (RuntimeError, ValueError) as error:
            LOGGER.warning("Mutation candidate %s was discarded: %s", candidate, error)

    def _select_survivors(self) -> None:
        evaluated = [item for item in self.population.items() if item[1] is not None]
        evaluated.sort(key=lambda item: item[1].bic)
        self.population = dict(evaluated[: self.num_population])
        self._refresh_probabilities()

    def evolve(self, train_x: torch.Tensor, train_y: torch.Tensor) -> str:
        """Run one CAKE generation and return its lowest-BIC kernel."""
        self.update_data(train_x, train_y)
        self.evaluate_population()
        self._crossover()
        self._mutate()
        self._select_survivors()
        return self.get_best_kernel()

    run = evolve


__all__ = ["BASE_KERNELS", "CAKE", "KernelResult", "OPERATORS"]
