"""Run the complete CAKE + BAKER loop on a BoTorch benchmark."""

from __future__ import annotations

import argparse
import logging

import botorch
import torch

from cake_bo import BAKER, CAKE, CAKEOptimizer

TEST_FUNCTIONS = {
    "ackley2": botorch.test_functions.Ackley(dim=2, negate=True),
    "griewank2": botorch.test_functions.Griewank(dim=2, negate=True),
    "levy2": botorch.test_functions.Levy(dim=2, negate=True),
    "rastrigin2": botorch.test_functions.Rastrigin(dim=2, negate=True),
    "rosenbrock2": botorch.test_functions.Rosenbrock(dim=2, negate=True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", choices=TEST_FUNCTIONS, default="ackley2")
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--initial", type=int, default=5)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--crossovers", type=int, default=5)
    parser.add_argument("--mutation-prob", type=float, default=0.7)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    benchmark = TEST_FUNCTIONS[args.function].to(device=args.device, dtype=torch.double)
    cake = CAKE(
        num_crossover=args.crossovers,
        num_population=args.population,
        mutation_prob=args.mutation_prob,
        model_name=args.model,
        device=args.device,
        random_seed=args.seed,
    )
    optimizer = CAKEOptimizer(cake, BAKER(), random_seed=args.seed)
    result = optimizer.optimize(
        benchmark,
        benchmark.bounds,
        budget=args.budget,
        num_initial=args.initial,
    )
    print(f"Best value: {result.best_y:.6f}")
    print(f"Best point: {result.best_x.tolist()}")


if __name__ == "__main__":
    main()
