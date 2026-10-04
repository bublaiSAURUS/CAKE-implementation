import importlib.util
import unittest
from types import SimpleNamespace


SCIENTIFIC_STACK_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("torch", "botorch", "gpytorch")
)


@unittest.skipUnless(
    SCIENTIFIC_STACK_AVAILABLE, "scientific dependencies not installed"
)
class OptimizerLoopTests(unittest.TestCase):
    def test_budget_includes_initial_observations(self) -> None:
        import torch

        from cake_bo.optimizer import CAKEOptimizer

        class FakeCake:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self.population = {}
                self.evolve_calls = 0
                self.reset_calls = 0

            def reset_population(self) -> None:
                self.reset_calls += 1

            def evolve(self, train_x, train_y) -> str:
                self.evolve_calls += 1
                return "SE"

        class FakeBaker:
            def select(self, population, train_y, bounds):
                value = 0.25 * (train_y.shape[0] - 1)
                return SimpleNamespace(
                    candidate=torch.tensor([[value]], dtype=torch.double),
                    kernel="SE",
                    score=0.5,
                )

        cake = FakeCake()
        optimizer = CAKEOptimizer(cake, FakeBaker(), random_seed=0)
        result = optimizer.optimize(
            lambda x: -(x**2).sum(dim=-1),
            torch.tensor([[0.0], [1.0]], dtype=torch.double),
            budget=4,
            initial_x=torch.tensor([[0.0], [1.0]], dtype=torch.double),
            initial_y=torch.tensor([0.0, -1.0], dtype=torch.double),
        )

        self.assertEqual(result.train_x.shape[0], 4)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(cake.evolve_calls, 2)
        self.assertEqual(cake.reset_calls, 1)
        self.assertEqual(result.best_y, 0.0)


if __name__ == "__main__":
    unittest.main()
