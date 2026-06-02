import torch
import botorch
from cake import CAKE

device = torch.device("cpu")

TEST_FUNCTION = {
    "ackley2": botorch.test_functions.Ackley(dim=2, negate=True),
    "griewank2": botorch.test_functions.Griewank(dim=2, negate=True),
    "levy2": botorch.test_functions.Levy(dim=2, negate=True),
    "rosenbrock": botorch.test_functions.Rosenbrock(dim=2, negate=True),
    "rastringin2": botorch.test_functions.Rastrigin(dim=2, negate=True),
}

T = 10

def define_objective_fn(func_name):
    fn = TEST_FUNCTION[func_name]
    bounds = fn.bounds.to(device)
    obj = lambda x: -fn.evaluate_true(x)
    ground_truth_y = fn.optimal_value
    return obj, bounds, ground_truth_y


if __name__ == "__main__":
    cake = CAKE(n_p = 4, p_m = 0.7, model_name = "gpt-4o-mini")
    obj, bounds, _ = define_objective_fn("ackley2")
    train_x = torch.rand(10, 2) * 2 - 1
    train_y = obj(train_x)
    print("Start")
    k_star = cake.run(train_x, train_y)
    print(f"k* = {k_star}")
    # x_t = cake.get_next_query(bounds)