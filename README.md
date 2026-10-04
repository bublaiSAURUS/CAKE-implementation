# CAKE — Context-Aware Kernel Evolution

An independent implementation of **Context-Aware Kernel Evolution (CAKE)** and
**BIC-Acquisition Kernel Ranking (BAKER)** from Suwandi et al. (NeurIPS 2025).
The package provides the complete sequential Bayesian-optimization loop: evolve
Gaussian-process kernels with an LLM, rank their proposed queries with BAKER,
evaluate the selected point, and condition the next generation on the expanded
dataset.

> This is a research reproduction, not the authors' official implementation.

## Method

CAKE begins with six base kernels and evolves compositions made with addition
and multiplication:

| Symbol | Kernel |
| --- | --- |
| `SE` | Squared exponential (RBF) |
| `PER` | Periodic |
| `LIN` | Linear |
| `RQ` | Rational quadratic |
| `M3` | Matérn 3/2 |
| `M5` | Matérn 5/2 |

At each Bayesian-optimization iteration:

1. every population member is refitted to all observations and assigned a BIC;
2. an LLM proposes crossover candidates and, with probability `p_m`, a mutation;
3. the `n_p` lowest-BIC kernels survive;
4. expected improvement (EI) is independently optimized for each survivor;
5. BAKER computes the kernel weight
   `w_k = exp(-BIC_k) / sum(exp(-BIC))` and score
   `w_k * normalized_EI_k`;
6. the highest-scoring query is evaluated and appended to the observations.

EI values are divided by the largest candidate EI to place them in `[0, 1]`
without changing their ratios. If every EI is zero, BAKER falls back to BIC
weights. The implementation maximizes objectives; negate the returned value for
minimization problems.

## Installation

CAKE requires Python 3.10 or newer. Install it in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Copy the environment template and add an OpenAI API key:

```bash
cp .env.example .env             # Windows: copy .env.example .env
```

```dotenv
OPENAI_API_KEY=your-api-key
```

`.env` is loaded with `python-dotenv` and excluded from version control.

## Complete optimization loop

```python
import torch

from cake_bo import BAKER, CAKE, CAKEOptimizer


def objective(x: torch.Tensor) -> torch.Tensor:
    # Objectives receive a tensor with shape (1, d).
    return -((x - 0.25) ** 2).sum(dim=-1)


bounds = torch.tensor(
    [[-1.0, -1.0], [1.0, 1.0]],
    dtype=torch.double,
)

cake = CAKE(
    num_crossover=5,
    num_population=10,
    mutation_prob=0.7,
    model_name="gpt-4o-mini",
    random_seed=0,
)
optimizer = CAKEOptimizer(cake, BAKER(), random_seed=0)

result = optimizer.optimize(
    objective,
    bounds,
    budget=15,       # Total evaluations, including the initial design.
    num_initial=5,
)

print(result.best_x)
print(result.best_y)
print(result.history[-1].kernel)
```

The optimizer uses a scrambled Sobol initial design. Existing observations can
be supplied instead:

```python
result = optimizer.optimize(
    objective,
    bounds,
    budget=20,
    initial_x=observed_x,
    initial_y=observed_y,
)
```

`initial_y` may be omitted to evaluate `initial_x`. A callback can receive each
`IterationRecord`, and `reset_population=False` can continue evolving an
existing population.

## Kernel evolution only

The lower-level API runs one CAKE generation without selecting or evaluating a
new query:

```python
from cake_bo import CAKE

cake = CAKE()
best_expression = cake.evolve(train_x, train_y)
print(best_expression, cake.population[best_expression].bic)
```

## BAKER results

`BAKER.select(population, train_y, bounds)` returns a `BAKERResult` containing:

- `candidate`: selected query with shape `(1, d)`;
- `kernel`: selected kernel expression;
- `score`: winning BAKER score; and
- `rankings`: BIC weights, raw/normalized EI, scores, and candidate points for
  every successfully optimized kernel.

`num_restarts` and `raw_samples` may be passed to `BAKER` to control acquisition
optimization. Defaults are `20 * d` and `50 * d`, matching the reference code.

## LLM providers and testing

Inject any client with a `generate(message, system_prompt) -> str` method to use
a different provider or a deterministic test double:

```python
cake = CAKE(llm_client=my_client)
```

Responses must follow this format:

```text
Kernel: SE + PER
Analysis: A short explanation of the proposed structure.
```

Expressions are parsed locally before GP construction. Only the six documented
base kernels, `+`, `*`, and parentheses are accepted. Multiplication has higher
precedence than addition.

## Synthetic experiment

Run the complete loop on a BoTorch test function:

```bash
python examples/synthetic.py --function ackley2 --budget 15 --initial 5
```

Use `python examples/synthetic.py --help` for all settings. Each BO iteration
fits several GPs and makes up to `num_crossover + 1` LLM calls, so experiments
can take several minutes and consume API credits.

## Package layout

```text
src/cake_bo/
├── __init__.py       Public package API
├── baker.py          BIC-Acquisition Kernel Ranking
├── evolution.py      CAKE population evolution
├── kernels.py        Safe expression grammar and parser
├── models.py         GPyTorch construction, fitting, and BIC
├── optimizer.py      Complete sequential BO loop
└── llm/
    ├── __init__.py
    └── llm_client.py  LLM protocol and OpenAI adapter
examples/
└── synthetic.py      End-to-end benchmark example
tests/
└── test_kernel_expression.py
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

The parser tests do not access an LLM. Full-loop runs require the scientific
dependencies and either an API key or an injected client.

## Reproducibility and limitations

- Paper defaults are `n_c=5`, `p_m=0.7`, and `n_p=10`.
- Random seeds control Sobol initialization and evolutionary sampling. LLM
  responses and GP hyperparameter fitting may still be nondeterministic.
- A model is refitted whenever the observations change. Consequently, runtime
  grows with population size and evaluation budget.
- Acquisition failures for individual kernels are logged and excluded; BAKER
  raises an error if every kernel fails.
- The final population's fitted models precede the final objective observation;
  they are refitted if optimization continues.
- The implementation currently supports scalar, continuous, bounded,
  single-objective maximization with `q=1` expected improvement.
- This code is intended for research and education, not production decisions.
- No software license has been declared for this independent reproduction.

## Reference

```bibtex
@inproceedings{suwandi2025cake,
  title     = {Adaptive Kernel Design for Bayesian Optimization Is a Piece of
               CAKE with LLMs},
  author    = {Suwandi, Richard Cornelius and Yin, Feng and Wang, Juntao and
               Li, Renjie and Chang, Tsung-Hui and Theodoridis, Sergios},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

- [Paper (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/c03a2610bca2712b984b331fd4f7bb6f-Paper-Conference.pdf)
- [Authors' official implementation](https://github.com/richardcsuwandi/cake)
