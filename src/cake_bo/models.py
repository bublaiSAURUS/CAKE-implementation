"""Gaussian-process models used by CAKE."""

from __future__ import annotations

import math

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import (
    Kernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
from gpytorch.mlls import ExactMarginalLogLikelihood

from .kernels import KernelNode, parse_expression


def _base_kernel(name: str, dimensions: int) -> Kernel:
    factories = {
        "SE": lambda: RBFKernel(ard_num_dims=dimensions),
        "PER": lambda: PeriodicKernel(ard_num_dims=dimensions),
        "LIN": lambda: LinearKernel(ard_num_dims=dimensions),
        "RQ": lambda: RQKernel(ard_num_dims=dimensions),
        "M3": lambda: MaternKernel(nu=1.5, ard_num_dims=dimensions),
        "M5": lambda: MaternKernel(nu=2.5, ard_num_dims=dimensions),
    }
    return factories[name]()


def _build_kernel(node: KernelNode, dimensions: int) -> Kernel:
    if node.is_leaf:
        assert node.name is not None
        return _base_kernel(node.name, dimensions)
    assert node.left is not None and node.right is not None
    left = _build_kernel(node.left, dimensions)
    right = _build_kernel(node.right, dimensions)
    if node.operator == "+":
        return left + right
    if node.operator == "*":
        return left * right
    raise ValueError(f"Unsupported operator: {node.operator}")


def parse_kernel(expression: str, dimensions: int) -> ScaleKernel:
    """Convert a validated CAKE expression into a GPyTorch kernel."""
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    return ScaleKernel(_build_kernel(parse_expression(expression), dimensions))


def fit_gp_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel: str = "SE",
    device: str | torch.device = "cpu",
) -> tuple[SingleTaskGP, object, float]:
    """Fit a GP candidate and return its model, likelihood, and BIC."""
    target_device = torch.device(device)
    train_x = train_x.to(target_device)
    train_y = train_y.to(target_device)
    covariance = parse_kernel(kernel, train_x.shape[-1]).to(target_device)
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y.unsqueeze(-1),
        covar_module=covariance,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=train_x.shape[-1]),
    )
    likelihood = model.likelihood
    likelihood.noise = 1e-4
    marginal_log_likelihood = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(marginal_log_likelihood)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        output = model(train_x)
        log_likelihood = marginal_log_likelihood(output, train_y).item()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    bic = -2.0 * log_likelihood + parameter_count * math.log(train_x.shape[0])
    return model, likelihood, float(bic)


__all__ = ["fit_gp_model", "parse_kernel"]
