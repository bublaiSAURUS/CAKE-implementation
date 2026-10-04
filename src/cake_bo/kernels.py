"""Safe parsing and construction of compositional GP kernels."""

from __future__ import annotations

import re
from dataclasses import dataclass

ALLOWED_KERNELS = frozenset({"SE", "PER", "LIN", "RQ", "M3", "M5"})
_TOKEN = re.compile(r"\s*(SE|PER|LIN|RQ|M3|M5|[()+*])")


class KernelExpressionError(ValueError):
    """Raised when an expression is outside the supported kernel grammar."""


@dataclass(frozen=True)
class KernelNode:
    """A node in a compositional-kernel expression tree."""

    name: str | None = None
    operator: str | None = None
    left: KernelNode | None = None
    right: KernelNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.name is not None

    def __str__(self) -> str:
        if self.is_leaf:
            return str(self.name)
        assert self.left is not None and self.right is not None and self.operator
        return f"({self.left} {self.operator} {self.right})"


def _tokenize(expression: str) -> list[str]:
    if not expression or not expression.strip():
        raise KernelExpressionError("Kernel expression must not be empty")
    tokens: list[str] = []
    position = 0
    while position < len(expression):
        match = _TOKEN.match(expression, position)
        if not match:
            fragment = expression[position : position + 12]
            raise KernelExpressionError(f"Unsupported token near {fragment!r}")
        tokens.append(match.group(1))
        position = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.position = 0

    def parse(self) -> KernelNode:
        node = self._sum()
        if self.position != len(self.tokens):
            raise KernelExpressionError(
                f"Unexpected token {self.tokens[self.position]!r}"
            )
        return node

    def _sum(self) -> KernelNode:
        node = self._product()
        while self._accept("+"):
            node = KernelNode(operator="+", left=node, right=self._product())
        return node

    def _product(self) -> KernelNode:
        node = self._factor()
        while self._accept("*"):
            node = KernelNode(operator="*", left=node, right=self._factor())
        return node

    def _factor(self) -> KernelNode:
        if self._accept("("):
            node = self._sum()
            if not self._accept(")"):
                raise KernelExpressionError("Unclosed parenthesis")
            return node
        if self.position >= len(self.tokens):
            raise KernelExpressionError("Expected a base kernel")
        token = self.tokens[self.position]
        if token not in ALLOWED_KERNELS:
            raise KernelExpressionError(f"Expected a base kernel, got {token!r}")
        self.position += 1
        return KernelNode(name=token)

    def _accept(self, token: str) -> bool:
        if self.position < len(self.tokens) and self.tokens[self.position] == token:
            self.position += 1
            return True
        return False


def parse_expression(expression: str) -> KernelNode:
    """Parse an expression using multiplication precedence over addition."""
    return _Parser(_tokenize(expression.strip())).parse()


__all__ = ["ALLOWED_KERNELS", "KernelExpressionError", "KernelNode", "parse_expression"]
