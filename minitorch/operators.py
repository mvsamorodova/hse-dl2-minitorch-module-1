"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Check whether x is less than y."""
    return 1 if x < y else 0


def eq(x: float, y: float) -> float:
    """Check whether two numbers are equal."""
    return 1 if x == y else 0


def max(x: float, y: float) -> float:
    """Return the larger number."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check whether two numbers are close."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))

    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)


def relu(x: float) -> float:
    """Apply the ReLU function."""
    return x if x > 0 else 0


def log(x: float) -> float:
    """Calculate the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the reciprocal."""
    return 1 / x


def log_back(x: float, d: float) -> float:
    """Calculate the logarithm derivative times d."""
    return d / x


def inv_back(x: float, d: float) -> float:
    """Calculate the reciprocal derivative times d."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Calculate the ReLU derivative times d."""
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to every element."""

    def _apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return _apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Apply a function to corresponding elements."""

    def _apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return _apply


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce an iterable to a single value."""

    def _apply(ls: Iterable[float]) -> float:
        res = start
        for x in ls:
            res = fn(res, x)
        return res

    return _apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate every element."""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements."""
    return zipWith(add)(ls1, ls2)


sum = reduce(add, 0.0)
prod = reduce(mul, 1.0)
