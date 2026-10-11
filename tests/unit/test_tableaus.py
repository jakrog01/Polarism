"""Rooted-tree order conditions for the source-derived RK coefficients."""
from functools import lru_cache

import numpy as np
import pytest

from polarism.solver.tableaus import TABLEAUS


@lru_cache(None)
def trees_through(order):
    trees = {1: [()]}
    for size in range(2, order + 1):
        candidates = [tree for lower in range(1, size) for tree in trees[lower]]
        ranks = {tree: rank for rank, tree in enumerate(candidates)}

        def forests(remaining, start):
            if not remaining:
                yield ()
            for index in range(start, len(candidates)):
                child = candidates[index]
                child_size = tree_size(child)
                if child_size <= remaining:
                    for tail in forests(remaining - child_size, index):
                        yield (child,) + tail

        trees[size] = sorted(set(forests(size - 1, 0)),
                             key=lambda tree: tuple(ranks[child] for child in tree))
    return trees


@lru_cache(None)
def tree_size(tree):
    return 1 + sum(tree_size(child) for child in tree)


@lru_cache(None)
def gamma(tree):
    result = tree_size(tree)
    for child in tree:
        result *= gamma(child)
    return result


def residuals(tableau, weights, through):
    a = np.asarray(tableau.A, dtype=float)
    b = np.asarray(weights, dtype=float)
    @lru_cache(None)
    def phi(tree):
        out = np.ones(tableau.stages)
        for child in tree:
            out *= a @ phi(child)
        return out
    return [abs(b @ phi(tree) - 1 / gamma(tree))
            for size in range(1, through + 1) for tree in trees_through(through)[size]]


def exact_conditions(tableau, weights, through):
    @lru_cache(None)
    def phi(tree):
        if not tree:
            return (1,) * tableau.stages
        children = [phi(child) for child in tree]
        return tuple(
            np.prod([sum(tableau.A[i][j] * child[j]
                         for j in range(tableau.stages)) for child in children])
            for i in range(tableau.stages)
        )

    for size in range(1, through + 1):
        for tree in trees_through(through)[size]:
            assert sum(weight * value for weight, value in zip(weights, phi(tree))) * gamma(tree) == 1


@pytest.mark.parametrize("name", TABLEAUS)
def test_tableau_order_conditions(name):
    tableau = TABLEAUS[name]
    a = np.asarray(tableau.A, dtype=float)
    c = np.asarray(tableau.c, dtype=float)
    assert a.shape == (tableau.stages, tableau.stages)
    assert np.all(a == np.tril(a, -1))
    np.testing.assert_allclose(c, a.sum(axis=1), atol=1e-13, rtol=0)
    assert max(residuals(tableau, tableau.b, tableau.order)) < 1e-13
    assert max(residuals(tableau, tableau.b, tableau.order + 1)) > 1e-13
    if name in {"rk4", "dp5", "vern7"}:
        exact_conditions(tableau, tableau.b, tableau.order)
    if tableau.b_hat is not None:
        assert max(residuals(tableau, tableau.b_hat, tableau.embedded_order)) < 1e-13
        if name in {"dp5", "vern7"}:
            exact_conditions(tableau, tableau.b_hat, tableau.embedded_order)
    if tableau.b_hat3 is not None:
        assert max(residuals(tableau, tableau.b_hat3, 3)) < 1e-13
    if tableau.e5 is not None:
        assert len(tableau.e5) == tableau.stages + 1
        assert tableau.e5[-1] == 0
        assert all(x - y == error for x, y, error in
                   zip(tableau.b, tableau.b_hat, tableau.e5))
    if tableau.e3 is not None:
        assert len(tableau.e3) == tableau.stages + 1
        assert tableau.e3[-1] == 0
        assert all(x - y == error for x, y, error in
                   zip(tableau.b, tableau.b_hat3, tableau.e3))
