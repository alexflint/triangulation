import numpy as np
from numericaltesting import assert_arrays_almost_equal

from .socp import ConeConstraint, ConeProblem, solve

def test_socp():
    constraints = [
        ConeConstraint(np.eye(2), np.zeros(2), np.zeros(2), 3.),
        ConeConstraint(np.eye(2), [2, 0], np.zeros(2), 3.)
    ]
    problem = ConeProblem([0., -1.], constraints)
    solution = solve(problem)
    assert_arrays_almost_equal(np.squeeze(solution['x']), [-1., 2.*np.sqrt(2)])
