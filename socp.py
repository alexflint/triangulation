import numpy as np
import cvxopt

class ConeConstraint(object):
    """
    Represents a cone constraint of the form || Ax + b || <= c * x + d
    """
    def __init__(self, a=None, b=None, c=None, d=None):
        if a is None:
            a = np.zeros(len(b), len(c))
        if b is None:
            b = np.zeros(len(a))
        if c is None:
            c = np.zeros(len(a[0]))
        if d is None:
            d = 0.
        assert np.ndim(a) == 2
        assert np.ndim(b) == 1
        assert np.ndim(c) == 1
        assert np.shape(a) == (len(b), len(c))
        self.a = np.asarray(a)
        self.b = np.asarray(b)
        self.c = np.asarray(c)
        self.d = float(d)

    def conditionalize(self, mask, values):
        """
        Given a constraint over x1...xn, return a new constraint over a subset of the variables given fixed values
        for the remaining variables.
        """
        assert len(values) == sum(mask)
        mask = np.asarray(mask)
        values = np.asarray(values)
        a = self.a[:, ~mask]
        b = self.b + np.dot(self.a[:, mask], values)
        c = self.c[~mask]
        d = self.d + float(np.dot(self.c[mask], values))
        return ConeConstraint(a, b, c, d)

    def conditionalize_at_zero(self, mask):
        """
        Given a constraint over x1...xn, return a new constraint over a subset of the variables given fixed values
        for the remaining variables.
        """
        mask = np.asarray(mask)
        return ConeConstraint(self.a[:, ~mask], self.b, self.c[~mask], self.d)

    def lhs(self, x):
        return np.linalg.norm(np.dot(self.a, x) + self.b)

    def rhs(self, x):
        return np.dot(self.c, x) + self.d

    def is_satisfied(self, x):
        return self.lhs(x) <= self.rhs(x)


class ConeProblem(object):
    """
    Represents a second order cone program
    """
    def __init__(self, objective, constraints=None):
        self.objective = np.asarray(objective)
        self.constraints = constraints or []

    def add_constraint(self, *args, **kwargs):
        self.constraints.append(ConeConstraint(*args, **kwargs))

    def conditionalize(self, mask, values=None):
        mask = np.asarray(mask)
        if values is None:
            return ConeProblem(self.objective[~mask], [x.conditionalize_at_zero(mask) for x in self.constraints])
        else:
            return ConeProblem(self.objective[~mask], [x.conditionalize(mask, values) for x in self.constraints])

    def conditionalize_indices(self, var_indices, values=None):
        if values is not None:
            assert len(var_indices) == len(values)
        mask = np.zeros(len(self.objective), bool)
        mask[np.array(var_indices)] = True
        return self.conditionalize(mask, values)

    def evaluate(self, x, verbose=False):
        print('Objective:', np.dot(self.objective, x))
        lhs = np.array([constraint.lhs(x) for constraint in self.constraints])
        rhs = np.array([constraint.rhs(x) for constraint in self.constraints])
        num_violated = np.sum(lhs > rhs)

        if verbose or num_violated > 0:
            for i, (lhs, rhs) in enumerate(zip(lhs, rhs)):
                label = 'satisfied' if (lhs <= rhs) else 'not satisfied'
                print('  Constraint %d: %s (lhs=%.8f, rhs=%.8f)' % (i, label, lhs, rhs))

        if num_violated == 0:
            print('  All constraints satisfied')
        else:
            print('  Not satisfied (%d constraints violated)' % num_violated)


def solve(problem, sparse=False, **kwargs):
    """
    Solve a second order cone program
    """
    gs = []
    hs = []
    for constraint in problem.constraints:
        a = constraint.a
        b = constraint.b
        c = constraint.c
        d = constraint.d
        g = np.vstack((-c, -a))
        hs.append(cvxopt.matrix(np.hstack((d, b))))
        if sparse:
            gs.append(cvxopt.sparse(cvxopt.matrix(g)))
        else:
            gs.append(cvxopt.matrix(g))
    cvxopt.solvers.options.update(kwargs)
    return cvxopt.solvers.socp(cvxopt.matrix(problem.objective), Gq=gs, hq=hs)
