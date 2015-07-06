import numpy as np
from rigidbody import normalized, unpr, dots, sumsq


def householder(x):
    """Compute a 2x3 matrix where the rows are orthogonal to x and orthogonal to each other."""
    assert len(x) == 3, 'x=%s' % x
    assert np.linalg.norm(x) > 1e-8
    a = (np.arange(3) == np.argmin(np.abs(x))).astype(float)
    u = normalized(np.cross(x, a))
    v = normalized(np.cross(x, u))
    return np.array([u, v])


def triangulate_midpoint(features, poses):
    """
    Triangulate a landmark from two or more views using the midpoint method.
    """
    assert len(features) > 0
    assert len(features) == len(poses)
    jtj, jtr = np.zeros((3, 3)), np.zeros(3)
    for z, pose in zip(features, poses):
        h = householder(unpr(z))
        a = dots(h, pose.orientation)
        b = dots(h, pose.orientation, pose.position)
        jtj += np.dot(a.T, a)
        jtr += np.dot(a.T, b)
    return np.linalg.solve(jtj, jtr)


def triangulate_linear(features, poses):
    """
    Triangulate a landmark from two or more views using the midpoint method.
    """
    assert len(features) > 0
    assert len(features) == len(poses)
    jtj, jtr = np.zeros((3, 3)), np.zeros(3)
    for z, pose in zip(features, poses):
        a = pose.orientation[:2] - np.outer(z, pose.orientation[2])
        b = np.dot(a, pose.position)
        jtj += np.dot(a.T, a)
        jtr += np.dot(a.T, b)
    return np.linalg.solve(jtj, jtr)


def triangulate_directional_relative_pair(z0, z1, relative_pose):
    """
    Triangulate a landmark from a relative pose between two cameras.
    """
    r, t = relative_pose.inverse().rt
    y0 = normalized(unpr(z0))
    y1 = normalized(unpr(z1))

    tlen = np.linalg.norm(t)
    tdir = normalized(t)
    tperp = normalized(np.cross(tdir, y0))
    if np.linalg.norm(tperp) < 1e-8:
        raise Exception("observation is in direction of epipole")

    lhs = np.array([
        y0 - tdir * np.dot(tdir, y0),
        tperp,
        tdir
        ])

    a = np.dot(lhs, y0)
    b = np.dot(lhs, np.dot(r, y1))
    bhead = b[0] * b[0] + b[1] * b[1]

    c = a[0] * a[0] - bhead
    d = np.sqrt(c * c + 4 * a[0] * a[0] * b[0] * b[0])
    e = 2 * b[0] * b[2] * a[0] + a[2] * (a[0] * a[0] - bhead - d)

    if abs(b[1]) < 1e-8:
        f = -b[0] / (a[0] * b[2] - a[2] * b[0])
    else:
        f = (a[0] * a[0] - bhead - d) * b[1] / (e * b[1])

    rhs = np.array([
        f * (a[0] / (2. * d) * (a[0] * a[0] + b[0] * b[0] - b[1] * b[1] + d)),
        f * (a[0] / (2. * d) * (2. * b[0] * b[1])),
        f * a[2]
        ])

    return np.linalg.solve(lhs, rhs) * tlen


def triangulate_directional_pair(feature1, feature2, pose1, pose2):
    """
    Triangulate a landmark from two observations by minimizing the directional error.
    """
    xrel = triangulate_directional_relative_pair(feature1, feature2, pose2 * pose1.inverse())
    return pose1.inverse().transform(xrel)


def triangulate_directional(features, poses, base_index=0):
    """
    Triangulate a landmark by finding the pose that is furthest from base_index and minimizing the
    directional error using only those two views.
    """
    ps = np.array([pose.position for pose in poses])
    other_index = np.argmax(sumsq(ps - ps[base_index], axis=1))
    return triangulate_directional_pair(features[base_index], features[other_index], poses[base_index], poses[other_index])


def make_triangulation_problem(features, poses, max_error):
    """
    Construct a second order cone program for infinity-norm triangulation from the given features.
    """
    from .socp import ConeProblem
    problem = ConeProblem(np.array([0., 0., 1.]))
    for z, pose in zip(features, poses):
        r, p = pose.rp
        problem.add_constraint(
            a=r[:2] - np.outer(z, r[2]),
            b=np.dot(np.outer(z, r[2]) - r[:2], p),
            c=max_error*r[2],
            d=-max_error*np.dot(r[2], p))
    return problem


def triangulate_infnorm_fixed(features, poses, max_error):
    """
    Find a landmark that projects with no greater than MAX_ERROR reprojection error into
    any view if one exists, or return None if no such landmark exists.
    """
    from .socp import solve
    problem = make_triangulation_problem(features, poses, max_error)
    solution = solve(problem)
    if solution['x'] is None:
        return None
    else:
        return np.squeeze(solution['x'])


def triangulate_infnorm(features, poses, begin_radius=.01, min_radius=0., max_radius=1., abstol=1e-12, reltol=1e-12):
    """
    Triangulate a landmark by minimizing the maximum reprojection error in any view.
    """
    best = None
    lower, upper, radius = 0., None, .01
    while radius <= max_radius and (upper is None or (upper-lower > abstol and upper > lower*(1.+reltol))):
        x = triangulate_infnorm_fixed(features, poses, radius)
        if x is None:
            lower = radius
            if upper is None:
                raius *= 5.
            else:
                radius = (lower + upper) / 2.
        else:
            best = x
            upper = radius  # TODO: set upper to the error achieved by x
            radius = (lower + upper) / 2.

    if best is None:
        raise Exception('there was no feasible solution with error <= %f' % max_radius)

    return best


"""
Dictionary mapping algorithm names to triangulation functions
"""
algorithms = {
    "linear": triangulate_linear,
    "midpoint": triangulate_midpoint,
    "directional": triangulate_directional,
    "infnorm": triangulate_infnorm,
}


def triangulate(features, poses, algorithm, **kwargs):
    """
    Triangulate a landmark. Features should be a list of 2D points representing observations
    of the landmark in N views. Poses should be a list of 3x4 matrices or rigidbody.SE3 objects
    representing the pose of the corresponding cameras. Algorithm must be one of: "linear",
    "midpoint", "directional", "infnorm".
    """
    return algorithms[algorithm](features, poses, **kwargs)
