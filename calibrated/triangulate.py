import numpy as np
from rigidbody import normalized, unpr, dots


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
    r, t = relative_pose.inverse().rt
    y0 = normalized(unpr(z0))
    y1 = normalized(unpr(z1))

    tlen = np.linalg.norm(t)
    tdir = normalized(t)

    lhs = np.array([
        y0 - tdir * np.dot(tdir, y0),
        normalized(np.cross(tdir, y0)),
        tdir
        ])

    a = np.dot(lhs, y0)
    b = np.dot(lhs, np.dot(r, y1))

    bhead = b[0] * b[0] + b[1] * b[1]

    c = a[0] * a[0] - bhead
    d = np.sqrt(c * c + 4 * a[0] * a[0] * b[0] * b[0])

    e = 2 * b[0] * b[2] * a[0] + a[2] * (a[0] * a[0] - bhead - d)

    if abs(b[1]) > 1e-8:
        f = (a[0] * a[0] - bhead - d) * b[1] / (e * b[1])
    else:
        f = -b[0] / (a[0] * b[2] - a[2] * b[0])

    rhs = np.array([
        f * (a[0] / (2. * d) * (a[0] * a[0] + b[0] * b[0] - b[1] * b[1] + d)),
        f * (a[0] / (2. * d) * (2. * b[0] * b[1])),
        f * a[2]
        ])

    return np.linalg.solve(lhs, rhs) * tlen
