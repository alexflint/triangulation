import itertools

import unittest
import numpy as np

from rigidbody import pr, rotation, SE3, assert_arrays_almost_equal

from .triangulate import triangulate_midpoint, triangulate_linear, triangulate_directional_relative_pair


def generate_features(num_frames, noise=0):
    ps = np.random.randn(num_frames, 3)
    rs = map(rotation.exp, np.random.randn(num_frames, 3)*.1)
    poses = [SE3(r, p) for r, p in zip(rs, ps)]
    point = np.random.randn(3) + [0, 0, 10]

    features = []
    for pose in poses:
        z = pr(np.dot(pose.orientation, point - pose.position))
        if noise > 0:
            z += np.random.randn(2) * noise
        features.append(z)

    return features, poses, point


def check_triangulation(f, num_frames, noise, decimals):
    features, poses, true_position = generate_features(num_frames, noise)
    estimated = f(features, poses)
    assert_arrays_almost_equal(estimated, true_position, decimals)


def test_triangulate_midpoint():
    np.random.seed(0)
    for num_frames in [2, 3, 10]:
        for noise, decimals in [(0, 12), (1e-8, 6), (1e-3, 2)]:
            yield check_triangulation, triangulate_midpoint, num_frames, noise, decimals


def test_triangulate_linear():
    np.random.seed(0)
    for num_frames in [2, 3, 10]:
        print("Num frames=%d" % num_frames)
        for noise, decimals in [(0, 12)]:  #, (1e-8, 6), (1e-3, 2)]:
            yield check_triangulation, triangulate_linear, num_frames, noise, decimals


def test_triangulate_directional_pair():
    poses = [SE3.identity(), SE3.from_tangent(np.random.randn(6)*.1)]
    point = np.random.randn(3) + [0, 0, 10]
    features = [pr(np.dot(pose.orientation, point - pose.position)) for pose in poses]
    estimated = triangulate_directional_relative_pair(features[0], features[1], poses[1])
    assert_arrays_almost_equal(estimated, point)
