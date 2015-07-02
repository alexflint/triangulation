import collections
import itertools

import numpy as np
import rigidbody


"""
Represents an observation together with the frame and landmark that it is associated with
"""
Observation = collections.namedtuple('Observation', ['frame_index', 'point_index', 'feature'])


"""
Represents a camera with a position, orientation, and intrinsics matrix
"""
Camera = collections.namedtuple('Camera', ['intrinsics', 'pose'])


"""
Represents a set of cameras, 3D points, and feature tracks
"""
Bundle = collections.namedtuple('Bundle', ['cameras', 'points', 'tracks'])


def factorize_pose_matrix(pose):
    assert pose.shape == (3, 4)
    qq, rr = np.linalg.qr(pose[:,:3].T)
    k = rr.T
    r = qq.T
    p = -np.dot(r.T, np.linalg.solve(k, pose[:, 3]))
    return k, r, p


def load_vgg_cameras(basename):
    cameras = []
    for index in itertools.count():
        path = '%s.%03d.P' % (basename, index)
        try:
            pose = np.loadtxt(path)
        except IOError:
            return cameras
        k, r, p = factorize_pose_matrix(pose)
        cameras.append(Camera(intrinsics=k, pose=rigidbody.SE3(r, p)))


def load_vgg_points(basename):
    return np.loadtxt(basename + '.p3d')


def load_vgg_corners(basename):
    corners = []
    for index in itertools.count():
        path = '%s.%03d.corners' % (basename, index)
        try:
            corners.append(np.loadtxt(path))
        except IOError:
            return corners


def load_vgg_tracks(basename):
    corners = load_vgg_corners(basename)
    tracks = []
    with open(basename+'.nview-corners') as fd:
        for point_index, line in enumerate(fd):
            track = []
            tokens = line.split()
            assert len(tokens) == len(corners)
            for frame_index, corner_index in enumerate(tokens):
                if corner_index != '*':
                    track.append(Observation(frame_index=frame_index,
                                             point_index=point_index,
                                             feature=corners[frame_index][int(corner_index)]))
            tracks.append(track)
    return tracks


def load_vgg_dataset(basename):
    """
    Load a VGG-format dataset in which:
    - cameras are in a file named BASENAME.NNN.P for N = 0,1,...
    - feature positions are a in a file named BASENAME.MMM.corners for M = 0,1,...
    - tracks are represented by a list of camera/feature indices in BASENAME.nview-corners
    - 3D points are in a file named BASENAME.p3d
    """

    return Bundle(cameras=load_vgg_cameras(basename),
                  points=load_vgg_points(basename),
                  tracks=load_vgg_tracks(basename))


def load_matlab_dataset(basename):
    """
    Load a matlab-format dataset in which:
    - poses are in a file named BASENAME_Ps.mat
    - tracked features are in a file named BASENAME_tracks.xy
    - there are no 3D points
    """
    import scipy.io
    poses = scipy.io.loadmat(basename+"_Ps.mat")['P'][0]
    features = np.loadtxt(basename+"_tracks.xy")

    cameras = []
    for pose in poses:
        k, r, p = factorize_pose_matrix(pose)
        cameras.append(Camera(intrinsics=k, pose=rigidbody.SE3(r, p)))

    tracks = []
    for i, row in enumerate(features):
        track = []
        for j, feature in enumerate(row.reshape((-1, 2))):
            if not np.all(feature == (-1, -1)):
                track.append(Observation(frame_index=j, point_index=i, feature=feature))
        tracks.append(track)

    return Bundle(cameras=cameras, points=None, tracks=tracks)
