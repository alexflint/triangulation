# Triangulation

[ ![Codeship Status for alexflint/triangulation](https://codeship.com/projects/a56819e0-05cc-0133-6af3-22315beb1c0d/status?branch=master)](https://codeship.com/projects/89535)

High quality implementations of triangulation algorithms in pure python.


### Installation

```bash
pip install triangulation
```

### Usage

```python
from rigidbody import pr, rotation, SE3
from triangulation import triangulate

noise = 1e-3
true_point = np.random.randn(3) + [0, 0, 10]

positions = np.random.randn(num_frames, 3)
orientations = map(rotation.exp, np.random.randn(num_frames, 3)*.1)
poses = [SE3(r, p) for r, p in zip(orientations, positions)]

features = []
for pose in poses:
    z = pr(np.dot(pose.orientation, point - pose.position))
    features.append(z + np.random.randn(2) * noise)

triangulated_point = triangulate(features, poses, algorithm="midpoint")
```

### Algorithms

**`triangulate_midpoint`**

Finds the 3D point minimizing the sum of squared distances to the rays from each camera.

**`triangulate_linear`**

Finds the 3D point minimizing the sum of squared alebraic errors.

**`triangulate_directional`**

Finds the 3D point minimizing the squared directional error in the first view and the view with longest baseline to the first view.

**`triangulate_directional_pair`**

Finds the 3D point minimizing the squared directional error in two views.

**`triangulate_infnorm`**

Finds the 3D point minimizing the infinity norm of the vector of reprojection errors in all views.

**`triangulate_infnorm_fixed`**

Finds any 3D point such that all reprojection errors are no greater than a given threshold.
