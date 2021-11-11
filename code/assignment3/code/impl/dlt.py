import numpy as np
from impl.util import MakeHomogeneous, HNormalize


def BuildProjectionConstraintMatrix(points2D, points3D):
    # TODO
    # For each correspondence, build the two rows of the constraint matrix and stack them

    num_corrs = points2D.shape[0]
    constraint_matrix = np.zeros((num_corrs * 2, 12))

    for i in range(num_corrs):
        xi, yi, wi = MakeHomogeneous(points2D[i])
        X_it = points3D[i]
        constraint_matrix[2 * i] = np.array([*[0, 0, 0, 0], *MakeHomogeneous(-wi * X_it), *MakeHomogeneous(yi * X_it)])
        constraint_matrix[2 * i + 1] = np.array([*MakeHomogeneous(wi * X_it), *[0, 0, 0, 0], *MakeHomogeneous(-xi * X_it)])

    return constraint_matrix
