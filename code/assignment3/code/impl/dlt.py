import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))
  zeros = np.array(4 * [0])
  for i in range(num_corrs):
    # TODO Add your code here
    xi, yi = points2D[i]
    X_t = np.append(points3D[i], 1)
    index = 2 * i
    constraint_matrix[index] = np.concatenate((zeros, -X_t, yi * X_t))
    constraint_matrix[index + 1] = np.concatenate((X_t, zeros, -xi * X_t))

  return constraint_matrix