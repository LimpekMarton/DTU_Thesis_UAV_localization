import numpy as np
from scipy.spatial import procrustes



def calculate_global_from_relative(env, relative_coordinates):
    # Here with disparity it is checked how different are the data after
    # it is 'shifted' to each other
    mtx1, mtx2, disparity = procrustes(env.get_uav_position_matrix_all(), relative_coordinates)

    # print(f'MDS disparity: {disparity}')

    # Calculate the global coordinates from the estimated relative coordinates
    # Step 1: Compute scaling factor (if Procrustes used scaling=True)
    scaling_factor = np.std(env.get_uav_position_matrix_all(), axis=0) / np.std(mtx2, axis=0)

    # Step 2: Rescale and shift back to original global frame
    X_global = mtx2 * scaling_factor + np.mean(env.get_uav_position_matrix_all(), axis=0)

    return X_global