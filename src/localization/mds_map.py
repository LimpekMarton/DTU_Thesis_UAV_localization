import numpy as np
from src.localization import multidimensional_scaling


def get_relative_locations_mds_map(distance_matrix: np.ndarray, env) -> np.ndarray:
    n = distance_matrix.shape[0]

    # Check if fully connected (all off-diagonal entries are 1)
    is_fully_connected = np.all(env.connection_matrix[np.logical_not(np.eye(n, dtype=bool))])

    if is_fully_connected:
        mds_input = np.copy(distance_matrix)
    else:
        mds_input = np.copy(distance_matrix)
        mds_input[~env.connection_matrix & (mds_input == 0)] = np.inf
        np.fill_diagonal(mds_input, 0)

        # Floyd-Warshall for missing distances
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if mds_input[i, j] > mds_input[i, k] + mds_input[k, j] and mds_input[i, k] != np.inf and mds_input[
                        k, j] != np.inf:
                        mds_input[i, j] = mds_input[i, k] + mds_input[k, j]
                        mds_input[j, i] = mds_input[i, j]

        mds_input[np.isinf(mds_input)] = 0

    # Get relative coordinates using provided MDS
    relative_coordinates = multidimensional_scaling.get_relative_locations_mds(mds_input)

    return relative_coordinates


def run_mds_map(env, noise_std=0.0) -> np.ndarray:
    #distance_matrix = env.estimate_distances_from_rss(env.get_rss_isotropic_matrix(noise_std=noise_std))
    distance_matrix = env.get_noisy_distance_matrix(noise_std)
    relative_coordinates = get_relative_locations_mds_map(distance_matrix, env)

    X_mds_map_global = multidimensional_scaling.coordinate_transformation.calculate_global_from_relative(env, relative_coordinates)

    return X_mds_map_global