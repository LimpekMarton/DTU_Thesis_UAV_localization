#from scipy.spatial import distance_matrix
import numpy as np
from src.localization import coordinate_transformation
from scipy.linalg import eigh
from scipy.optimize import least_squares


def complete_edm(distance_matrix_incomplete, mask, d=3, max_iter=1000, tol=1e-6):
    # Square the input distances
    D = np.nan_to_num(distance_matrix_incomplete ** 2, nan=0.0)
    observed_mask = mask & ~np.isnan(distance_matrix_incomplete)
    # Initialize with mean of observed squared distances
    observed_entries = distance_matrix_incomplete[observed_mask] ** 2
    mu = np.mean(observed_entries) if observed_entries.size > 0 else 0.0
    n = D.shape[0]
    D = np.full((n, n), mu)
    D[observed_mask] = distance_matrix_incomplete[observed_mask] ** 2
    np.fill_diagonal(D, 0)

    for _ in range(max_iter):
        D_old = D.copy()
        # Eigenvalue decomposition
        lambdas, U = np.linalg.eigh(D)
        idx = lambdas.argsort()[::-1]
        lambdas = lambdas[idx]
        U = U[:, idx]
        # Threshold to rank d+2 (per Algorithm 2)
        lambdas[d + 2:] = 0
        D = U @ np.diag(lambdas) @ U.T
        # Enforce known entries
        D[observed_mask] = distance_matrix_incomplete[observed_mask] ** 2
        # Zero diagonal and negative entries
        np.fill_diagonal(D, 0)
        D = np.maximum(D, 0)
        # Convergence check
        if np.linalg.norm(D - D_old) < tol * np.linalg.norm(D_old):
            break

    # Return square root of completed matrix
    return np.sqrt(np.abs(D))




def get_relative_locations_mds(distance_matrix) -> np.ndarray:
    """Return the estimated relative positions of the UAVs in the environment"""
    D = distance_matrix
    N = D.shape[0]
    I = np.identity(N)
    C = I - (1 / N) * np.ones((N, N))

    # Calculate A
    A = -0.5 * C @ (D ** 2) @ C

    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Handle negative eigenvalues - keep only non-negative
    non_negative_mask = eigenvalues >= -1e-10  # Allow small numerical errors
    eigenvalues = eigenvalues[non_negative_mask]
    eigenvectors = eigenvectors[:, non_negative_mask]

    # Set small negative values to zero
    eigenvalues[eigenvalues < 0] = 0

    # Determine actual dimensionality
    n_dims = min(len(eigenvalues), 3)

    if n_dims < 3:
        print(f'Warning: Only {n_dims} non-negative eigenvalues found.')
        return -1

    # Calculate coordinates - now safe to take sqrt
    sqrt_eigenvalues = np.sqrt(eigenvalues[:n_dims])
    coordinates = eigenvectors[:, :n_dims] * sqrt_eigenvalues

    # Pad with zeros if less than 3D
    if n_dims < 3:
        padding = np.zeros((coordinates.shape[0], 3 - n_dims))
        coordinates = np.hstack([coordinates, padding])

    return coordinates


def run_mds(env, noise_std=0.0,):
    """This function execute the MDS in a given Environmet"""
    #distance_matrix = env.estimate_distances_from_rss(env.get_rss_isotropic_matrix(noise_std=noise_std))
    distance_matrix = env.get_noisy_distance_matrix(noise_std)

    n = distance_matrix.shape[0]
    mask = (distance_matrix != 0) | (np.eye(n, dtype=bool))  # True for non-zero or diagonal
    distance_matrix_incomplete = distance_matrix.copy()
    distance_matrix_incomplete[~mask] = np.nan  # Set non-diagonal zeros to np.nan
    #print("Incomplete distance matrix:\n", distance_matrix_incomplete)
    #print("Mask (True for observed, False for missing):\n", mask)

    # Complete the matrix (using existing complete_edm function)
    completed_distance_matrix = complete_edm(distance_matrix_incomplete, d=3, mask=mask)


    # --- Compute relative coordinates ---
    #relative_coordinates = get_relative_locations_mds(completed_D)

    relative_coordinates = get_relative_locations_mds(distance_matrix=completed_distance_matrix)

    #print(relative_coordinates)

    X_mds_global = coordinate_transformation.calculate_global_from_relative(env, relative_coordinates)

    return X_mds_global


