import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import squareform, pdist
from scipy import linalg
from src.visualization import plot_smds_edge_vectors


def izma(M, B, X0, tol=1e-6, max_iter=100, lambda_val=None):
    X = X0.copy()
    for _ in range(max_iter):
        # Step 3: Replace known entries according to mask B
        X = X * (1 - B) + M * B

        # Step 4: Best approximation under nuclear norm constraint
        U, s, Vt = linalg.svd(X, full_matrices=False)
        if lambda_val is not None:
            # Soft-threshold singular values, scaled to avoid zeroing out
            s = np.maximum(s - lambda_val / np.max(s), 0)
        X_new = np.dot(U * s, Vt)

        # Check convergence
        error = linalg.norm((X_new * B - M * B), ord='fro')
        X = X_new
        if error < tol:
            break

    return X


def low_rank_completion(M, mask, tol=1e-6, lambda_tol=1e-6, max_iter=100):
    # Replace NaN with 0 for computation
    M = np.nan_to_num(M, nan=0.0)
    # Ensure mask is binary (0 or 1)
    mask = mask.astype(float)
    # Apply mask
    M = M * mask

    # Initialize lambda bounds
    lambda_min = 0
    lambda_max = linalg.norm(M, ord='nuc')  # Nuclear norm
    X = M.copy()  # Initial guess
    lambda_prev = None

    for _ in range(max_iter):
        lambda_val = (lambda_min + lambda_max) / 2

        # Step 7: Call IZMA to approximate M ⊙ B with nuclear norm constraint
        X = izma(M, mask, X, tol, max_iter, lambda_val)

        # Step 8: Compute error
        error = linalg.norm((X * mask - M * mask), ord='fro')

        # Update lambda bounds
        if error > tol:
            lambda_min = lambda_val
        else:
            lambda_max = lambda_val

        # Check convergence
        if lambda_prev is not None and abs(lambda_val - lambda_prev) < lambda_tol:
            break
        lambda_prev = lambda_val

    return X


def get_smds_pairwise_estimates(env, noise_std=0.0, phase_noise_std=0.0, N_A=4) -> tuple[np.ndarray, list]:
    N = len(env.uavs)
    if N < 2 or N_A > N:
        return np.array([]), []
    #rss_matrix = env.get_rss_isotropic_matrix(noise_std=noise_std)
    #distance_matrix = env.estimate_distances_from_rss(rss_matrix)
    distance_matrix = env.get_noisy_distance_matrix(noise_std)

    # print("Incomplete distance matrix:\n", distance_matrix_incomplete)
    # print("Mask (True for observed, False for missing):\n", mask)



    edge_vectors = []
    edge_pairs = []
    # AN-AN pairs
    for i in range(N_A):
        for j in range(i + 1, N_A):
            if env.connection_matrix[i, j] and distance_matrix[i, j] > 0:
                d = distance_matrix[i, j]
                #az, el = env.uavs[i].measure_aoa(env.uavs[j], phase_noise_std)
                az, el = env.uavs[i].calculate_ground_truth_angles(env.uavs[j])
                unit = np.array([
                    np.cos(np.radians(el)) * np.cos(np.radians(az)),
                    np.cos(np.radians(el)) * np.sin(np.radians(az)),
                    np.sin(np.radians(el))
                ])
                v_i = d * unit
                edge_vectors.append(v_i)
                edge_pairs.append((i, j))
            else:
                return np.array([]), []
    # AN-TN pairs
    for i in range(N_A):
        for j in range(N_A, N):
            if env.connection_matrix[i, j] and distance_matrix[i, j] > 0:
                d = distance_matrix[i, j]
                az, el = env.uavs[i].measure_aoa(env.uavs[j], phase_noise_std)
                unit = np.array([
                    np.cos(np.radians(el)) * np.cos(np.radians(az)),
                    np.cos(np.radians(el)) * np.sin(np.radians(az)),
                    np.sin(np.radians(el))
                ])
                v_i = d * unit
                edge_vectors.append(v_i)
                edge_pairs.append((i, j))
            else:
                return np.array([]), []
    return np.array(edge_vectors), edge_pairs

def construct_gram_edge_kernel(edge_vectors: np.ndarray, edge_pairs: list, env, N_A: int) -> tuple[np.ndarray, int, list]:
    N = len(env.uavs)
    N_T = N - N_A
    M_expected = int(N_A * (N_A - 1) / 2 + N_A * N_T)  # AN-AN + AN-TN edges
    # Filter edge_vectors and edge_pairs to AN-AN and AN-TN pairs
    valid_edges = []
    valid_pairs = []
    for i, (n, m) in enumerate(edge_pairs):
        if (n < N_A and m < N_A) or (n < N_A and m >= N_A) or (m < N_A and n >= N_A):
            if env.connection_matrix[n, m]:  # Ensure connectivity
                valid_edges.append(edge_vectors[i])
                valid_pairs.append((n, m))
    M = len(valid_edges)
    if M != M_expected:
        print(f"Shape mismatch: Expected M={M_expected}, got {M} valid edges")
        return -1, -1, []
    if not valid_edges or len(valid_edges[0]) != 3:
        print("Invalid edge vectors")
        return -1, -1, []
    edge_vectors_filtered = np.array(valid_edges)
    # Extract distances from filtered edge vectors
    distances = np.linalg.norm(edge_vectors_filtered, axis=1)
    diag_d = np.diag(distances)
    # Compute ADoA matrix (cos_alpha) for valid edges
    cos_alpha = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i <= j:  # Upper triangle including diagonal
                vi = edge_vectors_filtered[i]
                vj = edge_vectors_filtered[j]
                cos_alpha[i, j] = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
            cos_alpha[j, i] = cos_alpha[i, j]  # Symmetric
    # Construct K_r (Eq. 22)
    K_r = diag_d @ cos_alpha @ diag_d
    return K_r, M, valid_pairs


def estimate_edge_matrix_from_gek(K_r: np.ndarray, N: int, N_A: int, M: int = None) -> np.ndarray:
    N_T = N - N_A
    M_expected = int(N_A * ((N_A - 1) / 2) + N_A * N_T)  # Correct M per paper
    if M is None:
        M = M_expected
    if K_r.shape != (M, M):
        print(f"Shape mismatch: Expected ({M}, {M}), got {K_r.shape}")
        return -1
    # Step 4: Perform SVD of K̃_r
    try:
        U, s, Vt = np.linalg.svd(K_r, full_matrices=False)
        #print(f"Debug: SVD singular values = {s[:3]}")  # Top 3 for inspection
    except np.linalg.LinAlgError:
        print("SVD computation failed")
        return -1
    # Step 5: Obtain V̂ using Eq. (23), limited to M x 3
    eta = 3  # Dimensionality for 3D
    if len(s) < eta:
        print(f"Warning: Only {len(s)} singular values, expected at least {eta}")
        return -1
    # Select top 3 singular values and vectors
    U_3 = U[:, :eta]  # M x 3
    Lambda_half = np.diag(np.sqrt(s[:eta]))  # 3 x 3
    V_hat = U_3 @ Lambda_half  # M x 3
    #print(f"Debug: V_hat shape = {V_hat.shape}")
    return V_hat

def compute_coordinate_estimates_from_vhat(env, V_hat: np.ndarray, valid_pairs: list, N_A: int) -> np.ndarray:
    N = len(env.uavs)
    eta = V_hat.shape[1]
    M = V_hat.shape[0]
    N_T = N - N_A
    expected_M = int(N_A * (N_A - 1) / 2 + N_A * N_T)
    if M != expected_M:
        print(f"Warning: Mismatch in edge count, expected {expected_M}, got {M}")
        return -1
    XA = np.array([env.uavs[i].position for i in range(N_A)], dtype=float)  # N_A x eta
    #print(f"Debug: XA = \n{XA}")
    # Construct C matrix for AN-AN and AN-TN edges
    C = np.zeros((M, N))
    for k, (n, m) in enumerate(valid_pairs):  # n < m
        if not (isinstance(n, (int, np.integer)) and isinstance(m, (int, np.integer))):
            print(f"Error: Invalid indices in valid_pairs at k={k}: n={n}, m={m}")
            return -1
        C[k, n] = -1
        C[k, m] = 1
    # Solve V_hat ≈ C X using least squares (pseudoinverse)
    try:
        X_hat_rel = np.linalg.lstsq(C, V_hat, rcond=None)[0]  # N x eta
        #print(f"Debug: X_hat_rel shape = {X_hat_rel.shape}")
        #print(f"Debug: X_hat_rel = \n{X_hat_rel}")
    except np.linalg.LinAlgError as e:
        print(f"Coordinate estimation error (singular matrix): {e}")
        return -1
    except Exception as e:
        print(f"Coordinate estimation error: {e}")
        return -1
    # Apply Procrustes transformation to align using anchors (Step 7)
    anchors_idx = list(range(N_A))  # Ensure list of indices
    #print(f"Debug: Type of anchors_idx = {type(anchors_idx)}, Value = {anchors_idx}")
    X_est_anchors = X_hat_rel[anchors_idx, :]
    cen_est = np.mean(X_est_anchors, axis=0)
    cen_known = np.mean(XA, axis=0)
    Xe_c = X_est_anchors - cen_est
    Xk_c = XA - cen_known
    H = Xe_c.T @ Xk_c
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    trace_xe = np.trace(Xe_c.T @ Xe_c)
    scale = np.sum(S) / trace_xe if trace_xe != 0 else 1.0
    X_hat = scale * (X_hat_rel - cen_est) @ R + cen_known
    return X_hat




def run_smds(env, noise_std=0.0, phase_noise_std=0.0, N_A=4, missing_percentage=0.0) -> np.ndarray:
    edge_vectors, edge_pairs = get_smds_pairwise_estimates(env, noise_std=noise_std, phase_noise_std=phase_noise_std, N_A=N_A)
    #plot_smds_edge_vectors(env, edge_vectors=edge_vectors, edge_pairs=edge_pairs)
    #print(f"Debug: edge_vectors = {edge_vectors}, edge_pairs = {edge_pairs}")
    K_r, M, valid_pairs = construct_gram_edge_kernel(edge_vectors, edge_pairs, env, N_A=N_A)

    #print("Original K_r:\n", K_r)

    # Create incomplete matrix
    mask = np.random.random(K_r.shape) > missing_percentage  # missing values
    K_incomplete = K_r.copy()
    K_incomplete[~mask] = np.nan
    #print("Incomplete K_r:\n", K_incomplete)

    #print(mask)

    # Complete using low_rank_completion
    K_r = low_rank_completion(K_incomplete, mask)

    #print("Completed K_r:\n", K_r)
    #print(f"Debug: K_r = {K_r.shape}")
    if isinstance(K_r, (int, np.integer)) and K_r == -1:
        print("[Case] Geometry failure in K_r construction")
        return -1
    V_hat = estimate_edge_matrix_from_gek(K_r, len(env.uavs), N_A=N_A, M=M)
    #print(f"Debug: V_hat shape = {V_hat.shape}")
    if isinstance(V_hat, (int, np.integer)) and V_hat == -1:
        print("[Case] V_hat estimation failure")
        return -1
    X_hat = compute_coordinate_estimates_from_vhat(env, V_hat, valid_pairs, N_A=N_A)
    #print("Second few rows of X_hat:\n", X_hat[:3])
    if isinstance(X_hat, (int, np.integer)) and X_hat == -1:
        print("[Case] Coordinate estimation failure")
        return -1
    true_coords = env.get_uav_position_matrix_all()
    if true_coords.shape[1] != 3 or X_hat.shape[1] != 3:
        print(f"Warning: Expected 3D coordinates, got true_coords {true_coords.shape}, X_hat {X_hat.shape}")
        return -1
    disparity = np.linalg.norm(true_coords - X_hat)
    #print(f"Debug: Disparity with true = {disparity}")
    #print("[Case] Successful run")
    return X_hat