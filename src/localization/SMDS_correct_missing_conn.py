import numpy as np
from scipy import linalg

def izma(M, B, X0, tol=1e-6, max_iter=100, lambda_val=None):
    X = X0.copy()
    for _ in range(max_iter):
        X = X * (1 - B) + M * B
        U, s, Vt = linalg.svd(X, full_matrices=False)
        if lambda_val is not None:
            s = np.maximum(s - lambda_val / np.max(s), 0)
        X_new = np.dot(U * s, Vt)
        error = linalg.norm((X_new * B - M * B), ord='fro')
        X = X_new
        if error < tol:
            break
    return X

def low_rank_completion(M, mask, tol=1e-6, lambda_tol=1e-6, max_iter=100):
    M = np.nan_to_num(M, nan=0.0)
    mask = mask.astype(float)
    M = M * mask
    lambda_min = 0
    lambda_max = linalg.norm(M, ord='nuc')
    X = M.copy()
    lambda_prev = None
    for _ in range(max_iter):
        lambda_val = (lambda_min + lambda_max) / 2
        X = izma(M, mask, X, tol, max_iter, lambda_val)
        error = linalg.norm((X * mask - M * mask), ord='fro')
        if error > tol:
            lambda_min = lambda_val
        else:
            lambda_max = lambda_val
        if lambda_prev is not None and abs(lambda_val - lambda_prev) < lambda_tol:
            break
        lambda_prev = lambda_val
    return X

def get_smds_pairwise_estimates(env, noise_std=0.0, phase_noise_std=0.0, N_A=4) -> tuple[np.ndarray, list]:
    N = len(env.uavs)
    if N < 2 or N_A > N:
        return np.array([]), []
    distance_matrix = env.get_noisy_distance_matrix(noise_std)
    edge_vectors = []
    edge_pairs = []
    for i in range(N_A):
        for j in range(i + 1, N_A):
            if env.connection_matrix[i, j] and distance_matrix[i, j] > 0:
                d = distance_matrix[i, j]
                az, el = env.uavs[i].calculate_ground_truth_angles(env.uavs[j])
                unit = np.array([
                    np.cos(np.radians(el)) * np.cos(np.radians(az)),
                    np.cos(np.radians(el)) * np.sin(np.radians(az)),
                    np.sin(np.radians(el))
                ])
                v_i = d * unit
                edge_vectors.append(v_i)
                edge_pairs.append((i, j))
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
    return np.array(edge_vectors), edge_pairs

def construct_gram_edge_kernel(edge_vectors: np.ndarray, edge_pairs: list, env, N_A: int) -> tuple[np.ndarray, int, list, list, dict]:
    N = len(env.uavs)
    N_T = N - N_A
    possible_an_an = [(i, j) for i in range(N_A) for j in range(i + 1, N_A)]
    possible_an_tn = [(i, j) for i in range(N_A) for j in range(N_A, N)]
    possible_pairs = possible_an_an + possible_an_tn
    M_max = len(possible_pairs)
    pair_to_idx = {pair: k for k, pair in enumerate(possible_pairs)}
    observed_pairs = [pair for pair in edge_pairs if env.connection_matrix[pair[0], pair[1]]]
    M_obs = len(observed_pairs)
    if M_obs == 0:
        return -1, -1, [], possible_pairs, {}
    edge_vectors_filtered = edge_vectors[:M_obs]
    K_obs = np.dot(edge_vectors_filtered, edge_vectors_filtered.T)
    K_incomplete = np.full((M_max, M_max), np.nan)
    obs_idx = [pair_to_idx[pair] for pair in observed_pairs]
    for ii, i in enumerate(obs_idx):
        for jj, j in enumerate(obs_idx):
            K_incomplete[i, j] = K_obs[ii, jj]
    return K_incomplete, M_max, observed_pairs, possible_pairs, pair_to_idx

def estimate_edge_matrix_from_gek(K_r: np.ndarray, N: int, N_A: int, M: int = None) -> np.ndarray:
    N_T = N - N_A
    M_expected = int(N_A * (N_A - 1) / 2 + N_A * N_T)
    if M is None:
        M = M_expected
    if K_r.shape != (M, M):
        return -1
    try:
        U, s, Vt = np.linalg.svd(K_r, full_matrices=False)
    except np.linalg.LinAlgError:
        return -1
    eta = 3
    if len(s) < eta:
        return -1
    U_3 = U[:, :eta]
    Lambda_half = np.diag(np.sqrt(s[:eta]))
    V_hat = U_3 @ Lambda_half
    return V_hat

def compute_coordinate_estimates_from_vhat(env, V_hat: np.ndarray, possible_pairs: list, N_A: int) -> np.ndarray:
    N = len(env.uavs)
    eta = V_hat.shape[1]
    M = V_hat.shape[0]
    N_T = N - N_A
    expected_M = int(N_A * (N_A - 1) / 2 + N_A * N_T)
    if M != expected_M:
        return -1
    XA = np.array([env.uavs[i].position for i in range(N_A)], dtype=float)
    C = np.zeros((M, N))
    for k, (n, m) in enumerate(possible_pairs):
        C[k, n] = -1
        C[k, m] = 1
    try:
        X_hat_rel = np.linalg.lstsq(C, V_hat, rcond=None)[0]
    except np.linalg.LinAlgError:
        return -1
    anchors_idx = list(range(N_A))
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

def run_smds(env, noise_std=0.0, phase_noise_std=0.0, N_A=4) -> np.ndarray:
    edge_vectors, edge_pairs = get_smds_pairwise_estimates(env, noise_std=noise_std, phase_noise_std=phase_noise_std, N_A=N_A)
    K_incomplete, M_max, observed_pairs, possible_pairs, pair_to_idx = construct_gram_edge_kernel(edge_vectors, edge_pairs, env, N_A=N_A)
    if isinstance(K_incomplete, int) and K_incomplete == -1:
        return -1
    mask = ~np.isnan(K_incomplete)
    K_r = low_rank_completion(K_incomplete, mask)
    V_hat = estimate_edge_matrix_from_gek(K_r, len(env.uavs), N_A=N_A, M=M_max)
    if isinstance(V_hat, int) and V_hat == -1:
        return -1
    X_hat = compute_coordinate_estimates_from_vhat(env, V_hat, possible_pairs, N_A=N_A)
    if isinstance(X_hat, int) and X_hat == -1:
        return -1
    true_coords = env.get_uav_position_matrix_all()
    if true_coords.shape[1] != 3 or X_hat.shape[1] != 3:
        return -1
    return X_hat