import numpy as np


def get_smds_pairwise_estimates(env, noise_std=0.0, phase_noise_std=0.0) -> tuple[np.ndarray, list]:
    N = len(env.uavs)
    if N < 2:
        return np.array([]), []

    rss_matrix = env.get_rss_isotropic_matrix(noise_std=noise_std)
    distance_matrix = env.estimate_distances_from_rss(rss_matrix)

    edge_vectors = []
    edge_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
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

def construct_reduced_edge_gram_kernel(edge_vectors: np.ndarray, edge_pairs: list, N: int) -> tuple[np.ndarray, list]:
    if N < 2 or len(edge_vectors) < N - 1 or len(edge_pairs) < N - 1:
        return -1, None

    # Select N-1 edge pairs, preferring UAV 0
    selected_pairs = []
    selected_indices = []
    for i, (n, m) in enumerate(edge_pairs):
        if n == 0 and m <= N - 1 and len(selected_pairs) < N - 1:
            selected_pairs.append((n, m))
            selected_indices.append(i)
        if len(selected_pairs) == N - 1:
            break

    # Fallback to diverse pairs if insufficient or dependent
    if len(selected_pairs) < N - 1:
        print("[Geometry Warning] UAV 0 edges insufficient, trying diverse edges")
        used_nodes = set()
        selected_pairs = []
        selected_indices = []
        for i, (n, m) in enumerate(edge_pairs):
            if n not in used_nodes and m not in used_nodes and len(selected_pairs) < N - 1:
                selected_pairs.append((n, m))
                selected_indices.append(i)
                used_nodes.add(n)
                used_nodes.add(m)
            if len(selected_pairs) == N - 1:
                break
    if len(selected_pairs) < N - 1:
        print("[Geometry Warning] Insufficient edges, using first available")
        selected_pairs = edge_pairs[:N - 1]
        selected_indices = list(range(N - 1))

    # Compute K_bar directly from inner products
    K_bar = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        for j in range(N - 1):
            vi = edge_vectors[selected_indices[i]]
            vj = edge_vectors[selected_indices[j]]
            K_bar[i, j] = np.dot(vi, vj)

    # Verify rank (optional, for robustness)
    if np.linalg.matrix_rank(K_bar) < 3:
        print(f"[Geometry Warning] K_bar rank {np.linalg.matrix_rank(K_bar)} < 3, geometry may be degenerate")
        return -1, None

    return K_bar, selected_pairs



def compute_edge_gram_eigen_decomposition(K_bar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(K_bar, (int, np.integer)) and K_bar == -1:
        return -1, -1
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(K_bar)
        idx = np.argsort(eigenvalues)[::-1]
        sorted_eigs = eigenvalues[idx]
        eigenvalues = np.maximum(sorted_eigs, 0.0)
        eigenvectors = eigenvectors[:, idx]
        #print(f"Debug: Eigenvalues = {eigenvalues}, eigenvectors = {eigenvectors}")
        return eigenvalues, eigenvectors
    except:
        return -1, -1


def estimate_edge_vector_hat(eigenvalues: np.ndarray, eigenvectors: np.ndarray, eta: int = 3) -> np.ndarray:
    if isinstance(eigenvalues, (int, np.integer)) and eigenvalues == -1:
        return -1
    N_minus_1 = len(eigenvalues)
    if N_minus_1 < eta or eta > N_minus_1:
        return -1
    try:
        num_pos = min(eta, np.sum(eigenvalues > 1e-12))
        if num_pos < eta:
            print(f"Warning: only {num_pos} significant eigenvalues found (η={eta})")
            return -1
        top_k_eigenvalues = eigenvalues[:num_pos]
        top_k_eigenvectors = eigenvectors[:, :num_pos]  # (N-1) x eta
        top_k_eigenvalues = np.maximum(top_k_eigenvalues, 0)
        Lambda_sqrt = np.diag(np.sqrt(top_k_eigenvalues))  # eta x eta
        # Equation (17) adjusted: V_hat = top_k_eigenvectors @ Lambda_sqrt (no transpose for reduced case)
        V_hat = top_k_eigenvectors @ Lambda_sqrt  # (N-1) x eta
        if V_hat.shape != (N_minus_1, eta):
            print(f"Warning: V_hat shape {V_hat.shape} mismatch, expected ({N_minus_1}, {eta})")
            return -1
        #print(f"Debug: V_hat shape = {V_hat.shape}")
        return V_hat
    except:
        return -1






def retrieve_edge_vectors(env, V_hat: np.ndarray, selected_pairs: list, edge_vectors: np.ndarray,
                                     edge_pairs: list) -> tuple[np.ndarray, list, float]:
    if isinstance(V_hat, (int, np.integer)) and V_hat == -1:
        return -1, [], float('inf')

    N = len(env.uavs)
    eta = V_hat.shape[1]

    # Build V_ref (reference edge vectors)
    V_ref = np.zeros((N - 1, eta))
    for i, pair in enumerate(selected_pairs):
        if pair in edge_pairs:
            idx = edge_pairs.index(pair)
            V_ref[i] = edge_vectors[idx]
        elif (pair[1], pair[0]) in edge_pairs:
            idx = edge_pairs.index((pair[1], pair[0]))
            V_ref[i] = -edge_vectors[idx]  # Flip orientation
        else:
            print(f"Error: Pair {pair} not found")
            return -1, [], float('inf')

    # Try both V_hat and -V_hat (eigendecomposition sign ambiguity)
    best_error = float('inf')
    best_V_recovered = None

    for sign_flip in [1, -1]:
        try:
            V_hat_test = sign_flip * V_hat

            # Simple Procrustes: find R such that ||V_ref - V_hat_test @ R||_F is minimized
            U, s, Vt = np.linalg.svd(V_hat_test.T @ V_ref, full_matrices=False)
            R = U @ Vt

            # Ensure proper rotation (det = 1)
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            V_recovered_test = V_hat_test @ R
            error = np.linalg.norm(V_ref - V_recovered_test, 'fro') / np.linalg.norm(V_ref, 'fro')

            if error < best_error:
                best_error = error
                best_V_recovered = V_recovered_test

        except Exception:
            continue

    if best_V_recovered is None or best_error > 0.5:
        print(f"Procrustes failed, error: {best_error}")
        return -1, [], float('inf')

    return best_V_recovered, selected_pairs, best_error


def solve_coordinates_simple(env, V_recovered: np.ndarray, selected_pairs: list) -> np.ndarray:
    if isinstance(V_recovered, (int, np.integer)) and V_recovered == -1:
        return -1

    N, eta = len(env.uavs), V_recovered.shape[1]
    x1 = np.array(env.uavs[0].position, dtype=float)

    # Start with anchor UAV 0
    X = np.zeros((N, eta))
    X[0] = x1
    positioned = {0}  # Keep track of positioned UAVs

    # Position UAVs incrementally using edge constraints
    remaining_pairs = list(selected_pairs)

    while len(positioned) < N and remaining_pairs:
        progress_made = False

        for i, (n, m) in enumerate(remaining_pairs[:]):
            # If we know position of n, compute position of m
            if n in positioned and m not in positioned:
                X[m] = X[n] + V_recovered[selected_pairs.index((n, m))]
                positioned.add(m)
                remaining_pairs.remove((n, m))
                progress_made = True
                #print(f"Positioned UAV {m} using edge from UAV {n}")
            # If we know position of m, compute position of n
            elif m in positioned and n not in positioned:
                X[n] = X[m] - V_recovered[selected_pairs.index((n, m))]
                positioned.add(n)
                remaining_pairs.remove((n, m))
                progress_made = True
                #print(f"Positioned UAV {n} using edge from UAV {m}")

        if not progress_made:
            print("Warning: Could not position all UAVs incrementally")
            break

    # If incremental failed, fall back
    if len(positioned) < N:
        print("Falling back to full system solve")
        A = np.zeros((N, N))
        b = np.zeros((N, eta))

        A[0, 0] = 1.0
        b[0] = x1

        for i, (n, m) in enumerate(selected_pairs):
            row_idx = i + 1
            A[row_idx, n] = -1.0
            A[row_idx, m] = +1.0
            b[row_idx] = V_recovered[i]

        try:
            if np.linalg.cond(A) < 1e10:
                X = np.linalg.solve(A, b)
            else:
                X = np.linalg.lstsq(A, b, rcond=1e-10)[0]
            X[0] = x1  # Enforce anchor
        except:
            return -1

    return X



def run_smds(env, noise_std=0.0, phase_noise_std=0.0) -> np.ndarray:
    edge_vectors, edge_pairs = get_smds_pairwise_estimates(env, noise_std=noise_std, phase_noise_std=phase_noise_std)
    #plot_smds_edge_vectors(env, edge_vectors, edge_pairs)
    K_bar, selected_pairs = construct_reduced_edge_gram_kernel(edge_vectors, edge_pairs, len(env.uavs))



    if isinstance(K_bar, (int, np.integer)) and K_bar == -1:
        print("[Case] Geometry failure in K_bar construction")
        return -1

    eigenvalues, eigenvectors = compute_edge_gram_eigen_decomposition(K_bar)
    if isinstance(eigenvalues, (int, np.integer)) and eigenvalues == -1:
        print("[Case] Eigenvalue decomposition failure")
        return -1

    V_hat = estimate_edge_vector_hat(eigenvalues, eigenvectors, eta=3)
    if isinstance(V_hat, (int, np.integer)) and V_hat == -1:
        print("[Case] V_hat estimation failure")
        return -1

    V_recovered, selected_pairs, rel_err = retrieve_edge_vectors(env, V_hat, selected_pairs, edge_vectors,
                                                                            edge_pairs)
    if isinstance(V_recovered, (int, np.integer)) and V_recovered == -1:
        print("[Case] Procrustes alignment failure")
        return -1

    X_smds_global = solve_coordinates_simple(env, V_recovered, selected_pairs)
    if isinstance(X_smds_global, (int, np.integer)) and X_smds_global == -1:
        print("[Case] Coordinate estimation failure")
        return -1

    return X_smds_global