import numpy as np
from collections import deque
from src.localization.multidimensional_scaling import get_relative_locations_mds
from src.localization.coordinate_transformation import calculate_global_from_relative

def get_local_neighborhood(env, node_id, rlm=2):
    visited = set()
    queue = deque([(node_id, 0)])
    while queue:
        u, hop = queue.popleft()
        if u in visited:
            continue
        visited.add(u)
        if hop < rlm:
            neighbors = np.where(env.connection_matrix[u])[0]
            for v in neighbors:
                queue.append((v, hop + 1))
    return list(visited)

def build_local_distance_matrix(env, local_ids, distance_matrix):
    m = len(local_ids)
    local_conn = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            u = local_ids[a]
            v = local_ids[b]
            if env.connection_matrix[u, v]:
                d = distance_matrix[u, v]
                local_conn[a, b] = d if d > 0 else 1.0
                local_conn[b, a] = local_conn[a, b]
    return local_conn

def compute_shortest_paths(local_conn):
    m = local_conn.shape[0]
    shortest = np.full((m, m), np.inf)
    shortest[local_conn > 0] = local_conn[local_conn > 0]
    np.fill_diagonal(shortest, 0)
    for k in range(m):
        for i in range(m):
            for j in range(m):
                if shortest[i, k] + shortest[k, j] < shortest[i, j]:
                    shortest[i, j] = shortest[i, k] + shortest[k, j]
    return shortest

def align_maps(P, Q, rigid=False):
    if rigid:
        # Kabsch algorithm (rigid: rotation, translation, optional reflection)
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        P_c = P - centroid_P
        Q_c = Q - centroid_Q
        H = P_c.T @ Q_c
        U, S, Vt = np.linalg.svd(H)
        # No flip
        R_no = Vt.T @ U.T
        det_no = np.linalg.det(R_no)
        t_no = centroid_Q - R_no @ centroid_P
        aligned_no = (R_no @ P.T).T + t_no
        rmsd_no = np.sqrt(np.mean(np.linalg.norm(aligned_no - Q, axis=1)**2))
        # With potential flip
        R_flip, t_flip = None, None
        rmsd_flip = np.inf
        if det_no < 0:
            Vt_copy = Vt.copy()
            Vt_copy[2, :] *= -1
            R_flip = Vt_copy.T @ U.T
            t_flip = centroid_Q - R_flip @ centroid_P
            aligned_flip = (R_flip @ P.T).T + t_flip
            rmsd_flip = np.sqrt(np.mean(np.linalg.norm(aligned_flip - Q, axis=1)**2))
        if rmsd_flip < rmsd_no:
            return R_flip, t_flip
        return R_no, t_no
    else:
        # General linear transformation (includes scaling, rotation, reflection, translation)
        n = P.shape[0]
        # Formulate as least squares: find A (3x3) and t (3x1) such that A*P_i + t ≈ Q_i
        # Stack P with ones for translation: [x, y, z, 1]
        P_hom = np.hstack([P, np.ones((n, 1))])
        # Solve for A,t in one step: [A | t] * [x, y, z, 1]^T = [x', y', z']^T
        A_t, _, _, _ = np.linalg.lstsq(P_hom, Q, rcond=None)
        A = A_t[:3, :]  # 3x3 linear transformation
        t = A_t[3, :]   # 3x1 translation
        return A, t

def get_relative_locations_mds_map_p(env, rlm: int = 2, rigid: bool = False, noise_std=0.0) -> np.ndarray:
    distance_matrix = env.get_noisy_distance_matrix(noise_std)
    #print(distance_matrix)
    n = len(env.uavs)
    local_maps = []
    for i in range(n):
        local_ids = get_local_neighborhood(env, i, rlm)
        local_conn = build_local_distance_matrix(env, local_ids, distance_matrix)
        shortest = compute_shortest_paths(local_conn)
        relative = get_relative_locations_mds(shortest)
        local_map = {local_ids[k]: relative[k] for k in range(len(local_ids))}
        local_maps.append(local_map)
    # Merging
    start = np.random.randint(0, n)
    core_map = local_maps[start].copy()
    covered = set(core_map.keys())
    while len(covered) < n:
        candidates = []
        for j in range(n):
            if j == start: continue
            local_keys = set(local_maps[j].keys())
            common_set = covered & local_keys
            new_set = local_keys - covered
            if len(common_set) >= 3 and len(new_set) > 0:  # Relaxed to 3 for better merging
                candidates.append((len(common_set), j))
        if not candidates:
            # Fallback: return partial map
            relative_coordinates = np.zeros((n, 3))
            for i in range(n):
                relative_coordinates[i] = core_map.get(i, np.zeros(3))
            return relative_coordinates
        candidates.sort(reverse=True)
        best_j = candidates[0][1]
        common = sorted(list(covered & set(local_maps[best_j].keys())))
        pos_local = np.array([local_maps[best_j][c] for c in common])
        pos_core = np.array([core_map[c] for c in common])
        A, t = align_maps(pos_local, pos_core, rigid=rigid)
        # Transform local map
        local_ids_j = list(local_maps[best_j].keys())
        pos_j = np.array([local_maps[best_j][id_] for id_ in local_ids_j])
        pos_transformed = (A @ pos_j.T).T + t
        # Add new nodes
        new_nodes = set(local_ids_j) - covered
        for k, id_ in enumerate(local_ids_j):
            if id_ in new_nodes:
                core_map[id_] = pos_transformed[k]
        covered.update(new_nodes)
    # Build relative coordinates matrix
    relative_coordinates = np.zeros((n, 3))
    for i in range(n):
        relative_coordinates[i] = core_map.get(i, np.zeros(3))  # Fallback if missing
    return relative_coordinates

def run_mds_map_p(env, noise_std=0.0, rlm: int = 2, rigid: bool = False) -> np.ndarray:
    #distance_matrix = env.estimate_distances_from_rss(env.get_rss_isotropic_matrix(noise_std=noise_std))
    #distance_matrix = env.get_noisy_distance_matrix(noise_std)
    relative_coordinates = get_relative_locations_mds_map_p(env, rlm, rigid=rigid, noise_std=noise_std)
    X_mds_map_p_global = calculate_global_from_relative(env, relative_coordinates)
    return X_mds_map_p_global