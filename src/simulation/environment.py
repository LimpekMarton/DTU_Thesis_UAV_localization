import numpy as np
from src.models.uav import UAV
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp

class Environment:
    def __init__(self, dimensions=np.array([100.0, 100.0, 100.0]), uav_number=5, frequency=5e9, positions=None):
        self.dimensions = dimensions.astype(float)
        self.frequency = frequency
        self.uavs = {}
        self.connection_matrix = None
        if positions is None:
            for uav in range(uav_number):
                random_position = np.round(np.random.uniform(low=[0.0, 0.0, 0.0], high=self.dimensions, size=(3,)), 3)
                new_uav = UAV(id=uav, position=random_position)
                self.add_uav(new_uav)
        else:
            for (p, i) in zip(positions, range(len(positions))):
                new_uav = UAV(id=i, position=p)
                self.add_uav(new_uav)

    def add_uav(self, uav):
        self.uavs[uav.id] = uav
        n = len(self.uavs)

        # Initialize or expand matrix
        if self.connection_matrix is None:
            self.connection_matrix = np.zeros((n, n), dtype=bool)
        else:
            new_matrix = np.zeros((n, n), dtype=bool)
            new_matrix[:n - 1, :n - 1] = self.connection_matrix  # Copy old values
            self.connection_matrix = new_matrix

    def connect_uavs(self, id1: int, id2: int):
        if id1 in self.uavs and id2 in self.uavs:
            self.connection_matrix[id1][id2] = True
            self.connection_matrix[id2][id1] = True

    def connect_all_uavs(self):
        uav_ids = list(self.uavs.keys())
        for i in range(len(uav_ids)):
            for j in range(i + 1, len(uav_ids)):  # Avoid self-connections and duplicates
                self.connect_uavs(uav_ids[i], uav_ids[j])


    def connect_uavs_randomly(self, avg_degree=4.0):
        n = len(self.uavs)
        N_A = 4
        if n < N_A + 1:
            raise ValueError("Need at least 5 UAVs")
        # Reset connections
        self.connection_matrix = np.zeros((n, n), dtype=bool)
        # Ensure anchors (0 to N_A-1) are fully connected (clique)
        for i in range(N_A):
            for j in range(i + 1, N_A):
                self.connection_matrix[i, j] = True
                self.connection_matrix[j, i] = True
        # Ensure each target connects to at least one anchor
        for t in range(N_A, n):
            anchor = np.random.choice(N_A)
            self.connection_matrix[anchor, t] = True
            self.connection_matrix[t, anchor] = True
        # Current edges from guarantees
        current_edges = N_A * (N_A - 1) // 2 + (n - N_A)
        # Target total edges for average degree
        target_edges = int(round(n * avg_degree / 2))
        # Adjust probability for remaining edges
        remaining_edges = target_edges - current_edges
        possible_edges = (n * (n - 1) // 2) - current_edges
        p = min(1.2 * remaining_edges / possible_edges, 1.0) if possible_edges > 0 else 0.0  # Overshoot factor
        # Add random edges
        for i in range(n):
            for j in range(i + 1, n):
                if not self.connection_matrix[i, j] and current_edges < target_edges:
                    if np.random.random() < p:
                        self.connection_matrix[i, j] = True
                        self.connection_matrix[j, i] = True
                        current_edges += 1
        # Ensure overall connectivity

        adj = sp.csr_matrix(self.connection_matrix)
        num_components = connected_components(adj)[0]
        max_attempts = 1000
        attempts = 0
        while num_components > 1 and attempts < max_attempts:
            _, labels = connected_components(adj, return_labels=True)
            comp0 = np.where(labels == 0)[0]
            comp1 = np.where(labels == 1)[0]
            i = np.random.choice(comp0)
            j = np.random.choice(comp1)
            self.connection_matrix[i, j] = True
            self.connection_matrix[j, i] = True
            current_edges += 1
            adj = sp.csr_matrix(self.connection_matrix)
            num_components = connected_components(adj)[0]
            attempts += 1
        if num_components > 1:
            print(f"Warning: Failed to connect graph after {max_attempts} attempts")
        # Verify average degree
        degrees = np.sum(self.connection_matrix, axis=1)
        actual_avg = np.mean(degrees)
        if abs(actual_avg - avg_degree) > 1.0:
            print(f"Warning: Achieved avg degree {actual_avg:.2f}, target was {avg_degree}")

    def distance_between_uavs(self, id1: int, id2: int) -> float:
        return np.linalg.norm(self.uavs[id1].position - self.uavs[id2].position)

    def get_distance_matrix(self) -> np.ndarray:
        n = len(self.uavs)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):  # Upper triangle only
                if self.connection_matrix[i][j]:
                    dist = np.linalg.norm(self.uavs[i].position - self.uavs[j].position)
                    distance_matrix[i][j] = dist
                    distance_matrix[j][i] = dist  # Symmetric

        return distance_matrix

    def get_noisy_distance_matrix(self, noise_std=0.1) -> np.ndarray:
        n = len(self.uavs)
        noisy_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):  # Upper triangle only
                if self.connection_matrix[i][j]:
                    # Calculate true distance
                    true_dist = np.linalg.norm(self.uavs[i].position - self.uavs[j].position)
                    # Add Gaussian noise (ensure positive)
                    noisy_dist = abs(true_dist + np.random.normal(0, noise_std))
                    noisy_matrix[i][j] = noisy_dist
                    noisy_matrix[j][i] = noisy_dist  # Maintain symmetry

        return noisy_matrix


    #def get_uav_position_matrix_all(self) -> np.ndarray:
    #    """Returns a matrix with UAVs position"""
    #    return np.array([uav.position for uav in self.uavs.values()])

    def get_uav_position_matrix_all(self) -> np.ndarray:
        sorted_uavs = sorted(self.uavs.items(), key=lambda x: x[0])  # Sort by UAV ID
        return np.array([uav.position for _, uav in sorted_uavs])

    def get_rss_isotropic_matrix(self, noise_std=0.0) -> np.ndarray:
        n = len(self.uavs)
        rss_matrix = np.full((n, n), -np.inf)  # Default to -inf (no connection)

        for i in range(n):
            for j in range(n):
                if self.connection_matrix[i, j]:  # Only measure connected pairs
                    rss_matrix[i, j] = np.round(self.uavs[i].measure_rss_isotropic(
                        other_uav=self.uavs[j],
                        freq=self.frequency
                    )+ np.random.normal(0, noise_std), 3)
        return rss_matrix

    def estimate_distances_from_rss(self, rss_matrix, tx_power=20.0):
        n = len(self.uavs)
        distance_matrix = np.full((n, n), np.inf)  # Default: infinite distance

        for i in range(n):
            for j in range(n):
                if self.connection_matrix[i, j] and not np.isneginf(rss_matrix[i, j]):
                    # Extract RSS and handle UAV-specific tx_power if available
                    rss = rss_matrix[i, j]
                    pt = getattr(self.uavs[j], 'tx_power', tx_power)

                    # Inverse path loss formula
                    path_loss = pt - rss
                    distance = 10 ** ((path_loss - 20 * np.log10(self.frequency) + 147.55) / 20)
                    distance_matrix[i, j] = distance

        distance_matrix[np.isinf(distance_matrix)] = 0  #put 0 where no connection (because of MDS)
        return distance_matrix
