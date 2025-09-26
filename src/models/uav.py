import numpy as np
from scipy.constants import c
from scipy.stats import vonmises

class Antenna:
    def __init__(self,
                 gain_dBi: float = 0.0,
                 position: np.ndarray = np.zeros(3),
                 orientation: np.ndarray = np.zeros(3)):
        self.gain_dBi = gain_dBi
        self.local_position = np.array(position)  # In UAV body frame
        self.orientation = np.array(orientation)  # (azimuth, elevation, roll)

    def get_radiation_pattern(self, theta, phi):
        """Simplified radiation pattern (TODO: override for directional antennas)"""
        return self.gain_dBi  # Default: isotropic

class UAV:
    def __init__(self, id: int, position: np.ndarray, frequency = 5e9, tx_power: float = 20):
        self.id = id
        self.position = position.astype(float)  # [x, y, z] as float64
        self.tx_power = tx_power  # dBm
        self.frequency = frequency
        # Tetrahedral antenna array (minimal 3D AoA)
        self.wavelength = c/self.frequency  # (λ/2 spacing)
        self.antennas = [
            Antenna(position=np.array([0, 0, 0])),  # Center
            Antenna(position=np.array([self.wavelength / 2, 0, 0])),  # X-axis
            Antenna(position=np.array([0, self.wavelength / 2, 0])),  # Y-axis
            Antenna(position=np.array([0, 0, self.wavelength / 2]))  # Z-axis
        ]

    def measure_rss_isotropic(self, other_uav, freq=5e9):
        """Measures RSS at `self` (receiver) from `other_uav` (transmitter)."""
        d = np.linalg.norm(self.position - other_uav.position)
        PL = 20 * np.log10(d) + 20 * np.log10(freq) - 147.55
        return other_uav.tx_power - PL

    def calculate_ground_truth_angles(self, other_uav) -> tuple[float, float]:
        # Vector from uav1 to other_uav
        delta = other_uav.position - self.position
        dist = np.linalg.norm(delta)

        # Azimuth (0° to 360°)
        azimuth = np.degrees(np.arctan2(delta[1], delta[0])) % 360

        # Elevation (-90° to 90°)
        elevation = np.degrees(np.arcsin(delta[2] / dist))


        return azimuth, elevation


    def measure_aoa(self, other_uav, angle_noise_std=0.0) -> tuple[float, float]:
        # 1. Verify antenna array configuration
        if len(self.antennas) < 4:
            raise ValueError("Need at least 4 antennas for 3D AoA")

        # Antenna positions in wavelengths (must be non-coplanar)
        ant_positions = np.array([ant.local_position / self.wavelength for ant in self.antennas])

        # 2. Calculate phase differences
        phases = []
        for ant in self.antennas:
            ant_pos_global = self.position + ant.local_position
            dist = np.linalg.norm(other_uav.position - ant_pos_global)
            phase = (dist / self.wavelength) * 2 * np.pi
            phases.append(phase)

        # 3. Create measurement matrix (A) and phase vector (b)
        A = ant_positions[1:] - ant_positions[0]  # Baselines relative to first antenna
        b = np.array(phases[1:]) - phases[0]  # Phase differences

        # 4. Solve for wave vector (k) using least squares
        k, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # 5. Convert to direction vector
        with np.errstate(invalid='ignore'):
            direction = k / (2 * np.pi)
            norm = np.linalg.norm(direction)
            if norm == 0:
                raise ValueError("Invalid direction vector: norm is zero")
            direction /= norm

            # Align with ground truth direction
            gt_az, gt_el = self.calculate_ground_truth_angles(other_uav)
            gt_direction = np.array([
                np.cos(np.radians(gt_el)) * np.cos(np.radians(gt_az)),
                np.cos(np.radians(gt_el)) * np.sin(np.radians(gt_az)),
                np.sin(np.radians(gt_el))
            ])
            if np.dot(direction, gt_direction) < 0:
                direction = -direction

        # 6. Convert to angles
        if abs(direction[2]) > 0.99999:
            azimuth = 0.0
            elevation = np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
        else:
            azimuth = np.degrees(np.arctan2(direction[1], direction[0])) % 360
            elevation = np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0)))



        # Add Gaussian-distributed angle errors
        sigma_rad = np.radians(angle_noise_std)  # Convert degrees to radians
        azimuth_error = np.random.normal(0, sigma_rad)  # In radians
        elevation_error = np.random.normal(0, sigma_rad)  # In radians
        # Apply errors and ensure azimuth stays in [0, 360°]
        azimuth = (azimuth + np.degrees(azimuth_error)) % 360
        elevation = np.clip(elevation + np.degrees(elevation_error), -90, 90)

        return azimuth, elevation