import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter
from scipy.spatial.transform import Rotation
from src.simulation import environment
from src.models import uav
import os
import glob



def load_uav_data(N=5):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, f'../../data/input/{N}UAV/uav*_data.csv')
        base_dir = os.path.dirname(path)
        if not os.path.exists(base_dir):
            print(f"Error: Directory '{base_dir}' does not exist")
            return None
        files = glob.glob(path)
        if not files:
            print(f"Error: No CSV files found at {path}")
            return None
        df_dict = {}
        for file in files:
            uav_name = os.path.basename(file).split('_')[0]
            try:
                df_dict[uav_name] = pd.read_csv(file, on_bad_lines='skip')
            except pd.errors.ParserError as e:
                print(f"Error parsing {file}: {str(e)}")
                return None
        return df_dict
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def compute_real_positions(all_data):
    real_pos = {}
    timesteps = min(len(df) for df in all_data.values())
    for uav_name in all_data:
        real_pos[uav_name] = pd.DataFrame({
            'real_x': all_data[uav_name]['Local_X'][:timesteps],
            'real_y': all_data[uav_name]['Local_Y'][:timesteps],
            'real_z': all_data[uav_name]['Local_Z'][:timesteps]
        })
    return real_pos

def analyze_performance(real_pos, results):
    performance = {}
    for uav_name in results:
        est_df = results[uav_name][['x', 'y', 'z']]
        real_df = real_pos[uav_name][['real_x', 'real_y', 'real_z']]
        real_df.columns = ['x', 'y', 'z']
        rmse = np.sqrt(((est_df - real_df) ** 2).mean())
        performance[uav_name] = rmse
        print(f'{uav_name} RMSE: x={rmse["x"]:.4f}, y={rmse["y"]:.4f}, z={rmse["z"]:.4f}')
    return performance


class JointUAV_EKF(ExtendedKalmanFilter):
    def __init__(self, dt=0.1, num_uavs=5, initial_positions=None, gravity=9.81,
                 imu_acc_noise_var=0.1, imu_quat_noise_var=0.01, gps_pos_var=0.01, range_var=3.0):
        dim_x = 6 * num_uavs
        dim_z = 3 * 4 + num_uavs * (num_uavs - 1) // 2
        super().__init__(dim_x=dim_x, dim_z=dim_z)

        self.dt = dt
        self.num_uavs = num_uavs
        self.gravity = gravity

        # State: [x,y,z,vx,vy,vz] for each UAV
        self.x = np.zeros(dim_x)
        if initial_positions is not None:
            for i in range(num_uavs):
                self.x[i*6:i*6+3] = np.array(initial_positions[i], dtype=float)

        # Covariances
        self.P = np.eye(dim_x) * 45.0

        # Process model: x_{k+1} = F x_k + B u_k + w
        # F and B are block-diagonal (per-UAV identical blocks)
        self.F = np.eye(dim_x)
        self.B = np.zeros((dim_x, 3 * num_uavs))
        for i in range(num_uavs):
            bi = np.zeros((6, 3))
            bi[0:3, 0:3] = 0.5 * (dt ** 2) * np.eye(3)  # position update from accel
            bi[3:6, 0:3] = dt * np.eye(3)               # velocity update from accel
            # place bi into big B
            self.B[i*6:(i+1)*6, i*3:(i+1)*3] = bi
            # F block
            self.F[i*6:i*6+3, i*6+3:i*6+6] = dt * np.eye(3)

        # Process noise from accel + quat uncertainties
        sigma_acc_total = np.sqrt(imu_acc_noise_var + imu_quat_noise_var)
        S = (sigma_acc_total ** 2) * np.eye(3 * num_uavs)  # block diag with identical 3x3s is equivalent here
        self.Q = self.B @ S @ self.B.T

        # Measurement noise (GPS anchors + ranges)
        R_gps = np.eye(3 * 4) * gps_pos_var
        R_rng = np.eye(num_uavs * (num_uavs - 1) // 2) * range_var
        self.R = np.block([
            [R_gps, np.zeros((R_gps.shape[0], R_rng.shape[0]))],
            [np.zeros((R_rng.shape[0], R_gps.shape[0])), R_rng]
        ])

        # store noise params if needed later
        self._sigma_acc_total = sigma_acc_total

    def _stack_control(self, imu_accs, imu_quats):
        u_list = []
        g = np.array([0.0, 0.0, self.gravity])
        for i in range(self.num_uavs):
            q = np.asarray(imu_quats[i], dtype=float)
            nq = np.linalg.norm(q)
            if nq < 1e-12:
                q = np.array([0, 0, 0, 1.0])  # fallback to identity
            else:
                q = q / nq
            rot = Rotation.from_quat(q)
            acc_body = np.asarray(imu_accs[i], dtype=float)
            acc_world = rot.apply(acc_body) - g
            u_list.append(acc_world)
        return np.concatenate(u_list, axis=0)

    def predict(self, imu_accs, imu_quats):
        # control
        u = self._stack_control(imu_accs, imu_quats)  # shape (3*N,)
        # state propagation
        self.x = self.F @ self.x + self.B @ u
        # covariance propagation
        self.P = self.F @ self.P @ self.F.T + self.Q
        # symmetrize + jitter
        self.P = (self.P + self.P.T) / 2.0 + 1e-9 * np.eye(self.dim_x)

    def update(self, meas):
        def Hx(x):
            out = []
            # GPS: anchors uav0..uav3 positions
            for i in range(4):
                pos_i = x[i*6:i*6+3]
                out.extend(pos_i)
            # ranges (all unordered pairs i<j)
            for i in range(self.num_uavs):
                for j in range(i + 1, self.num_uavs):
                    pi = x[i*6:i*6+3]
                    pj = x[j*6:j*6+3]
                    out.append(np.linalg.norm(pi - pj))
            return np.asarray(out, dtype=float)

        def HJacobian(x):
            H = np.zeros((self.dim_z, self.dim_x))
            r = 0
            # GPS Jacobian for anchors (w.r.t. positions only)
            for i in range(4):
                H[r:r+3, i*6:i*6+3] = np.eye(3)
                r += 3
            # Range Jacobians
            for i in range(self.num_uavs):
                for j in range(i + 1, self.num_uavs):
                    pi = x[i*6:i*6+3]
                    pj = x[j*6:j*6+3]
                    d = pi - pj
                    dist = np.linalg.norm(d)
                    if dist > 1e-9:
                        H[r, i*6:i*6+3] = d / dist
                        H[r, j*6:j*6+3] = -d / dist
                    # velocities don't affect instantaneous range
                    r += 1
            return H

        super().update(np.asarray(meas, dtype=float), HJacobian=HJacobian, Hx=Hx)
        self.P = (self.P + self.P.T) / 2.0 + 1e-9 * np.eye(self.dim_x)

    def get_states(self):
        states = {}
        for i in range(self.num_uavs):
            states[f'uav{i}'] = self.x[i*6:(i+1)*6].copy()
        return states


def run_joint_ekf(noise_std=0.0, number_of_uavs=5, delay=20):
    all_data = load_uav_data(N=number_of_uavs)
    if not all_data:
        return -1
    real_pos = compute_real_positions(all_data)

    env = environment.Environment(dimensions=np.array([150.0, 150.0, 100.0]), uav_number=0)
    initial_positions = []
    for uav_name in sorted(all_data.keys(), key=lambda x: int(x.replace('uav', ''))):
        uav_id = int(uav_name.replace('uav', ''))
        df = all_data[uav_name]
        raw_pos = np.array([df.iloc[0]['Local_X'], df.iloc[0]['Local_Y'], df.iloc[0]['Local_Z']])
        env.add_uav(uav.UAV(id=uav_id, position=raw_pos))
        initial_positions.append(raw_pos)
    env.connect_all_uavs()

    kf = JointUAV_EKF(dt=0.1, num_uavs=number_of_uavs, initial_positions=initial_positions)

    results = {uav_name: [] for uav_name in all_data}
    state_buffers = []
    meas_buffers = []
    meas_times = []
    imu_by_t = {uav_name: {} for uav_name in all_data}

    timesteps = min(len(df) for df in all_data.values())
    timesteps = min(timesteps, 10000)

    for t in range(timesteps):
        imu_accs, imu_quats = [], []
        # gather IMU and move env positions
        for uav_name in sorted(all_data.keys(), key=lambda x: int(x.replace('uav', ''))):
            uav_id = int(uav_name.replace('uav', ''))
            df = all_data[uav_name]
            raw_pos = np.array([df.iloc[t]['Local_X'], df.iloc[t]['Local_Y'], df.iloc[t]['Local_Z']])
            env.uavs[uav_id].position = raw_pos
            row = df.iloc[t]
            imu_acc = np.array([row['Linear_Acceleration_X'],
                                row['Linear_Acceleration_Y'],
                                row['Linear_Acceleration_Z']], dtype=float)
            imu_quat = np.array([row['Orientation_X'],
                                 row['Orientation_Y'],
                                 row['Orientation_Z'],
                                 row['Orientation_W']], dtype=float)
            imu_accs.append(imu_acc)
            imu_quats.append(imu_quat)
            imu_by_t[uav_name][t] = (imu_acc, imu_quat)

        # predict jointly
        kf.predict(imu_accs, imu_quats)

        # store for delayed correction
        state_buffers.append((kf.x.copy(), kf.P.copy(), t))
        if len(state_buffers) > delay + 1:
            state_buffers.pop(0)

        # build measurements every 'delay' steps (GPS for first 4 + ranges)
        if t % delay == 0:
            gps_positions = env.get_uav_position_matrix_all()[:4]  # (4,3)
            gps_meas = gps_positions.flatten()
            rng_mat = env.get_noisy_distance_matrix(noise_std=noise_std)
            rng_list = []
            for i in range(number_of_uavs):
                for j in range(i + 1, number_of_uavs):
                    rng_list.append(rng_mat[i, j])
            meas = np.concatenate([gps_meas, np.array(rng_list, dtype=float)])
            meas_buffers.append(meas)
            meas_times.append(t)
            if len(meas_buffers) > 2:
                meas_buffers.pop(0)
                meas_times.pop(0)

        # apply delayed update
        if meas_times and meas_times[0] == t - delay:
            meas = meas_buffers[0]
            delay_idx = None
            for i, (_, _, past_t) in enumerate(state_buffers):
                if past_t == t - delay:
                    delay_idx = i
                    break
            if delay_idx is not None:
                past_x, past_P, _ = state_buffers[delay_idx]
                kf.x = past_x.copy()
                kf.P = past_P.copy()
                kf.update(meas)

                # re-propagate to present using stored IMUs
                for step in range(t - delay + 1, t + 1):
                    step_accs, step_quats = [], []
                    for uav_name in sorted(all_data.keys(), key=lambda x: int(x.replace('uav', ''))):
                        if step in imu_by_t[uav_name]:
                            acc, quat = imu_by_t[uav_name][step]
                            step_accs.append(acc)
                            step_quats.append(quat)
                    if step_accs:
                        kf.predict(step_accs, step_quats)
                        buffer_idx = delay_idx + (step - (t - delay))
                        if buffer_idx < len(state_buffers):
                            state_buffers[buffer_idx] = (kf.x.copy(), kf.P.copy(), step)

        # collect states per UAV
        states = kf.get_states()
        for uav_name in results:
            results[uav_name].append(states[uav_name])

    # to DataFrames
    for uav_name in results:
        results[uav_name] = pd.DataFrame(results[uav_name],
                                         columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
    return results, real_pos
