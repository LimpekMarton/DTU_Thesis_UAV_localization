import pandas as pd
import glob
import os
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy.spatial.transform import Rotation
from src.localization import mds_map_p
from src.simulation import environment
from src.models import uav




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
        print(f"Found files: {files}")
        df_dict = {}
        for file in files:
            uav_name = os.path.basename(file).split('_')[0]
            try:
                df_dict[uav_name] = pd.read_csv(file, on_bad_lines='skip')  # Skip bad lines
            except pd.errors.ParserError as e:
                print(f"Error parsing {file}: {str(e)}")
                return None
        return df_dict
    except FileNotFoundError:
        print("Error: One or more files not found")
        return None
    except pd.errors.EmptyDataError:
        print("Error: One or more CSV files are empty")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        return None

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

class UAV_KF(ExtendedKalmanFilter):
    def __init__(self, dt=0.1, initial_pos=None, initial_vel=None, gravity=9.81):
        super().__init__(dim_x=6, dim_z=3)  # x: [pos, vel], z: [x, y, z]
        self.dt = dt
        self.gravity = gravity
        self.x = np.zeros(6)  # [x, y, z, vx, vy, vz]
        if initial_pos is not None:
            self.x[:3] = np.array(initial_pos, dtype=float)
        if initial_vel is not None:
            self.x[3:6] = np.array(initial_vel, dtype=float)

        # Covariance matrix
        self.P = np.eye(6) * 50.0

        # Process noise
        self.imu_acc_noise_var = 0.008  # IMU acceleration noise
        self.imu_quat_noise_var = 0.001  # Quaternion noise
        sigma_acc_total = np.sqrt(self.imu_acc_noise_var + self.imu_quat_noise_var)
        B = np.zeros((6, 3))
        B[0:3, 0:3] = 0.5 * dt ** 2 * np.eye(3)  # Position control
        B[3:6, 0:3] = dt * np.eye(3)  # Velocity control
        self.Q = B @ (sigma_acc_total ** 2 * np.eye(3)) @ B.T

        # Measurement noise
        self.R = np.eye(3) * 35.0  # MDS

    def predict(self, imu_acc, imu_quat):
        dt = self.dt
        # 1. Kinematic State Propagation
        prev_x = self.x.copy()
        imu_quat_normalized = imu_quat / np.linalg.norm(imu_quat)
        rot = Rotation.from_quat(imu_quat_normalized)
        acc_world = rot.apply(np.array(imu_acc, dtype=float))
        acc_world[2] -= self.gravity

        new_x = np.zeros(6)
        new_x[0:3] = prev_x[0:3] + prev_x[3:6] * dt + 0.5 * acc_world * dt ** 2
        new_x[3:6] = prev_x[3:6] + acc_world * dt
        self.x = new_x

        # 2. Covariance Propagation
        # State Transition Matrix F
        F = np.eye(6)
        F[0:3, 3:6] = dt * np.eye(3)
        self.F = F

        # Process Noise Q (already set in __init__, reused here)
        # Numerical stability
        self.P = F @ self.P @ F.T + self.Q
        self.P = (self.P + self.P.T) / 2
        self.P += 1e-6 * np.eye(6)

    def update(self, mds_meas):
        meas = np.array(mds_meas[:3], dtype=float)

        def HJacobian(x):
            return np.hstack((np.eye(3), np.zeros((3, 3))))

        def Hx(x):
            return x[:3]

        super().update(meas, HJacobian=HJacobian, Hx=Hx)

        # Numerical stability
        self.P = (self.P + self.P.T) / 2
        self.P += 1e-6 * np.eye(6)

    def get_state(self):
        return self.x.copy()

def run_kf_mds(noise_std=0.0, mds_method=mds_map_p.run_mds_map_p, delay=10, number_of_uavs=5):
    if mds_method is None:
        raise ValueError("mds_method must be provided")
    all_data = load_uav_data(N=number_of_uavs)
    if not all_data:
        return -1
    real_pos = compute_real_positions(all_data)
    env = environment.Environment(dimensions=np.array([150.0, 150.0, 100.0]), uav_number=0)
    for uav_name in all_data:
        uav_id = int(uav_name.replace('uav', ''))
        df = all_data[uav_name]
        raw_pos = np.array([df.iloc[0]['Local_X'], df.iloc[0]['Local_Y'], df.iloc[0]['Local_Z']])
        uav_ = uav.UAV(id=uav_id, position=raw_pos)
        env.add_uav(uav_)
    env.connect_all_uavs()
    initial_mds = mds_method(env, noise_std=noise_std)

    kfs = {}
    results = {}
    state_buffers = {}
    mds_buffer = {uav_name: [] for uav_name in all_data}
    mds_time = {uav_name: [] for uav_name in all_data}
    imu_by_t = {uav_name: {} for uav_name in all_data}
    timesteps = min(len(df) for df in all_data.values())
    timesteps = min(timesteps, 10000)

    for uav_name in all_data:
        uav_id = int(uav_name.replace('uav', ''))
        initial_pos = initial_mds[uav_id, :]
        if uav_id < 4:
            continue
        kfs[uav_name] = UAV_KF(dt=0.1, initial_pos=initial_pos)
        results[uav_name] = [kfs[uav_name].get_state()]
        state_buffers[uav_name] = []

    for t in range(timesteps):
        for uav_name in all_data:
            uav_id = int(uav_name.replace('uav', ''))
            raw_pos = np.array([all_data[uav_name].iloc[t]['Local_X'], all_data[uav_name].iloc[t]['Local_Y'],
                                all_data[uav_name].iloc[t]['Local_Z']])
            env.uavs[uav_id].position = raw_pos
            row = all_data[uav_name].iloc[t]
            imu_acc = np.array(
                [row['Linear_Acceleration_X'], row['Linear_Acceleration_Y'], row['Linear_Acceleration_Z']], dtype=float)
            imu_quat = np.array(
                [row['Orientation_X'], row['Orientation_Y'], row['Orientation_Z'], row['Orientation_W']])

            if uav_id < 4:
                imu_by_t[uav_name][t] = (imu_acc, imu_quat)
                continue

            kf = kfs[uav_name]
            kf.predict(imu_acc, imu_quat)
            state_buffers[uav_name].append((kf.x.copy(), kf.P.copy(), t))
            if len(state_buffers[uav_name]) > delay + 1:
                state_buffers[uav_name].pop(0)

            imu_by_t[uav_name][t] = (imu_acc, imu_quat)

            if t < timesteps - 1:
                results[uav_name].append(kf.get_state())

        if t % delay == 0 and t >= delay:
            mds_global = mds_method(env, noise_std=noise_std)
            for uav_name in all_data:
                uav_id = int(uav_name.replace('uav', ''))
                mds_buffer[uav_name].append(mds_global[uav_id, :])
                mds_time[uav_name].append(t)
                if len(mds_buffer[uav_name]) > 2:
                    mds_buffer[uav_name].pop(0)
                    mds_time[uav_name].pop(0)

        for uav_name in all_data:
            uav_id = int(uav_name.replace('uav', ''))
            if uav_id < 4:
                continue
            if mds_time[uav_name] and mds_time[uav_name][0] == t - delay:
                mds_meas = mds_buffer[uav_name][0]
                kf = kfs[uav_name]
                delay_idx = None
                for i, (_, _, past_t) in enumerate(state_buffers[uav_name]):
                    if past_t == t - delay:
                        delay_idx = i
                        break
                if delay_idx is not None:
                    past_x, past_P, _ = state_buffers[uav_name][delay_idx]
                    kf.x = past_x.copy()
                    kf.P = past_P.copy()
                    kf.update(mds_meas)
                    for step in range(t - delay + 1, t + 1):
                        if step in imu_by_t[uav_name]:
                            imu_acc, imu_quat = imu_by_t[uav_name][step]
                            kf.predict(imu_acc, imu_quat)
                        buffer_idx = delay_idx + (step - (t - delay))
                        if buffer_idx < len(state_buffers[uav_name]):
                            state_buffers[uav_name][buffer_idx] = (kf.x.copy(), kf.P.copy(), step)
                    for step in range(t - delay, t):
                        if step < len(results[uav_name]):
                            results[uav_name][step] = state_buffers[uav_name][delay_idx + (step - (t - delay))][0]

    for uav_name in results:
        results[uav_name] = pd.DataFrame(results[uav_name], columns=['x', 'y', 'z', 'vx', 'vy', 'vz'])
    return results, real_pos