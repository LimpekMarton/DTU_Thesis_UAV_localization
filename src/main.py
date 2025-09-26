#!/usr/bin/env python3
"""
UAV Simulator Main Entry Point
"""
import numpy as np
import time
from simulation.environment import Environment
from simulation.test import test_localization_method_full, test_localization_method_full_smds
from simulation.test import test_localization_method_sparse
from simulation.test import test_localization_method_one
import visualization
from src.simulation.test import test_all_methods_vary_connectivity
from src.simulation.test import test_localization_method_one_smds_based, test_localization_method_one_sparse, \
    test_all_methods_no_missing, test_all_methods_vary_connectivity, test_execution_times
from src.visualization import plot_uavs_with_distances, visualize_absolute_errors_total, visualize_execution_times, visualize_execution_times_no_smds
from visualization import plot_uavs_position, plot_real_and_estimated_positions, visualize_positions, visualize_absolute_errors
from localization import multidimensional_scaling, static_localization_error
from src.visualization import visualize_error_variances, visualize_robust_variances, visualize_connectivity_results, visualize_execution_times2
from localization import SMDS_correct_missing_conn
from localization import mds_map
from localization import mds_map_p
from localization import smds
from localization import smds_more_AN
from localization import joint_EKF_full
from models.uav import UAV
import matplotlib.pyplot as plt
from localization import joint_EKF_full
from localization import EKF_full_test
from scipy import linalg
from src.visualization import visualize_error_distribution, visualize_mean_errors
import json
import pickle
import numpy as np
from scipy.stats import shapiro
from scipy.linalg import eigh




def error_variance(delta):
    # Euclidean error per sample
    e = np.linalg.norm(delta, axis=-1)
    e_flat = e.reshape(e.shape[0], -1)  # flatten all but noise axis
    return e_flat.var(axis=1, ddof=1)  # sample variance per noise level






def test_normality_errors_no_outliers(results):
    deltas = results['deltas']
    noises = results['noises']
    uav_number = results['uav_number']
    num_targets = uav_number - 4

    def remove_outliers(data):
        if len(data) < 4:  # Need enough data to calculate quartiles
            return np.array([])
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]

    for method, delta_array in deltas.items():
        print(f"\nNormality Test Results for {method} (UAVs: {uav_number})")
        for j, noise in enumerate(noises):
            # Extract errors for this noise level
            errors_x = delta_array[j, :, :, 0].flatten()  # x-axis errors
            errors_y = delta_array[j, :, :, 1].flatten()  # y-axis errors
            errors_z = delta_array[j, :, :, 2].flatten()  # z-axis errors

            # Remove NaN values
            errors_x = errors_x[~np.isnan(errors_x)]
            errors_y = errors_y[~np.isnan(errors_y)]
            errors_z = errors_z[~np.isnan(errors_z)]

            # Remove outliers
            errors_x_no_outliers = remove_outliers(errors_x)
            errors_y_no_outliers = remove_outliers(errors_y)
            errors_z_no_outliers = remove_outliers(errors_z)

            # Perform Shapiro-Wilk test for each axis
            p_x = np.nan
            p_y = np.nan
            p_z = np.nan

            if len(errors_x_no_outliers) > 3:  # Shapiro-Wilk requires at least 3 samples
                _, p_x = shapiro(errors_x_no_outliers)
            if len(errors_y_no_outliers) > 3:
                _, p_y = shapiro(errors_y_no_outliers)
            if len(errors_z_no_outliers) > 3:
                _, p_z = shapiro(errors_z_no_outliers)

            # Print results
            print(f"Noise STD: {noise:.1f}")
            print(f"  X-Axis: p-value = {p_x:.4f}, {'Normal' if p_x > 0.05 else 'Not Normal' if not np.isnan(p_x) else 'Insufficient Data'}")
            print(f"  Y-Axis: p-value = {p_y:.4f}, {'Normal' if p_y > 0.05 else 'Not Normal' if not np.isnan(p_y) else 'Insufficient Data'}")
            print(f"  Z-Axis: p-value = {p_z:.4f}, {'Normal' if p_z > 0.05 else 'Not Normal' if not np.isnan(p_z) else 'Insufficient Data'}")





def main():
    #"""Main execution flow"""
    #env = Environment(dimensions=np.array([30.0, 30.0, 30.0]), uav_number=5, frequency = 5e9)
    #env.connect_uavs_randomly(min_neighbors=4)
    #env.connect_all_uavs()

    #azimuth1, elevation1 = env.uavs[0].calculate_ground_truth_angles(env.uavs[1])
    #print(f"True Signal from: {azimuth1:.1f}° azimuth, {elevation1:.1f}° elevation")

    #azimuth, elevation = env.uavs[0].measure_aoa(env.uavs[1], phase_noise_std=0.1)
    #print(f"Signal from: {azimuth:.1f}° azimuth, {elevation:.1f}° elevation")

    # Antenna array centered at [0,0,0]
    #uav1 = UAV(id=1, position=np.array([0, 0, 0]))
    #uav2 = UAV(id=2, position=np.array([0, 10, 10]))  # Directly along +X axis

    #azimuth1, elevation1 = uav1.calculate_ground_truth_angles(uav2)
    #print(f"True Signal from: {azimuth1:.1f}° azimuth, {elevation1:.1f}° elevation")

    #UAV1 is the receiver here and calculates the angles supposing UAV2 send something
    #azimuth, elevation = uav1.measure_aoa(uav2)
    #print(f"Signal from: {azimuth:.1f}° azimuth, {elevation:.1f}° elevation")

    #azimuth2, elevation2 = uav2.calculate_ground_truth_angles(uav1)
    #print(f"True Signal from: {azimuth2:.1f}° azimuth, {elevation2:.1f}° elevation")

    #azimuth3, elevation3 = uav2.measure_aoa(uav1)
    #print(f"Signal from: {azimuth3:.1f}° azimuth, {elevation3:.1f}° elevation")

    #print(env.get_rss_isotropic_matrix())
    #print(env.get_distance_matrix())
    #print(env.estimate_distances_from_rss(env.get_rss_isotropic_matrix()))

    #mean_error=np.zeros(20)
    #envs = []
    #for i in range(0,20):
    #    envs.append(Environment(dimensions=np.array([30.0, 30.0, 30.0]), uav_number=5, frequency=5e9))
    #    envs[i].connect_all_uavs()
    #    dist = envs[i].estimate_distances_from_rss(envs[i].get_rss_isotropic_matrix(noise_std=i))
    #    print(dist)
    #    positions_mds_global_estimated = multidimensional_scaling.run_mds(envs[i], dist)
    #    mean_error[i]=np.mean(np.abs(envs[i].get_uav_position_matrix_all() - positions_mds_global_estimated))

    # Create plot
    #plt.figure(figsize=(10, 6))
    #plt.scatter(np.arrange(0,20), mean_error, alpha=0.7, s=50)
    #plt.plot(np.arange(0,20), mean_error, '.-', alpha=0.5)


    #plt.xlabel('Standard Deviation')
    #plt.ylabel('Mean Error')
    #plt.title('Mean Errors vs Standard Deviation')
    #plt.grid(True, alpha=0.3)
    #plt.show()





    #positions_mds_global_estimated = multidimensional_scaling.run_mds(env, env.estimate_distances_from_rss(env.get_rss_isotropic_matrix(noise_std=1)))
    #positions_mds_global_estimated = multidimensional_scaling.run_mds(env, env.get_distance_matrix())
    #visualization.plot_real_and_estimated_positions(env.get_uav_position_matrix_all(), positions_mds_global_estimated)

    # Visualize
    #plot_uavs_position(env)
    #visualization.plot_uavs_with_distances(env)
    #print(env.get_distance_matrix())

    ######Classic MDS method#####

    #Run MDS algorithm
    #positions_mds_global_estimated = multidimensional_scaling.run_mds(env, distance_measurement_noise_std=0.1)
    #plot_real_and_estimated_positions(env.get_uav_position_matrix_all(), positions_mds_global_estimated)
    #mse_mds_sum = 0
    #for i in range(10000):
    #    env = Environment(dimensions=np.array([30.0, 30.0, 30.0]), uav_number=10)
    #    env.connect_all_uavs()
    #    positions_mds_global_estimated = multidimensional_scaling.run_mds(env, env.get_noisy_distance_matrix(noise_std=0.2))
    #    mse_mds = static_localization_error.calculate_MSE(env.get_uav_position_matrix_all(), positions_mds_global_estimated)
    #    mse_mds_sum += mse_mds

    #print(f'\nMean MSE: {mse_mds_sum / 10000}')

    #test_localization_method_full(multidimensional_scaling.run_mds, number_of_test_cases=1000)
    #test_localization_method_sparse(multidimensional_scaling.run_mds, number_of_test_cases=1000, min_neighbours=4)

    #test_localization_method_full(smds.run_smds, number_of_test_cases=10)
    #print('-------------------------------------------------------------------')
    #test_localization_method_full(mds_map.run_mds_map, number_of_test_cases=10)
    #test_localization_method_one_smds_based(.run_smds, number_of_uavs=5, noise_std=0.0, phase_noise_std=0.0)
    #test_localization_method_full_smds(smds_more_AN.run_smds, number_of_test_cases=100)
    #test_localization_method_one_smds_based(SMDS_correct_missing_conn.run_smds, number_of_uavs=20, noise_std=0.0, phase_noise_std=0.0)
    #test_localization_method_one(mds_map_p.run_mds_map_p, number_of_uavs=15, noise_std=0.0)
    #test_localization_method_one(mds_map.run_mds_map, number_of_uavs=10, noise_std=10.0)

    #start_time = time.perf_counter()
    #est, real_pos = EKF_full_test.run_kf_mds(noise_std=0.0, number_of_uavs=9, mds_method=mds_map.run_mds_map, delay=15)
    #est, real_pos = EKF_full_test.run_kf_mds(noise_std=0.0, number_of_uavs=9, mds_method=smds_more_AN.run_smds, delay=25)
    #est, real_pos = joint_EKF_full.run_joint_ekf(number_of_uavs=9, noise_std=0.0, delay=1)

    #elapsed_time_ms = (time.perf_counter() - start_time) * 1000  # Convert to ms
    #print(f"Elapsed time: {elapsed_time_ms:.2f} milliseconds")
    #print(f"Average elapsed time per timestep: {elapsed_time_ms/10000:.3f} ms")





    #print(f"uav4 velocities:\n{est['uav4'][['x', 'y', 'z', 'vx', 'vy', 'vz']][100:145]}")

    #print(shifted_real_pos['uav1'].values[:20])
    #print(est['uav1'].values[:20])

    #print(est)
    #print(real_pos)


    #visualize_positions(real_pos, est)

    #visualize_absolute_errors(shifted_real_pos=real_pos, estimates=est)

    #visualize_absolute_errors_total(shifted_real_pos=real_pos, estimates=est)

    # Known real-domain GEK matrix

    #test_localization_method_one_sparse(multidimensional_scaling.run_mds, number_of_uavs=10, noise_std=0.0, min_neighbours=4)
    #test_localization_method_one(multidimensional_scaling.run_mds, number_of_uavs=5, noise_std=0.0)


    #env = Environment(dimensions=np.array([30.0, 30.0, 30.0]), uav_number=5, frequency=5e9)
    #env.connect_uavs_randomly(min_neighbors=1)
    #env.connect_all_uavs()

    #distance_matrix = env.get_noisy_distance_matrix(noise_std=0.0)
    #print(f"Noisy distance matrix original: {distance_matrix}")

    # Create incomplete matrix
    #mask = np.random.random(distance_matrix.shape) > 0.1  # 40% missing
    #distance_matrix_incomplete = distance_matrix.copy()
    #distance_matrix_incomplete[~mask] = np.nan
    #print("Incomplete distance matrix:\n", distance_matrix_incomplete)


    #n = distance_matrix.shape[0]
    #mask = (distance_matrix != 0) | (np.eye(n, dtype=bool))  # True for non-zero or diagonal
    #distance_matrix_incomplete = distance_matrix.copy()
    #distance_matrix_incomplete[~mask] = np.nan  # Set non-diagonal zeros to np.nan
    #print("Incomplete distance matrix:\n", distance_matrix_incomplete)
    #print("Mask (True for observed, False for missing):\n", mask)

    # Complete the matrix (using existing complete_edm function)
    #completed_distance_matrix = complete_edm(distance_matrix_incomplete, d=3, mask=mask)
    #print("Completed distance matrix:\n", completed_distance_matrix)


    #print("Completed distance matrix:\n", completed_distance_matrix)

    #n = completed_distance_matrix.shape[0]
    #H = np.eye(n) - np.ones((n, n)) / n
    #G = -0.5 * H @ (completed_distance_matrix ** 2) @ H
    #eigenvalues = np.linalg.eigh(G)[0]
    #print("Eigenvalues of Gram matrix:", eigenvalues)



    #results = test_all_methods_no_missing(N=10, uav_number=20)
    #results = test_all_methods_vary_connectivity(N=50, uav_number=15, noise_std=2.0)
    #results = test_execution_times(N=10)



    #print(results)

    # Save dictionary
    #with open('/Users/marton/PycharmProjects/UAV_Swarm_Simulator/data/output/mds_errors_15_UAV_spare.pkl', 'wb') as f:
    #    pickle.dump(results, f)

    # Load dictionary
    #with open('/Users/marton/PycharmProjects/UAV_Swarm_Simulator/data/output/mds_errors_15_UAV_spare.pkl', 'rb') as f:
    #    results = pickle.load(f)


    #print(results)

    #visualize_error_distribution(results)

    #visualize_mean_errors(results)

    #visualize_robust_variances(results)

    #test_normality_errors_no_outliers(results)

    #visualize_connectivity_results(results)

    #visualize_execution_times2(results)

    #visualize_execution_times_no_smds(results)

    # Save DataFrame to CSV
    #results.to_csv('/Users/marton/PycharmProjects/UAV_Swarm_Simulator/data/output/mds_errors_5_UAV_full_dist_test.csv', index=False)



################################################################







if __name__ == "__main__":
    main()