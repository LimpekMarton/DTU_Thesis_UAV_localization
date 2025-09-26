import numpy as np
import time
import pandas as pd
from werkzeug.debug.repr import missing

from src.localization.static_localization_error import calculate_abs_mean_error
from src.simulation.environment import Environment
from src.localization import multidimensional_scaling
from src.visualization import plot_real_and_estimated_positions
from src.localization import multidimensional_scaling, mds_map, mds_map_p, smds_more_AN
from src.localization import SMDS_correct_missing_conn



def test_localization_method_full(method, number_of_test_cases):
    failed_tests = 0
    NUMBER_OF_TEST_CASES = number_of_test_cases
    UAV_NUMBER_OF_TEST = np.array([5, 7, 10, 20, 40, 80])
    NOISE_STD = np.arange(0, 15, 0.5)
    all_errors = np.zeros((len(UAV_NUMBER_OF_TEST), len(NOISE_STD), NUMBER_OF_TEST_CASES))
    for (uav_number, k) in zip(UAV_NUMBER_OF_TEST, range(len(UAV_NUMBER_OF_TEST))):
        errors_per_uav = np.zeros((len(NOISE_STD), NUMBER_OF_TEST_CASES))
        #print(errors_per_uav.shape)
        for (noise, j) in zip(NOISE_STD, range(len(NOISE_STD))):
            errors_per_noise = np.zeros(NUMBER_OF_TEST_CASES)
            for i in range(NUMBER_OF_TEST_CASES):
                try:
                    env = Environment(dimensions=np.array([140.0, 140.0, 95.0]), uav_number=uav_number, frequency=5e9)
                    env.connect_all_uavs()
                    positions_global_estimated = method(env, noise_std=noise)
                    if isinstance(positions_global_estimated, (int, np.integer)) and positions_global_estimated == -1:
                        errors_per_noise[i] = np.nan
                        failed_tests += 1
                    else:
                        errors_per_noise[i] = calculate_abs_mean_error(real_positions=env.get_uav_position_matrix_all(),
                                                         estimated_positions=positions_global_estimated)
                except Exception as e:
                    #print(f'Failed test case: {e}')
                    errors_per_noise[i] = np.nan
                    failed_tests += 1
            #print(errors_per_noise)
            errors_per_uav[j] = np.transpose(errors_per_noise)
        all_errors[k] = errors_per_uav

    for (uav_number, k) in zip(UAV_NUMBER_OF_TEST, range(len(UAV_NUMBER_OF_TEST))):
        for (noise, j) in zip(NOISE_STD, range(len(NOISE_STD))):
            print(f'Number of UAVs:{uav_number}, Noise std:{noise}, Failed tests:{np.isnan(all_errors[k][j]).sum()},'
                  f'Average mean abs error:{np.nanmean(all_errors[k][j])}')



def test_localization_method_sparse(method, number_of_test_cases, min_neighbours):
    failed_tests = 0
    NUMBER_OF_TEST_CASES = number_of_test_cases
    UAV_NUMBER_OF_TEST = np.array([5, 7, 10, 20, 40, 80])
    NOISE_STD = np.arange(0, 15, 0.5)
    all_errors = np.zeros((len(UAV_NUMBER_OF_TEST), len(NOISE_STD), NUMBER_OF_TEST_CASES))
    for (uav_number, k) in zip(UAV_NUMBER_OF_TEST, range(len(UAV_NUMBER_OF_TEST))):
        errors_per_uav = np.zeros((len(NOISE_STD), NUMBER_OF_TEST_CASES))
        for (noise, j) in zip(NOISE_STD, range(len(NOISE_STD))):
            errors_per_noise = np.zeros(NUMBER_OF_TEST_CASES)
            for i in range(NUMBER_OF_TEST_CASES):
                try:
                    env = Environment(dimensions=np.array([140.0, 140.0, 95.0]), uav_number=uav_number, frequency=5e9)
                    env.connect_uavs_randomly(min_neighbors=min_neighbours)
                    positions_global_estimated = method(env, noise_std=noise)
                    if isinstance(positions_global_estimated, (int, np.integer)) and positions_global_estimated == -1:
                        errors_per_noise[i] = np.nan
                        failed_tests += 1
                    else:
                        errors_per_noise[i] = calculate_abs_mean_error(real_positions=env.get_uav_position_matrix_all(),
                                                         estimated_positions=positions_global_estimated)
                except Exception as e:
                    #print(f'Failed test case: {e}')
                    errors_per_noise[i] = np.nan
                    failed_tests += 1
            #print(errors_per_noise)
            errors_per_uav[j] = np.transpose(errors_per_noise)
        all_errors[k] = errors_per_uav

    for (uav_number, k) in zip(UAV_NUMBER_OF_TEST, range(len(UAV_NUMBER_OF_TEST))):
        for (noise, j) in zip(NOISE_STD, range(len(NOISE_STD))):
            print(f'Number of UAVs:{uav_number}, Noise std:{noise}, Failed tests:{np.isnan(all_errors[k][j]).sum()},'
                  f' Average mean abs error:{np.nanmean(all_errors[k][j])}')



def test_localization_method_one(method, number_of_uavs=10, noise_std=0.0):
    UAV_NUMBER_OF_TEST = number_of_uavs
    NOISE_STD = noise_std
    try:
        env = Environment(dimensions=np.array([140.0, 140.0, 95.0]), uav_number=UAV_NUMBER_OF_TEST, frequency=5e9)
        env.connect_all_uavs()
        positions_global_estimated = method(env, noise_std=NOISE_STD)
        abs_error = calculate_abs_mean_error(real_positions=env.get_uav_position_matrix_all(),
                                         estimated_positions=positions_global_estimated)
        plot_real_and_estimated_positions(env.get_uav_position_matrix_all(), positions_global_estimated)
        print(f'Number of UAVs:{UAV_NUMBER_OF_TEST}, Noise std:{NOISE_STD}, Mean abs error:{abs_error}')

    except Exception as e:
        print(f'Error: {e}')


def test_localization_method_one_sparse(method, number_of_uavs=10, noise_std=0.0, min_neighbours=4):
    UAV_NUMBER_OF_TEST = number_of_uavs
    NOISE_STD = noise_std
    try:
        env = Environment(dimensions=np.array([140.0, 140.0, 95.0]), uav_number=UAV_NUMBER_OF_TEST, frequency=5e9)
        env.connect_uavs_randomly(min_neighbors=min_neighbours)
        positions_global_estimated = method(env, noise_std=NOISE_STD)
        abs_error = calculate_abs_mean_error(real_positions=env.get_uav_position_matrix_all(),
                                         estimated_positions=positions_global_estimated)
        plot_real_and_estimated_positions(env.get_uav_position_matrix_all(), positions_global_estimated)
        print(f'Number of UAVs:{UAV_NUMBER_OF_TEST}, Noise std:{NOISE_STD}, Mean abs error:{abs_error}')

    except Exception as e:
        print(f'Error: {e}')

def test_localization_method_one_smds_based(method, number_of_uavs=10, noise_std=0.0, phase_noise_std=0.0):
    UAV_NUMBER_OF_TEST = number_of_uavs
    NOISE_STD = noise_std
    try:
        env = Environment(dimensions=np.array([140.0,140.0, 95.0]), uav_number=UAV_NUMBER_OF_TEST, frequency=5e9)
        #env.connect_all_uavs()
        env.connect_uavs_randomly(min_neighbors=15)
        #positions_global_estimated = method(env, noise_std=noise_std, phase_noise_std=phase_noise_std,
        #                                    missing_percentage=0.0)
        positions_global_estimated = method(env, noise_std=noise_std, phase_noise_std=phase_noise_std)
        abs_error = calculate_abs_mean_error(real_positions=env.get_uav_position_matrix_all(),
                                             estimated_positions=positions_global_estimated)

        print(f'Number of UAVs:{UAV_NUMBER_OF_TEST}, Noise std:{NOISE_STD}, Phase noise std: {phase_noise_std} Mean abs error:{abs_error}')
        plot_real_and_estimated_positions(env.get_uav_position_matrix_all(), positions_global_estimated)
    except Exception as e:
        print(f'Error: {e}')

def test_localization_method_full_smds(method, number_of_test_cases):
    failed_tests = 0
    NUMBER_OF_TEST_CASES = number_of_test_cases
    UAV_NUMBER_OF_TEST = np.array([5, 7, 10, 20, 40, 80])
    NOISE_STD = np.arange(0, 4, 0.5)
    all_errors = np.zeros((len(UAV_NUMBER_OF_TEST), len(NOISE_STD), NUMBER_OF_TEST_CASES))
    for (uav_number, k) in zip(UAV_NUMBER_OF_TEST, range(len(UAV_NUMBER_OF_TEST))):
        errors_per_uav = np.zeros((len(NOISE_STD), NUMBER_OF_TEST_CASES))
        #print(errors_per_uav.shape)
        for (noise, j) in zip(NOISE_STD, range(len(NOISE_STD))):
            errors_per_noise = np.zeros(NUMBER_OF_TEST_CASES)
            for i in range(NUMBER_OF_TEST_CASES):
                try:
                    env = Environment(dimensions=np.array([140.0, 140.0, 95.0]), uav_number=uav_number, frequency=5e9)
                    env.connect_all_uavs()
                    positions_global_estimated = method(env, noise_std=noise, phase_noise_std=0.0)
                    if isinstance(positions_global_estimated, (int, np.integer)) and positions_global_estimated == -1:
                        errors_per_noise[i] = np.nan
                        failed_tests += 1
                    else:
                        errors_per_noise[i] = calculate_abs_mean_error(real_positions=env.get_uav_position_matrix_all(),
                                                         estimated_positions=positions_global_estimated)
                except Exception as e:
                    #print(f'Failed test case: {e}')
                    errors_per_noise[i] = np.nan
                    failed_tests += 1
            #print(errors_per_noise)
            errors_per_uav[j] = np.transpose(errors_per_noise)
        all_errors[k] = errors_per_uav

    for (uav_number, k) in zip(UAV_NUMBER_OF_TEST, range(len(UAV_NUMBER_OF_TEST))):
        for (noise, j) in zip(NOISE_STD, range(len(NOISE_STD))):
            print(f'Number of UAVs:{uav_number}, Noise std:{noise}, Failed tests:{np.isnan(all_errors[k][j]).sum()},'
                  f'Average mean abs error:{np.nanmean(all_errors[k][j])}')


def test_localization_method_all_and_plot(method, number_of_test_cases):
    failed_tests = 0
    NUMBER_OF_TEST_CASES = number_of_test_cases
    UAV_NUMBER_OF_TEST = np.array([5])
    NOISE_STD = np.arange(0, 15, 0.5)
    all_errors = np.zeros((len(UAV_NUMBER_OF_TEST), len(NOISE_STD), NUMBER_OF_TEST_CASES))
    for (uav_number, k) in zip(UAV_NUMBER_OF_TEST, range(len(UAV_NUMBER_OF_TEST))):
        errors_per_uav = np.zeros((len(NOISE_STD), NUMBER_OF_TEST_CASES))
        #print(errors_per_uav.shape)
        for (noise, j) in zip(NOISE_STD, range(len(NOISE_STD))):
            errors_per_noise = np.zeros(NUMBER_OF_TEST_CASES)
            for i in range(NUMBER_OF_TEST_CASES):
                try:
                    env = Environment(dimensions=np.array([140.0, 140.0, 95.0]), uav_number=uav_number, frequency=5e9)
                    env.connect_all_uavs()
                    positions_global_estimated = method(env, noise_std=noise)
                    if isinstance(positions_global_estimated, (int, np.integer)) and positions_global_estimated == -1:
                        errors_per_noise[i] = np.nan
                        failed_tests += 1
                    else:
                        errors_per_noise[i] = calculate_abs_mean_error(real_positions=env.get_uav_position_matrix_all(),
                                                         estimated_positions=positions_global_estimated)
                except Exception as e:
                    #print(f'Failed test case: {e}')
                    errors_per_noise[i] = np.nan
                    failed_tests += 1
            #print(errors_per_noise)
            errors_per_uav[j] = np.transpose(errors_per_noise)
        all_errors[k] = errors_per_uav



def test_all_methods_no_missing(N=1000, uav_number=10):
    num_targets = uav_number - 4

    def calculate_euclidean_error(real_positions, estimated_positions):
        deltas = estimated_positions - real_positions
        euclidean_dists = np.sqrt(np.sum(deltas**2, axis=1))  # Euclidean distance per target
        return np.mean(euclidean_dists)  # Mean over all targets

    def get_errors(method, number_of_test_cases, uav_number, phase_std=None):
        NUMBER_OF_TEST_CASES = number_of_test_cases
        NOISE_STD = np.arange(0, 10.5, 0.5)
        all_euclidean_errors = np.full((len(NOISE_STD), NUMBER_OF_TEST_CASES), np.nan)
        all_deltas = np.full((len(NOISE_STD), NUMBER_OF_TEST_CASES, num_targets, 3), np.nan)
        failed_per = np.zeros(len(NOISE_STD), dtype=int)

        for j, noise in enumerate(NOISE_STD):
            for i in range(NUMBER_OF_TEST_CASES):
                try:
                    env = Environment(dimensions=np.array([140.0, 140.0, 95.0]), uav_number=uav_number, frequency=5e9)
                    env.connect_all_uavs()
                    if phase_std is not None:
                        positions_global_estimated = method(env, noise_std=noise, phase_noise_std=phase_std, missing_percentage=0.0)
                    else:
                        positions_global_estimated = method(env, noise_std=noise)
                    if isinstance(positions_global_estimated, (int, np.integer)) and positions_global_estimated == -1:
                        failed_per[j] += 1
                    else:
                        target_real_positions = env.get_uav_position_matrix_all()[4:]
                        target_estimated_positions = positions_global_estimated[4:]
                        deltas = target_estimated_positions - target_real_positions
                        all_deltas[j, i] = deltas
                        all_euclidean_errors[j, i] = calculate_euclidean_error(
                            real_positions=target_real_positions,
                            estimated_positions=target_estimated_positions
                        )
                except Exception:
                    failed_per[j] += 1
        mean_euclidean_errors = np.nanmean(all_euclidean_errors, axis=1)
        return mean_euclidean_errors, failed_per, NOISE_STD, all_deltas

    methods = {
        'MDS': multidimensional_scaling.run_mds,
        'MDS-MAP': mds_map.run_mds_map,
        'MDS-MAP(P)': mds_map_p.run_mds_map_p,
    }
    smds_phase_stds = [0, 5, 10, 15, 20, 30]
    results_euclidean = {}
    failed_counts = {}
    results_deltas = {}

    for name, method in methods.items():
        mean_euclidean_errors, failed_per, noises, all_deltas = get_errors(method, N, uav_number)
        results_euclidean[name] = mean_euclidean_errors
        failed_counts[name] = failed_per
        results_deltas[name] = all_deltas
        print(f"Results for {name}")
        for j, noise in enumerate(noises):
            print(
                f'Number of UAVs:{uav_number}, Noise std:{noise}, Failed tests:{failed_per[j]}, Average Euclidean error:{mean_euclidean_errors[j]}'
            )

    for phase in smds_phase_stds:
        name = 'SMDS (no angle noise)' if phase == 0 else f'SMDS (angle noise std {phase} deg)'
        method = smds_more_AN.run_smds
        mean_euclidean_errors, failed_per, noises, all_deltas = get_errors(method, N, uav_number, phase_std=phase)
        results_euclidean[name] = mean_euclidean_errors
        failed_counts[name] = failed_per
        results_deltas[name] = all_deltas
        print(f"Results for {name}")
        for j, noise in enumerate(noises):
            print(
                f'Number of UAVs:{uav_number}, Noise std:{noise}, Failed tests:{failed_per[j]}, Average Euclidean error:{mean_euclidean_errors[j]}'
            )

    return {
        'deltas': results_deltas,  # dict of method: np.array (num_noise, N, num_targets, 3)
        'mean_euclidean_errors': results_euclidean,
        'failed_counts': failed_counts,
        'noises': noises,
        'uav_number': uav_number
    }



def test_all_methods_vary_connectivity2(N=1000, uav_number=10, noise_std=2.0):
    """Test MDS methods (MDS, MDS-MAP, MDS-MAP(P), SMDS) with fixed noise and varying average connectivity."""
    num_targets = uav_number - 4
    def get_errors(method, number_of_test_cases, uav_number, phase_std=None):
        NUMBER_OF_TEST_CASES = number_of_test_cases
        AVG_DEGREES = np.arange(4, uav_number, 1)  # Vary average degree from 2 to max
        all_mean_abs_errors = np.full((len(AVG_DEGREES), NUMBER_OF_TEST_CASES), np.nan)
        all_deltas = np.full((len(AVG_DEGREES), NUMBER_OF_TEST_CASES, num_targets, 3), np.nan)
        failed_per = np.zeros(len(AVG_DEGREES), dtype=int)
        for j, avg_degree in enumerate(AVG_DEGREES):
            for i in range(NUMBER_OF_TEST_CASES):
                try:
                    env = Environment(dimensions=np.array([140.0, 140.0, 95.0]), uav_number=uav_number, frequency=5e9)
                    env.connect_uavs_randomly(avg_degree=avg_degree)
                    if phase_std is not None:
                        positions_global_estimated = method(env, noise_std=noise_std, phase_noise_std=phase_std)
                    else:
                        positions_global_estimated = method(env, noise_std=noise_std)
                    if isinstance(positions_global_estimated, (int, np.integer)) and positions_global_estimated == -1:
                        failed_per[j] += 1
                    else:
                        target_real_positions = env.get_uav_position_matrix_all()[4:]
                        target_estimated_positions = positions_global_estimated[4:]
                        deltas = target_estimated_positions - target_real_positions
                        all_deltas[j, i] = deltas
                        all_mean_abs_errors[j, i] = np.mean(np.abs(deltas))
                except Exception:
                    failed_per[j] += 1
        mean_abs_errors = np.nanmean(all_mean_abs_errors, axis=1)
        return mean_abs_errors, failed_per, AVG_DEGREES, all_deltas

    methods = {
        'MDS': multidimensional_scaling.run_mds,
        'MDS-MAP': mds_map.run_mds_map,
        'MDS-MAP(P)': mds_map_p.run_mds_map_p,
    }
    smds_phase_stds = [0, 5, 10, 15, 20, 30]
    results_mean_abs = {}
    failed_counts = {}
    results_deltas = {}
    for name, method in methods.items():
        mean_abs_errors, failed_per, avg_degrees, all_deltas = get_errors(method, N, uav_number)
        results_mean_abs[name] = mean_abs_errors
        failed_counts[name] = failed_per
        results_deltas[name] = all_deltas
        print(f"Results for {name}")
        for j, degree in enumerate(avg_degrees):
            print(f'Number of UAVs:{uav_number}, Avg degree:{degree:.1f}, Failed tests:{failed_per[j]}, Average mean abs error:{mean_abs_errors[j]:.4f}')
    for phase in smds_phase_stds:
        name = 'SMDS (no angle noise)' if phase == 0 else f'SMDS (angle noise std {phase} deg)'
        #method = smds_more_AN.run_smds
        method = SMDS_correct_missing_conn.run_smds
        mean_abs_errors, failed_per, avg_degrees, all_deltas = get_errors(method, N, uav_number, phase_std=phase)
        results_mean_abs[name] = mean_abs_errors
        failed_counts[name] = failed_per
        results_deltas[name] = all_deltas
        print(f"Results for {name}")
        for j, degree in enumerate(avg_degrees):
            print(f'Number of UAVs:{uav_number}, Avg degree:{degree:.1f}, Failed tests:{failed_per[j]}, Average mean abs error:{mean_abs_errors[j]:.4f}')
    return {
        'deltas': results_deltas,  # dict of method: np.array (num_degrees, N, num_targets, 3)
        'mean_abs_errors': results_mean_abs,
        'failed_counts': failed_counts,
        'avg_degrees': avg_degrees,
        'uav_number': uav_number
    }

def test_all_methods_vary_connectivity(N=1000, uav_number=10, noise_std=2.0):
    """Test MDS methods (MDS, MDS-MAP, MDS-MAP(P), SMDS) with fixed noise and varying average connectivity."""
    num_targets = uav_number - 4

    def calculate_euclidean_error(real_positions, estimated_positions):
        """Calculate mean Euclidean distance error between real and estimated positions."""
        deltas = estimated_positions - real_positions
        euclidean_dists = np.sqrt(np.sum(deltas**2, axis=1))  # Euclidean distance per target
        return np.mean(euclidean_dists)  # Mean over all targets

    def get_errors(method, number_of_test_cases, uav_number, phase_std=None):
        NUMBER_OF_TEST_CASES = number_of_test_cases
        AVG_DEGREES = np.arange(4, uav_number, 1)  # Vary average degree from 4 to max
        all_mean_euclidean_errors = np.full((len(AVG_DEGREES), NUMBER_OF_TEST_CASES), np.nan)
        all_deltas = np.full((len(AVG_DEGREES), NUMBER_OF_TEST_CASES, num_targets, 3), np.nan)
        failed_per = np.zeros(len(AVG_DEGREES), dtype=int)

        for j, avg_degree in enumerate(AVG_DEGREES):
            for i in range(NUMBER_OF_TEST_CASES):
                try:
                    env = Environment(dimensions=np.array([140.0, 140.0, 95.0]), uav_number=uav_number, frequency=5e9)
                    env.connect_uavs_randomly(avg_degree=avg_degree)
                    if phase_std is not None:
                        positions_global_estimated = method(env, noise_std=noise_std, phase_noise_std=phase_std)
                    else:
                        positions_global_estimated = method(env, noise_std=noise_std)
                    if isinstance(positions_global_estimated, (int, np.integer)) and positions_global_estimated == -1:
                        failed_per[j] += 1
                    else:
                        target_real_positions = env.get_uav_position_matrix_all()[4:]
                        target_estimated_positions = positions_global_estimated[4:]
                        deltas = target_estimated_positions - target_real_positions
                        all_deltas[j, i] = deltas
                        all_mean_euclidean_errors[j, i] = calculate_euclidean_error(
                            real_positions=target_real_positions,
                            estimated_positions=target_estimated_positions
                        )
                except Exception:
                    failed_per[j] += 1
        mean_euclidean_errors = np.nanmean(all_mean_euclidean_errors, axis=1)
        return mean_euclidean_errors, failed_per, AVG_DEGREES, all_deltas

    methods = {
        'MDS': multidimensional_scaling.run_mds,
        'MDS-MAP': mds_map.run_mds_map,
        'MDS-MAP(P)': mds_map_p.run_mds_map_p,
    }
    smds_phase_stds = [0, 5, 10, 15, 20, 30]
    results_mean_euclidean = {}
    failed_counts = {}
    results_deltas = {}

    for name, method in methods.items():
        mean_euclidean_errors, failed_per, avg_degrees, all_deltas = get_errors(method, N, uav_number)
        results_mean_euclidean[name] = mean_euclidean_errors
        failed_counts[name] = failed_per
        results_deltas[name] = all_deltas
        print(f"Results for {name}")
        for j, degree in enumerate(avg_degrees):
            print(f'Number of UAVs:{uav_number}, Avg degree:{degree:.1f}, Failed tests:{failed_per[j]}, Average mean Euclidean error:{mean_euclidean_errors[j]:.4f}')

    for phase in smds_phase_stds:
        name = 'SMDS (no angle noise)' if phase == 0 else f'SMDS (angle noise std {phase} deg)'
        method = SMDS_correct_missing_conn.run_smds
        mean_euclidean_errors, failed_per, avg_degrees, all_deltas = get_errors(method, N, uav_number, phase_std=phase)
        results_mean_euclidean[name] = mean_euclidean_errors
        failed_counts[name] = failed_per
        results_deltas[name] = all_deltas
        print(f"Results for {name}")
        for j, degree in enumerate(avg_degrees):
            print(f'Number of UAVs:{uav_number}, Avg degree:{degree:.1f}, Failed tests:{failed_per[j]}, Average mean Euclidean error:{mean_euclidean_errors[j]:.4f}')

    return {
        'deltas': results_deltas,  # dict of method: np.array (num_degrees, N, num_targets, 3)
        'mean_euclidean_errors': results_mean_euclidean,
        'failed_counts': failed_counts,
        'avg_degrees': avg_degrees,
        'uav_number': uav_number
    }


def test_execution_times(N=100, uav_numbers=np.arange(5, 31, 5)):
    def get_times(method, number_of_test_cases, uav_number):
        NUMBER_OF_TEST_CASES = number_of_test_cases
        times = np.zeros(NUMBER_OF_TEST_CASES)
        failed = 0
        for i in range(NUMBER_OF_TEST_CASES):
            try:
                env = Environment(dimensions=np.array([140.0, 140.0, 95.0]), uav_number=uav_number, frequency=5e9)
                env.connect_all_uavs()  # Fully connected graph
                if uav_number < 4:
                    raise ValueError(f"Need at least 4 UAVs for anchors, got {uav_number}")
                start_time = time.time()
                result = method(env, noise_std=0.0)  # No noise
                times[i] = time.time() - start_time
                if isinstance(result, (int, np.integer)) and result == -1:
                    failed += 1
                    times[i] = np.nan
                    print(f"Failure in {method.__name__} with {uav_number} UAVs: Returned -1")
            except Exception as e:
                failed += 1
                times[i] = np.nan
                print(f"Error in {method.__name__} with {uav_number} UAVs: {str(e)}")
        avg_time = np.nanmean(times)
        return avg_time, failed, times

    methods = {
        'MDS': multidimensional_scaling.run_mds,
        'MDS-MAP': mds_map.run_mds_map,
        'MDS-MAP(P)': mds_map_p.run_mds_map_p,
        'SMDS': smds_more_AN.run_smds
    }
    results_avg_times = {}
    failed_counts = {}
    results_times = {}

    for name, method in methods.items():
        avg_times = []
        fails = []
        all_times = []
        for uav_number in uav_numbers:
            avg_time, failed, times = get_times(method, N, uav_number)
            avg_times.append(avg_time)
            fails.append(failed)
            all_times.append(times)
            print(f"{name}, UAVs={uav_number}, Avg time={avg_time:.4f}s, Failed={failed}/{N}")
        results_avg_times[name] = np.array(avg_times)
        failed_counts[name] = np.array(fails)
        results_times[name] = np.array(all_times)

    return {
        'avg_times': results_avg_times,  # dict of method: np.array (num_uav_numbers)
        'failed_counts': failed_counts,  # dict of method: np.array (num_uav_numbers)
        'times': results_times,  # dict of method: np.array (num_uav_numbers, N)
        'uav_numbers': np.array(uav_numbers)
    }