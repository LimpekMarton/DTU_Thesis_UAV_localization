import numpy as np

def calculate_rmse(real_positions, estimated_positions):

    # Calculate position errors
    errors = np.linalg.norm(real_positions - estimated_positions, axis=1)
    rmse = np.sqrt(np.mean(errors**2))


    return rmse


def calculate_mse(real_positions, estimated_positions):

    # Calculate position errors
    errors = np.linalg.norm(real_positions - estimated_positions, axis=1)
    mse = np.mean(errors**2)

    return mse

def calculate_abs_mean_error(real_positions, estimated_positions):

    # Calculate position errors
    errors = np.linalg.norm(real_positions - estimated_positions, axis=1)
    mean_error = np.mean(np.abs(errors))

    return mean_error