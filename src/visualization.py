import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
#plt.style.use("ggplot")
sns.set_style("whitegrid")


def plot_uavs_position(env):
    """3D plot of UAV positions"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each UAV
    for uav_id, uav in env.uavs.items():
        ax.scatter(*uav.position, alpha=0.5, s=100, label=f'UAV {uav_id}')
        ax.text(*uav.position, f"{uav_id}", fontsize=20)

    # Environment bounds
    ax.set_xlim([0, env.dimensions[0]])
    ax.set_ylim([0, env.dimensions[1]])
    ax.set_zlim([0, env.dimensions[2]])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('UAV Positions')
    ax.legend()
    #ax.set_facecolor('lightgray')
    ax.view_init(elev=30, azim=45)
    plt.show()


def plot_uavs_with_distances(env):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot UAVs
    for uav_id, uav in env.uavs.items():
        ax.scatter(*uav.position, alpha=0.5, s=100, label=f'UAV {uav_id}')
        ax.text(*uav.position, f"{uav_id}", fontsize=20)

    # Plot connections with distance labels
    dist_matrix = env.get_distance_matrix()
    for i in range(dist_matrix.shape[0]):
        for j in range(i + 1, dist_matrix.shape[1]):
            if dist_matrix[i][j] > 0:
                pos1 = env.uavs[i].position
                pos2 = env.uavs[j].position

                # Connection line
                ax.plot([pos1[0], pos2[0]],
                        [pos1[1], pos2[1]],
                        [pos1[2], pos2[2]],
                        'b--', alpha=0.3)

                # Distance label at midpoint
                mid = (pos1 + pos2) / 2
                #ax.text(*mid, f"{dist_matrix[i][j]:.1f}m", color='red', fontsize=10)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.legend()
    ax.view_init(elev=30, azim=45)
    plt.show()


def visualize_3d_coordinates(coordinates, title="3D Coordinates Visualization"):
    coords = np.array(coordinates)

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    # Plot the points
    scatter = ax.scatter(x, y, z, c='r', marker='o', s=50, depthshade=True)

    # Add labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(title)

    # Add annotations for each point
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.text(xi, yi, zi, f' {i}', color='blue', fontsize=8)

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Requires matplotlib >= 3.3.0
    # Alternative for older versions:
    # ax.set_aspect('equal')

    # Add grid
    ax.grid(True)
    ax.view_init(elev=30, azim=45)
    plt.show()


def plot_real_and_estimated_positions(real_positions, estimated_positions):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    real = np.array(real_positions)
    est = np.array(estimated_positions)

    # Calculate position errors
    errors = np.linalg.norm(real - est, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    # Plot with size proportional to error
    sc1 = ax.scatter(real[:, 0], real[:, 1], real[:, 2],
                     c='b', marker='o', s=50, label='Real Positions', alpha=0.8)
    sc2 = ax.scatter(est[:, 0], est[:, 1], est[:, 2],
                     c=errors, cmap='autumn_r', marker='^', s=50,
                     label='Estimated Positions', alpha=0.8)

    # Add error vectors
    for r, e, err in zip(real, est, errors):
        ax.quiver(r[0], r[1], r[2],
                  e[0] - r[0], e[1] - r[1], e[2] - r[2],
                  color='purple', arrow_length_ratio=0.1, linestyle=':',
                  linewidth=0.8, alpha=0.5)
        # Label each point with error distance
        ax.text(e[0], e[1], e[2], f'{err:.2f}m', color='darkred', fontsize=8)

    # Add colorbar for error magnitudes
    cbar = fig.colorbar(sc2, ax=ax, shrink=0.6)
    cbar.set_label('Position Error (m)')

    # Add stats annotation
    #stats_text = f'Mean Error: {mean_error:.2f}m\nMax Error: {max_error:.2f}m'
    #ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes,
    #          bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('UAV Position Estimation Error Analysis')
    ax.legend()
    ax.view_init(elev=30, azim=45)
    ax.grid(True)

    plt.tight_layout()
    plt.show()



def plot_smds_edge_vectors(env, edge_vectors: np.ndarray, edge_pairs: list, title="SMDS Edge Vectors Visualization"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot UAV positions
    for uav_id, uav in env.uavs.items():
        ax.scatter(*uav.position, c='b', marker='o', s=100, label=f'UAV {uav_id}' if uav_id == 0 else "", alpha=0.8)
        ax.text(*uav.position, f"{uav_id}", fontsize=15, color='red')
        #print(f'UAV {uav_id}, position: {uav.position}')

    # Compute true edge vectors and errors
    true_edge_vectors = []
    errors = []
    for (i, j), v_est in zip(edge_pairs, edge_vectors):
        true_v = env.uavs[j].position - env.uavs[i].position
        true_edge_vectors.append(true_v)
        errors.append(np.linalg.norm(v_est - true_v))
    true_edge_vectors = np.array(true_edge_vectors)
    mean_error = np.mean(errors) if errors else 0
    max_error = np.max(errors) if errors else 0

    # Plot estimated edge vectors
    for (i, j), v_est in zip(edge_pairs, edge_vectors):
        pos_i = env.uavs[i].position
        ax.quiver(
            pos_i[0], pos_i[1], pos_i[2],
            v_est[0], v_est[1], v_est[2],
            color='purple', linestyle='-', linewidth=2, alpha=0.6, arrow_length_ratio=0.2,
            label='Estimated Edge Vectors' if (i, j) == edge_pairs[0] else ""
        )
        # Label magnitude at midpoint
        mid = pos_i + v_est / 2
        ax.text(*mid, f"{np.linalg.norm(v_est):.1f}m", color='purple', fontsize=8)

    # Plot true edge vectors
    for (i, j), v_true in zip(edge_pairs, true_edge_vectors):
        pos_i = env.uavs[i].position
        ax.quiver(
            pos_i[0], pos_i[1], pos_i[2],
            v_true[0], v_true[1], v_true[2],
            color='green', linestyle='--', linewidth=1, alpha=0.4, arrow_length_ratio=0.2,
            label='True Edge Vectors' if (i, j) == edge_pairs[0] else ""
        )


    # Add error statistics
    stats_text = f'Mean Edge Error: {mean_error:.2f}m\nMax Edge Error: {max_error:.2f}m'
    ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Set axes and labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.set_xlim([0, env.dimensions[0]])
    ax.set_ylim([0, env.dimensions[1]])
    ax.set_zlim([0, env.dimensions[2]])
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()


def plot_smds_true_edge_vectors(env, edge_pairs: list, title="SMDS True Edge Vectors"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot UAVs
    for uav_id, uav in env.uavs.items():
        ax.scatter(*uav.position, c='b', s=100, alpha=0.5, label=f'UAV {uav_id}' if uav_id == 0 else "")
        ax.text(*uav.position, f"{uav_id}", fontsize=10)
        #print(f'UAV {uav_id}, position: {uav.position}')

    # Count and plot true edge vectors
    edge_count = 0
    for i, (n, m) in enumerate(edge_pairs):
        v = env.uavs[m].position - env.uavs[n].position
        pos_n = env.uavs[n].position
        ax.quiver(
            pos_n[0], pos_n[1], pos_n[2],
            v[0], v[1], v[2],
            color='green', linestyle='--', linewidth=0.8, alpha=0.7,
            arrow_length_ratio=0.1,  # Smaller arrowheads for clarity
            label='True Edge Vectors' if i == 0 else ""
        )
        mid = pos_n + v / 2
        ax.text(*mid, f"{np.linalg.norm(v):.1f}m", color='green', fontsize=8)
        edge_count += 1

    # Add edge count to title
    title_with_count = f"{title} (Edges: {edge_count})"

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim([0, env.dimensions[0]])
    ax.set_ylim([0, env.dimensions[1]])
    ax.set_zlim([0, env.dimensions[2]])
    ax.set_title(title_with_count)
    plt.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()


def visualize_positions(shifted_real_pos, estimates):
    for uav_name in estimates:
        uav_id = int(uav_name.replace('uav', ''))
        if uav_id < 4:  # Skip anchor UAVs (uav0 to uav3)
            continue
        real = shifted_real_pos[uav_name].values[:10000]
        est = estimates[uav_name][['x', 'y', 'z']].values
        plot_trajectory(real, est, uav_name)  # Pass uav_name for title

def visualize_absolute_errors(shifted_real_pos, estimates):
    plt.figure(figsize=(10, 6))
    for uav_name in estimates:
        uav_id = int(uav_name.replace('uav', ''))
        if uav_id < 4:  # Skip anchor UAVs
            continue
        est = estimates[uav_name][['x', 'y', 'z']].values
        real = shifted_real_pos[uav_name].values[:10000]
        errors = np.abs(est - real)
        timesteps = np.arange(len(errors))
        plt.plot(timesteps, errors[:, 0], label=f'{uav_name} |x_error|')
        plt.plot(timesteps, errors[:, 1], linestyle='--', label=f'{uav_name} |y_error|')
        plt.plot(timesteps, errors[:, 2], linestyle=':', label=f'{uav_name} |z_error|')
    plt.title('Absolute Position Errors for Target UAVs')
    plt.xlabel('Timestep')
    plt.ylabel('Absolute Error (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_absolute_errors_total(shifted_real_pos, estimates):
    plt.figure(figsize=(10, 6))
    for uav_name in estimates:
        uav_id = int(uav_name.replace('uav', ''))
        if uav_id < 4:  # Skip anchor UAVs
            continue
        est = estimates[uav_name][['x', 'y', 'z']].values
        real = shifted_real_pos[uav_name].values[:10000]
        errors = np.linalg.norm(est - real, axis=1)  # Euclidean distance
        timesteps = np.arange(len(errors))
        plt.plot(timesteps, errors, label=f'{uav_name} Absolute Error')
    plt.title('Total Absolute Position Error for Target UAVs')
    plt.xlabel('Timestep')
    plt.ylabel('Absolute Error (m)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_trajectory(real, est, uav_name):
    # Calculate Euclidean distance errors: sqrt(x_error^2 + y_error^2 + z_error^2)
    error = np.abs(real - est)
    euclidean_error = np.sqrt(np.sum(error ** 2, axis=1))
    mae = np.mean(euclidean_error)  # Mean of Euclidean errors
    variance = np.var(euclidean_error)  # Variance of Euclidean errors
    max_error = np.max(euclidean_error)  # Max Euclidean error

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(real[:, 0], real[:, 1], real[:, 2], label=f'{uav_name} Real', color='blue', linewidth=2)
    ax.plot(est[:, 0], est[:, 1], est[:, 2], label=f'{uav_name} Estimated', color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Real vs Estimated Positions for {uav_name}', fontsize=14, pad=20)
    ax.text2D(0.05, 0.95,
              f'Mean Euclidean Error: {mae:.2f} m\nVariance Euclidean Error: {variance:.2f} m²\nMax Euclidean Error: {max_error:.2f} m',
              transform=ax.transAxes, fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9))
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_error_distribution(results):
    deltas = results['deltas']
    noises = results['noises']
    uav_number = results['uav_number']
    num_targets = uav_number - 4

    # Set seaborn style globally
    sns.set_style("whitegrid")

    # Define a color palette for consistent colors across all noise levels
    palette = sns.color_palette("husl", n_colors=sum(noise % 2 == 0 for noise in noises))

    for method, delta_array in deltas.items():
        # Create a figure with subplots for each axis
        fig, (ax_x, ax_y, ax_z) = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        fig.suptitle(f'Error Distribution for {method} (UAVs: {uav_number})', fontsize=14, fontweight='bold')

        # Track handles and labels for the shared legend
        handles, labels = [], []
        color_idx = 0

        # Filter for even noise levels (noise_std % 2 == 0)
        for j, noise in enumerate(noises):
            if noise % 2 != 0:
                continue
            # Extract errors for this noise level
            errors_x = delta_array[j, :, :, 0].flatten()  # x-axis errors
            errors_y = delta_array[j, :, :, 1].flatten()  # y-axis errors
            errors_z = delta_array[j, :, :, 2].flatten()  # z-axis errors

            # Remove NaN values
            errors_x = errors_x[~np.isnan(errors_x)]
            errors_y = errors_y[~np.isnan(errors_y)]
            errors_z = errors_z[~np.isnan(errors_z)]

            # Plot normalized KDE for each axis
            if len(errors_x) > 1:  # Need at least 2 points for KDE
                kde_x = gaussian_kde(errors_x)
                x_range = np.linspace(max(min(errors_x), -50), min(max(errors_x), 50), 200)
                y_x = kde_x(x_range) / kde_x(x_range).max()  # Normalize to max height of 1
                line_x, = ax_x.plot(x_range, y_x, label=f'Noise {noise:.1f}', linewidth=1.5, color=palette[color_idx])
                handles.append(line_x)
                labels.append(f'Noise {noise:.1f}')
            if len(errors_y) > 1:
                kde_y = gaussian_kde(errors_y)
                y_range = np.linspace(max(min(errors_y), -50), min(max(errors_y), 50), 200)
                y_y = kde_y(y_range) / kde_y(y_range).max()  # Normalize to max height of 1
                ax_y.plot(y_range, y_y, linewidth=1.5, color=palette[color_idx])
            if len(errors_z) > 1:
                kde_z = gaussian_kde(errors_z)
                z_range = np.linspace(max(min(errors_z), -50), min(max(errors_z), 50), 200)
                y_z = kde_z(z_range) / kde_z(z_range).max()  # Normalize to max height of 1
                ax_z.plot(z_range, y_z, linewidth=1.5, color=palette[color_idx])

            color_idx += 1

        # Configure plots
        ax_x.set_title('X-Axis', fontsize=12)
        ax_x.set_xlabel('Error (m)', fontsize=10)
        ax_x.set_xlim(-40, 40)  # Set x-axis limits
        ax_x.grid(True, linestyle='--', alpha=0.7)

        ax_y.set_title('Y-Axis', fontsize=12)
        ax_y.set_xlabel('Error (m)', fontsize=10)
        ax_y.set_xlim(-40, 40)  # Set x-axis limits
        ax_y.grid(True, linestyle='--', alpha=0.7)

        ax_z.set_title('Z-Axis', fontsize=12)
        ax_z.set_xlabel('Error (m)', fontsize=10)
        ax_z.set_xlim(-40, 40)  # Set x-axis limits
        ax_z.grid(True, linestyle='--', alpha=0.7)

        # Shared y-label
        fig.text(0.04, 0.5, 'Normalized Density', va='center', rotation='vertical', fontsize=10)

        # Add single legend outside the plots
        if handles:
            fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=10,
                       title='Noise STD (m)', bbox_transform=fig.transFigure)

        # Remove individual legends if they exist
        for ax in [ax_x, ax_y, ax_z]:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        plt.tight_layout(rect=[0.05, 0.05, 0.9, 0.95])
        plt.show()


def visualize_mean_errors(results):
    """Visualize the mean Euclidean distance errors for all methods across noise levels in a single plot with highly distinct colors."""
    #mean_euclidean_errors = results['mean_euclidean_errors']
    mean_euclidean_errors = results['mean_abs_errors']
    noises = results['noises']
    uav_number = results['uav_number']

    # Set seaborn style
    sns.set_style("whitegrid")

    # Define a color palette with highly distinct colors
    palette = sns.color_palette("tab10", n_colors=len(mean_euclidean_errors))

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot mean errors for each method with distinct colors
    for (method, errors), color in zip(mean_euclidean_errors.items(), palette):
        plt.plot(noises, errors, label=method, linewidth=2, marker='o', markersize=5, color=color)

    # Configure plot
    plt.title(f'Mean Euclidean Distance Errors for All Methods (UAVs: {uav_number})', fontsize=14, fontweight='bold')
    plt.xlabel('Noise STD (m)', fontsize=12)
    plt.ylabel('Mean Euclidean Distance Error (m)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=10, title='Method')
    plt.tight_layout()

    plt.show()

def visualize_error_variances(results):
    """Visualize the error variances for all methods across noise levels in a single plot with highly distinct colors."""
    deltas = results['deltas']
    noises = results['noises']
    uav_number = results['uav_number']

    # Compute variances per method
    def error_variance(delta):
        e = np.linalg.norm(delta, axis=-1)
        e_flat = e.reshape(e.shape[0], -1)
        return e_flat.var(axis=1, ddof=1)

    error_vars = {method: error_variance(delta) for method, delta in deltas.items()}

    # Set seaborn style
    sns.set_style("whitegrid")

    # Define a color palette with highly distinct colors
    palette = sns.color_palette("tab10", n_colors=len(error_vars))

    # Create a figure
    plt.figure(figsize=(10, 6))

    # Plot variances for each method with distinct colors
    for (method, vars_), color in zip(error_vars.items(), palette):
        plt.plot(noises, vars_, label=method, linewidth=2, marker='o', markersize=5, color=color)

    # Configure plot
    plt.title(f'Error Variances for All Methods (UAVs: {uav_number})', fontsize=14, fontweight='bold')
    plt.xlabel('Noise STD (m)', fontsize=12)
    plt.ylabel('Error Variance (m²)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=10, title='Method')
    plt.tight_layout()

    plt.show()


def visualize_robust_variances(results):
    """Print and visualize robust error variances for all methods."""
    deltas = results['deltas']
    noises = results['noises']
    uav_number = results['uav_number']

    def robust_var(delta, q=0.95):
        e = np.linalg.norm(delta, axis=-1).reshape(delta.shape[0], -1)
        vars_ = []
        for errs in e:
            cutoff = np.quantile(errs, q)
            trimmed = errs[errs <= cutoff]
            vars_.append(trimmed.var(ddof=1))
        return np.array(vars_)

    error_vars = {method: robust_var(delta) for method, delta in deltas.items()}

    # --- PRINT variances ---
    for method, vars_ in error_vars.items():
        print(f"\n{method} – variance:")
        for n, v in zip(noises, vars_):
            print(f"  noise={n}: var={v:.6f}")

    # --- PLOT variances ---
    sns.set_style("whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(error_vars))

    plt.figure(figsize=(10, 6))
    for (method, vars_), color in zip(error_vars.items(), palette):
        plt.plot(noises, vars_, label=method, linewidth=2, marker='o', markersize=5, color=color)

    plt.title(f'Error Variances for All Methods (UAVs: {uav_number})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Noise STD (m)', fontsize=12)
    plt.ylabel('Error Variance (m^2)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=10, title='Method')
    plt.tight_layout()
    plt.show()


def visualize_connectivity_results2(results):
    mean_abs_errors = results['mean_abs_errors']
    avg_degrees = results['avg_degrees']
    uav_number = results['uav_number']

    # Plot mean absolute errors
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, errors in mean_abs_errors.items():
        ax.plot(avg_degrees, errors, label=method, marker='o')
    ax.set_xlabel('Average Connectivity Degree')
    ax.set_ylabel('Mean Absolute Error (m)')
    ax.set_title(f'Mean Absolute Error vs Average Connectivity (UAVs={uav_number}, Noise Std=0.0m)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_connectivity_results(results):
    mean_euclidean_errors = results['mean_euclidean_errors']
    avg_degrees = results['avg_degrees']
    uav_number = results['uav_number']

    # Set seaborn style
    sns.set_style("whitegrid")

    # Define a color palette with highly distinct colors
    palette = sns.color_palette("tab10", n_colors=len(mean_euclidean_errors))

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean errors for each method with distinct colors
    for (method, errors), color in zip(mean_euclidean_errors.items(), palette):
        ax.plot(avg_degrees, errors, label=method, marker='o', linewidth=2, markersize=5, color=color)

    # Configure plot
    ax.set_xlabel('Average Connectivity Degree', fontsize=12)
    ax.set_ylabel('Mean Euclidean Distance Error (m)', fontsize=12)
    ax.set_title(f'Mean Euclidean Distance Error vs Average Connectivity (UAVs={uav_number}, Noise Std=2.0m)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10, title='Method')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()




def visualize_execution_times(results):
    avg_times = results['avg_times']
    uav_numbers = results['uav_numbers']

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot average execution times for each method
    for method, times in avg_times.items():
        ax.plot(uav_numbers, times, label=method, marker='o')
    ax.set_xlabel('Number of UAVs')
    ax.set_ylabel('Average Execution Time (s)')
    ax.set_title('Average Execution Time vs Number of UAVs (Fully Connected, No Noise)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_execution_times2(results):
    avg_times = results['avg_times']
    uav_numbers = results['uav_numbers']

    # Print table of execution times in milliseconds
    print("\nAverage Execution Times (milliseconds):")
    print("UAVs\t" + "\t".join(avg_times.keys()))
    for i, uav_num in enumerate(uav_numbers):
        times_ms = [avg_times[method][i] * 1000 for method in avg_times]
        print(f"{uav_num}\t" + "\t".join(f"{t:.2f}" for t in times_ms))

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot average execution times in milliseconds
    for method, times in avg_times.items():
        ax.plot(uav_numbers, times * 1000, label=method, marker='o')
    ax.set_xlabel('Number of UAVs')
    ax.set_ylabel('Average Execution Time (ms)')
    ax.set_title('Average Execution Time vs Number of UAVs (Fully Connected, No Noise)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_execution_times_no_smds(results):
    avg_times = {k: v for k, v in results['avg_times'].items() if k != 'SMDS'}
    uav_numbers = results['uav_numbers']

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot average execution times for each method (excluding SMDS)
    for method, times in avg_times.items():
        ax.plot(uav_numbers, times, label=method, marker='o')
    ax.set_xlabel('Number of UAVs')
    ax.set_ylabel('Average Execution Time (s)')
    ax.set_title('Average Execution Time vs Number of UAVs (Fully Connected, No Noise)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()
