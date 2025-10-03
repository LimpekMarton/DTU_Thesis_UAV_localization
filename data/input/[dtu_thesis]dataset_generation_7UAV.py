#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
import joblib
import matplotlib.ticker as mtick
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use("ggplot")


# # Load raw data from simulator

# In[4]:


def open_csv(file_path):
  try:
    df=pd.read_csv(file_path)
    return df
  except:
    print("File not found")
    return -1


# In[5]:


#load files
uav0_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav0_data.csv')
uav1_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav1_data.csv')
uav2_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav2_data.csv')
uav3_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav3_data.csv')
uav4_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav4_data.csv')
uav5_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav5_data.csv')
uav6_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav6_data.csv')


# In[6]:


all_data = {}


# In[7]:


all_data['uav0'] = uav0_df_raw
all_data['uav1'] = uav1_df_raw
all_data['uav2'] = uav2_df_raw
all_data['uav3'] = uav3_df_raw
all_data['uav4'] = uav4_df_raw
all_data['uav5'] = uav5_df_raw
all_data['uav6'] = uav6_df_raw


# # True distances

# In[13]:


def prepare_data_distances_7uav(all_data):
    num_samples = min(len(all_data[f'uav{i}']) for i in range(7))

    all_positions = np.array([
        all_data[f'uav{i}'][['Local_X', 'Local_Y', 'Local_Z']].iloc[:num_samples].values
        for i in range(7)
    ])  # Shape: (7, num_samples, 3)

    anchor_positions = all_positions[:4]  # 0-3 anchors

    # Calculate unique pairwise distances (21 pairs for 7 UAVs)
    distances = np.zeros((num_samples, 21))
    pair_idx = 0
    for j in range(7):
        for k in range(j + 1, 7):
            distances[:, pair_idx] = np.sqrt(np.sum((all_positions[j] - all_positions[k])**2, axis=1))
            pair_idx += 1

    # Prepare input: anchor positions + all pairwise distances
    X = np.hstack([
        anchor_positions.transpose(1, 0, 2).reshape(num_samples, -1),  # 4 * 3 = 12
        distances  # 21
    ])  # Total: 12 + 21 = 33 features

    # Prepare output: positions of target UAVs (uav4, uav5, uav6)
    y = np.array([
        all_data[f'uav{i}'][['Local_X', 'Local_Y', 'Local_Z']].iloc[:num_samples].values
        for i in range(4, 7)
    ]).transpose(1, 0, 2).reshape(num_samples, -1)  # Shape: (num_samples, 3 * 3 = 9)

    return X, y

# Prepare data from all_data
X_dist, y_dist = prepare_data_distances_7uav(all_data)
np.save('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/X_ground_truth_dist_data.npy', X_dist)
np.save('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/y_ground_truth_dist_data.npy', y_dist)


# # True distances and true AoA

# In[14]:


def calculate_ground_truth_angles(pos1, pos2):
    """Calculate true azimuth and elevation between two positions.
    Args:
        pos1: (x, y, z) coordinates of first UAV
        pos2: (x, y, z) coordinates of second UAV
    Returns:
        (azimuth1, elevation1, azimuth2, elevation2): angles in degrees
    """
    delta = pos2 - pos1
    dist = np.linalg.norm(delta)
    azimuth1 = np.degrees(np.arctan2(delta[1], delta[0])) % 360
    elevation1 = np.degrees(np.arcsin(delta[2] / dist))
    delta_reverse = pos1 - pos2
    azimuth2 = np.degrees(np.arctan2(delta_reverse[1], delta_reverse[0])) % 360
    elevation2 = np.degrees(np.arcsin(delta_reverse[2] / dist))
    return azimuth1, elevation1, azimuth2, elevation2

def prepare_data_dist_and_aoa_7uav(all_data):
    num_samples = min(len(all_data[f'uav{i}']) for i in range(7))

    anchor_positions = np.array([
        all_data[f'uav{i}'][['Local_X', 'Local_Y', 'Local_Z']].iloc[:num_samples].values
        for i in range(4)
    ])  # Shape: (4, num_samples, 3)
    all_positions = np.array([
        all_data[f'uav{i}'][['Local_X', 'Local_Y', 'Local_Z']].iloc[:num_samples].values
        for i in range(7)
    ])  # Shape: (7, num_samples, 3)

    distances = np.zeros((num_samples, 21))
    pair_idx = 0
    for j in range(7):
        for k in range(j + 1, 7):
            distances[:, pair_idx] = np.sqrt(np.sum((all_positions[j] - all_positions[k])**2, axis=1))
            pair_idx += 1

    angles = np.zeros((num_samples, 21 * 4))  # 84
    pair_idx = 0
    for j in range(7):
        for k in range(j + 1, 7):
            for i in range(num_samples):
                az1, el1, az2, el2 = calculate_ground_truth_angles(all_positions[j, i], all_positions[k, i])
                angles[i, pair_idx * 4:(pair_idx + 1) * 4] = [az1, el1, az2, el2]
            pair_idx += 1

    X = np.hstack([
        anchor_positions.transpose(1, 0, 2).reshape(num_samples, -1),  # 12
        distances,  # 21
        angles  # 84
    ])  # Total: 12 + 21 + 84 = 117 features

    y = np.array([
        all_data[f'uav{i}'][['Local_X', 'Local_Y', 'Local_Z']].iloc[:num_samples].values
        for i in range(4, 7)
    ]).transpose(1, 0, 2).reshape(num_samples, -1)  # Shape: (num_samples, 3 * 3 = 9)

    print(f"X shape: {X.shape}")  # Should be (num_samples, 117)

    return X, y

X_dist_aoa, y_dist_aoa = prepare_data_dist_and_aoa_7uav(all_data)
np.save('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/X_ground_truth_dist_and_aoa_data.npy', X_dist_aoa)
np.save('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/y_ground_truth_dist_and_aoa_data.npy', y_dist_aoa)


# # Path loss (RSS)

# In[15]:


def prepare_data_rss_7uav(all_data):
    num_samples = min(len(all_data[f'uav{i}']) for i in range(7))

    anchor_positions = np.array([
        all_data[f'uav{i}'][['Local_X', 'Local_Y', 'Local_Z']].iloc[:num_samples].values
        for i in range(4)
    ])  # Shape: (4, num_samples, 3)

    if np.any(np.isnan(anchor_positions)):
        raise ValueError("NaN values detected in anchor positions")

    target_positions = np.array([
        all_data[f'uav{i}'][['Local_X', 'Local_Y', 'Local_Z']].iloc[:num_samples].values
        for i in range(4, 7)
    ]).transpose(1, 0, 2).reshape(num_samples, -1)  # Shape: (num_samples, 3 * 3 = 9)

    if np.any(np.isnan(target_positions)):
        raise ValueError("NaN values detected in target positions")

    path_losses = np.zeros((num_samples, 21))  # 21 pairs
    pair_idx = 0
    for j in range(7):
        for k in range(j + 1, 7):
            path_loss_jk = all_data[f'uav{j}'][f'Transmitter_{k}_path_loss'].iloc[:num_samples].values
            path_loss_kj = all_data[f'uav{k}'][f'Transmitter_{j}_path_loss'].iloc[:num_samples].values
            path_losses[:, pair_idx] = np.nanmean([path_loss_jk, path_loss_kj], axis=0)
            pair_idx += 1

    path_losses = np.nan_to_num(path_losses, nan=200.0)

    X = np.hstack([
        anchor_positions.transpose(1, 0, 2).reshape(num_samples, -1),  # 12
        path_losses  # 21
    ])  # Total: 12 + 21 = 33 features
    y = target_positions

    print(f"X shape: {X.shape}")  # Should be (num_samples, 33)
    print(f"y shape: {y.shape}")  # Should be (num_samples, 9)

    np.save('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/X_rss_data.npy', X)
    np.save('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/y_rss_data.npy', y)

    return X, y

X_rss, y_rss = prepare_data_rss_7uav(all_data)


# # Path loss (RSS) and AoA

# In[16]:


def prepare_data_rss_aoa_7uav(all_data):
    num_samples = min(len(all_data[f'uav{i}']) for i in range(7))

    anchor_positions = np.array([
        all_data[f'uav{i}'][['Local_X', 'Local_Y', 'Local_Z']].iloc[:num_samples].values
        for i in range(4)
    ])  # Shape: (4, num_samples, 3)
    all_positions = np.array([
        all_data[f'uav{i}'][['Local_X', 'Local_Y', 'Local_Z']].iloc[:num_samples].values
        for i in range(7)
    ])  # Shape: (7, num_samples, 3)

    path_losses = np.zeros((num_samples, 21))  # 21 pairs
    pair_idx = 0
    for j in range(7):
        for k in range(j + 1, 7):
            path_loss_jk = all_data[f'uav{j}'][f'Transmitter_{k}_path_loss'].iloc[:num_samples].values
            path_loss_kj = all_data[f'uav{k}'][f'Transmitter_{j}_path_loss'].iloc[:num_samples].values
            path_losses[:, pair_idx] = np.nanmean([path_loss_jk, path_loss_kj], axis=0)
            pair_idx += 1
    path_losses = np.nan_to_num(path_losses, nan=200.0)

    angles = np.zeros((num_samples, 21 * 4))  # 84
    pair_idx = 0
    for j in range(7):
        for k in range(j + 1, 7):
            for i in range(num_samples):
                az1, el1, az2, el2 = calculate_ground_truth_angles(all_positions[j, i], all_positions[k, i])
                angles[i, pair_idx * 4:(pair_idx + 1) * 4] = [az1, el1, az2, el2]
            pair_idx += 1

    X = np.hstack([
        anchor_positions.transpose(1, 0, 2).reshape(num_samples, -1),  # 12
        path_losses,  # 21
        angles  # 84
    ])  # Total: 12 + 21 + 84 = 117 features

    y = np.array([
        all_data[f'uav{i}'][['Local_X', 'Local_Y', 'Local_Z']].iloc[:num_samples].values
        for i in range(4, 7)
    ]).transpose(1, 0, 2).reshape(num_samples, -1)  # Shape: (num_samples, 3 * 3 = 9)

    print(f"X shape: {X.shape}")  # Should be (num_samples, 117)
    print(f"y shape: {y.shape}")  # Should be (num_samples, 9)

    np.save('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/X_rss_aoa_data.npy', X)
    np.save('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/y_rss_aoa_data.npy', y)

    return X, y

X_rss_aoa, y_rss_aoa = prepare_data_rss_aoa_7uav(all_data)


# In[ ]:





# In[ ]:




