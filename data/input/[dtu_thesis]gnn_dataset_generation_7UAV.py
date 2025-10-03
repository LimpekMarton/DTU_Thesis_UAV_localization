#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import os
from time import time
import shutil
import argparse
import configparser
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from itertools import combinations
from torch_geometric.nn import GATConv, BatchNorm
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import global_add_pool
import copy
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GINEConv
from torch_geometric.nn import MessagePassing
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use("ggplot")


# # Load raw data from simulator

# In[2]:


def open_csv(file_path):
  try:
    df=pd.read_csv(file_path)
    return df
  except:
    print("File not found")
    return -1


# In[3]:


#load files
uav0_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav0_data.csv')
uav1_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav1_data.csv')
uav2_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav2_data.csv')
uav3_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav3_data.csv')
uav4_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav4_data.csv')
uav5_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav5_data.csv')
uav6_df_raw = open_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/uav6_data.csv')


# In[4]:


all_data = {}
all_data['uav0'] = uav0_df_raw
all_data['uav1'] = uav1_df_raw
all_data['uav2'] = uav2_df_raw
all_data['uav3'] = uav3_df_raw
all_data['uav4'] = uav4_df_raw
all_data['uav5'] = uav5_df_raw
all_data['uav6'] = uav6_df_raw


# # True distances

# In[5]:


def create_graph_dataset_distances_7uav(all_data, anchor_ids=[0, 1, 2, 3]):
    try:
        num_nodes = len(all_data)
        min_length = min(len(all_data[key]) for key in all_data)
        print(f"Number of nodes: {num_nodes}, Min length: {min_length}")

        feature_cols = ['Local_X', 'Local_Y', 'Local_Z']
        label_cols = ['Local_X', 'Local_Y', 'Local_Z']

        target_ids = list(range(max(anchor_ids) + 1, num_nodes))  # [4, 5, 6]
        for target_idx in target_ids:
            pos_df = all_data[f'uav{target_idx}'][label_cols]
            print(f"Position std for uav{target_idx}:\n", pos_df.std())
            print(f"Sample positions (uav{target_idx}, first 5 rows):\n", pos_df.head())

        dataset = []

        for t in range(min_length):
            node_features = []
            positions = []

            for i, uav_key in enumerate(sorted(all_data.keys())):
                df = all_data[uav_key]
                row = df.iloc[t]
                positions.append(row[label_cols].values)
                if i in anchor_ids:
                    features = row[feature_cols].values
                else:
                    features = np.zeros(len(feature_cols))
                node_features.append(features)

            x = torch.tensor(np.array(node_features), dtype=torch.float)
            # y contains all targets' positions, flattened
            target_positions = np.concatenate([positions[i] for i in target_ids])
            y = torch.tensor(target_positions, dtype=torch.float)

            edge_index = []
            edge_attr = []
            seen_pairs = set()
            for i, j in combinations(range(num_nodes), 2):
                if (i, j) not in seen_pairs and (j, i) not in seen_pairs:
                    pos_i, pos_j = positions[i], positions[j]
                    dist = np.sqrt(np.sum((pos_i - pos_j) ** 2))
                    if not np.isnan(dist):
                        edge_index.extend([[i, j]])
                        edge_attr.extend([[dist]])
                    seen_pairs.add((i, j))

            edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long) if edge_index else torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float) if edge_attr else torch.empty((0, 1), dtype=torch.float)

            if t < 5:
                edge_dict = {(i, j): edge_attr[idx][0] for idx, (i, j) in enumerate(edge_index.t().numpy())}
                print(f"Time step {t}, reduced edge distances: {edge_dict}")
                print(f"Number of edges: {len(edge_dict)}")

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.anchor_ids = anchor_ids
            dataset.append(data)

        print(f"Dataset created with {len(dataset)} graphs")
        return dataset

    except Exception as e:
        print(f"Error in create_graph_dataset: {str(e)}")
        return None

dataset_distances = create_graph_dataset_distances_7uav(all_data)
torch.save(dataset_distances, '/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/graph_data_distances.pt')


# # True distances and true AoA

# In[6]:


def calculate_ground_truth_angles(pos1, pos2):
    delta = pos2 - pos1
    dist = np.linalg.norm(delta)
    azimuth = np.degrees(np.arctan2(delta[1], delta[0])) % 360
    elevation = np.degrees(np.arcsin(delta[2] / dist))
    return azimuth, elevation

def create_graph_dataset_distances_aoa_7uav(all_data, anchor_ids=[0, 1, 2, 3]):
    try:
        num_nodes = len(all_data)
        min_length = min(len(all_data[key]) for key in all_data)
        print(f"Number of nodes: {num_nodes}, Min length: {min_length}")

        feature_cols = ['Local_X', 'Local_Y', 'Local_Z']
        label_cols = ['Local_X', 'Local_Y', 'Local_Z']

        target_ids = list(range(max(anchor_ids) + 1, num_nodes))  # [4, 5, 6]
        for target_idx in target_ids:
            pos_df = all_data[f'uav{target_idx}'][label_cols]
            print(f"Position std for uav{target_idx}:\n", pos_df.std())
            print(f"Sample positions (uav{target_idx}, first 5 rows):\n", pos_df.head())

        dataset = []

        for t in range(min_length):
            node_features = []
            positions = []

            # Collect positions and check for NaNs
            for i, uav_key in enumerate(sorted(all_data.keys())):
                df = all_data[uav_key]
                row = df.iloc[t]
                pos = row[label_cols].values
                if np.any(np.isnan(pos)):
                    raise ValueError(f"NaN detected in position for {uav_key} at time {t}")
                positions.append(pos)

            # Calculate AoA for each node (2 angles per incoming edge)
            aoa_features = []
            for i in range(num_nodes):
                angles = []
                for j in range(num_nodes):
                    if i != j:
                        az, el = calculate_ground_truth_angles(positions[i], positions[j])
                        angles.extend([az, el])
                aoa_features.append(angles)  # 6 other nodes * 2 angles = 12 features

            # Combine position and AoA features
            for i, uav_key in enumerate(sorted(all_data.keys())):
                if i in anchor_ids:
                    features = np.concatenate([positions[i], aoa_features[i]])  # 3 pos + 12 angles
                else:
                    features = np.zeros(3 + 12)  # Zero-pad target
                node_features.append(features)

            x = torch.tensor(np.array(node_features), dtype=torch.float)  # Shape: (7, 15)
            # y contains all targets' positions, flattened
            target_positions = np.concatenate([positions[i] for i in target_ids])
            y = torch.tensor(target_positions, dtype=torch.float)

            edge_index = []
            edge_attr = []
            seen_pairs = set()
            for i, j in combinations(range(num_nodes), 2):
                if (i, j) not in seen_pairs and (j, i) not in seen_pairs:
                    pos_i, pos_j = positions[i], positions[j]
                    dist = np.sqrt(np.sum((pos_i - pos_j) ** 2))
                    if not np.isnan(dist):
                        edge_index.extend([[i, j]])
                        edge_attr.extend([[dist]])
                    seen_pairs.add((i, j))

            edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long) if edge_index else torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float) if edge_attr else torch.empty((0, 1), dtype=torch.float)

            if t < 5:
                edge_dict = {(i, j): edge_attr[idx][0] for idx, (i, j) in enumerate(edge_index.t().numpy())}
                print(f"Time step {t}, reduced edge distances: {edge_dict}")
                print(f"Number of edges: {len(edge_dict)}")

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.anchor_ids = anchor_ids
            dataset.append(data)

        print(f"Dataset created with {len(dataset)} graphs")
        return dataset

    except Exception as e:
        print(f"Error in create_graph_dataset: {str(e)}")
        return None

dataset_dist_aoa = create_graph_dataset_distances_aoa_7uav(all_data)
torch.save(dataset_dist_aoa, '/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/graph_data_dist_aoa.pt')


# # Path loss (RSS)

# In[7]:


def create_graph_dataset_rss_7uav(all_data, anchor_ids=[0, 1, 2, 3]):
    try:
        num_nodes = len(all_data)
        min_length = min(len(all_data[key]) for key in all_data)
        print(f"Number of nodes: {num_nodes}, Min length: {min_length}")

        feature_cols = ['Local_X', 'Local_Y', 'Local_Z']
        label_cols = ['Local_X', 'Local_Y', 'Local_Z']

        target_ids = list(range(max(anchor_ids) + 1, num_nodes))  # [4, 5, 6]
        for target_idx in target_ids:
            pos_df = all_data[f'uav{target_idx}'][label_cols]
            print(f"Position std for uav{target_idx}:\n", pos_df.std())
            print(f"Sample positions (uav{target_idx}, first 5 rows):\n", pos_df.head())

        dataset = []

        for t in range(min_length):
            node_features = []
            positions = []

            for i, uav_key in enumerate(sorted(all_data.keys())):
                df = all_data[uav_key]
                row = df.iloc[t]
                positions.append(row[label_cols].values)
                if i in anchor_ids:
                    features = row[feature_cols].values
                else:
                    features = np.zeros(len(feature_cols))
                node_features.append(features)

            x = torch.tensor(np.array(node_features), dtype=torch.float)
            # y contains all targets' positions, flattened
            target_positions = np.concatenate([positions[i] for i in target_ids])
            y = torch.tensor(target_positions, dtype=torch.float)

            edge_index = []
            edge_attr = []
            seen_pairs = set()
            for i, j in combinations(range(num_nodes), 2):
                if (i, j) not in seen_pairs and (j, i) not in seen_pairs:
                    path_loss_ij = all_data[f'uav{i}'][f'Transmitter_{j}_path_loss'].iloc[t]
                    path_loss_ji = all_data[f'uav{j}'][f'Transmitter_{i}_path_loss'].iloc[t]
                    avg_path_loss = np.nanmean([path_loss_ij, path_loss_ji])
                    if not np.isnan(avg_path_loss):
                        edge_index.extend([[i, j]])
                        edge_attr.extend([[avg_path_loss]])
                    seen_pairs.add((i, j))

            edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long) if edge_index else torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float) if edge_attr else torch.empty((0, 1), dtype=torch.float)

            if t < 5:
                edge_dict = {(i, j): edge_attr[idx][0] for idx, (i, j) in enumerate(edge_index.t().numpy())}
                print(f"Time step {t}, edge path losses: {edge_dict}")
                print(f"Number of edges: {len(edge_dict)}")

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.anchor_ids = anchor_ids
            dataset.append(data)

        print(f"Dataset created with {len(dataset)} graphs")
        return dataset

    except Exception as e:
        print(f"Error in create_graph_dataset: {str(e)}")
        return None

dataset_rss = create_graph_dataset_rss_7uav(all_data)
torch.save(dataset_rss, '/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/graph_data_rss.pt')


# # Path loss (RSS) and AoA

# In[8]:


def calculate_ground_truth_angles(pos1, pos2):
    """Calculate true azimuth and elevation from pos1 to pos2."""
    delta = pos2 - pos1
    dist = np.linalg.norm(delta)
    azimuth = np.degrees(np.arctan2(delta[1], delta[0])) % 360
    elevation = np.degrees(np.arcsin(delta[2] / dist))
    return azimuth, elevation

def create_graph_dataset_rss_aoa_7uav(all_data, anchor_ids=[0, 1, 2, 3]):
    try:
        num_nodes = len(all_data)
        min_length = min(len(all_data[key]) for key in all_data)
        print(f"Number of nodes: {num_nodes}, Min length: {min_length}")

        feature_cols = ['Local_X', 'Local_Y', 'Local_Z']
        label_cols = ['Local_X', 'Local_Y', 'Local_Z']

        target_ids = list(range(max(anchor_ids) + 1, num_nodes))  # [4, 5, 6]
        for target_idx in target_ids:
            pos_df = all_data[f'uav{target_idx}'][label_cols]
            print(f"Position std for uav{target_idx}:\n", pos_df.std())
            print(f"Sample positions (uav{target_idx}, first 5 rows):\n", pos_df.head())

        dataset = []

        for t in range(min_length):
            node_features = []
            positions = []

            for i, uav_key in enumerate(sorted(all_data.keys())):
                df = all_data[uav_key]
                row = df.iloc[t]
                pos = row[label_cols].values
                if np.any(np.isnan(pos)):
                    raise ValueError(f"NaN detected in position for {uav_key} at time {t}")
                positions.append(pos)

            aoa_features = []
            for i in range(num_nodes):
                angles = []
                for j in range(num_nodes):
                    if i != j:
                        az, el = calculate_ground_truth_angles(positions[i], positions[j])
                        angles.extend([az, el])
                aoa_features.append(angles)  # 6 other nodes * 2 angles = 12 features

            for i, uav_key in enumerate(sorted(all_data.keys())):
                if i in anchor_ids:
                    features = np.concatenate([positions[i], aoa_features[i]])  # 3 pos + 12 angles
                else:
                    features = np.zeros(3 + 12)  # Zero-pad target
                node_features.append(features)

            x = torch.tensor(np.array(node_features), dtype=torch.float)  # Shape: (7, 15)
            target_positions = np.concatenate([positions[i] for i in target_ids])
            y = torch.tensor(target_positions, dtype=torch.float)

            edge_index = []
            edge_attr = []
            seen_pairs = set()
            for i, j in combinations(range(num_nodes), 2):
                if (i, j) not in seen_pairs and (j, i) not in seen_pairs:
                    path_loss_ij = all_data[f'uav{i}'][f'Transmitter_{j}_path_loss'].iloc[t]
                    path_loss_ji = all_data[f'uav{j}'][f'Transmitter_{i}_path_loss'].iloc[t]
                    avg_path_loss = np.nanmean([path_loss_ij, path_loss_ji])
                    if not np.isnan(avg_path_loss):
                        edge_index.extend([[i, j]])
                        edge_attr.extend([[avg_path_loss]])
                    seen_pairs.add((i, j))

            edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long) if edge_index else torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float) if edge_attr else torch.empty((0, 1), dtype=torch.float)

            if t < 5:
                edge_dict = {(i, j): edge_attr[idx][0] for idx, (i, j) in enumerate(edge_index.t().numpy())}
                print(f"Time step {t}, edge path losses: {edge_dict}")
                print(f"Number of edges: {len(edge_dict)}")

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.anchor_ids = anchor_ids
            dataset.append(data)

        print(f"Dataset created with {len(dataset)} graphs")
        return dataset

    except Exception as e:
        print(f"Error in create_graph_dataset: {str(e)}")
        return None

dataset_rss_aoa = create_graph_dataset_rss_aoa_7uav(all_data)
torch.save(dataset_rss_aoa, '/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/graph_data_rss_aoa.pt')


# In[ ]:





# In[ ]:




