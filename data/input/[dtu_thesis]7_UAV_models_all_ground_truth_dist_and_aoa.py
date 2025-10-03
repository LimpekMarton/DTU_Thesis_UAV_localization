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
import copy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Ada5
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from xgboost import XGBRegressor
from torch_geometric.loader import DataLoader  # Updated import
from torch_geometric.nn import NNConv, global_mean_pool, GINEConv
from egnn_pytorch import EGNN_Network
from egnn_pytorch import EGNN
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use("ggplot")
sns.set_style("whitegrid")


# In[4]:


X = np.load('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/X_ground_truth_dist_and_aoa_data.npy')
y = np.load('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/y_ground_truth_dist_and_aoa_data.npy')
# Split data with validation
X_temp, X_test, y_temp, y_test = train_test_split(X[1000:], y[1000:], test_size=0.2, random_state=42, shuffle=True) #random_state=42,
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True)  # 0.25 x 0.8 = 0.2 validation


# In[5]:


# Scale data
scaler_X = StandardScaler().fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler().fit(y_train)
y_train_scaled = scaler_y.transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)


# In[6]:


joblib.dump(scaler_X, '/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/X_ground_truth_dist_aoa_scaler.joblib')
joblib.dump(scaler_y, '/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/y_ground_truth_dist_aoa_scaler.joblib')


# In[12]:


def add_noise(X, noise_level=0.0):
    # X shape: (num_samples, 117), cols 0-11: anchor positions, 12-32: distances, 33-116: AoA
    num_distance_cols = 21  # 21 pairwise distances for 7 UAVs
    noise = np.zeros(X.shape)
    noise[:, 12:12+num_distance_cols] = np.random.normal(0, noise_level, (X.shape[0], num_distance_cols))  # Noise for distances only
    return X + noise


# Noise levels
noise_levels = np.arange(11)
results = []

for noise_level in noise_levels:
    print(f"Training DNN and RFR with noise level: {noise_level}")

    # Add noise to train, val, test distances, then scale
    X_train_noisy = add_noise(X_train, noise_level)
    X_val_noisy = add_noise(X_val, noise_level)
    X_test_noisy = add_noise(X_test, noise_level)
    X_train_noisy_scaled = scaler_X.transform(X_train_noisy)
    X_val_noisy_scaled = scaler_X.transform(X_val_noisy)
    X_test_noisy_scaled = scaler_X.transform(X_test_noisy)

    # DNN
    model_dnn = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(9)  # 3 targets * 3 coords
    ])
    model_dnn.compile(optimizer='adam', loss='mse')
    model_dnn.fit(X_train_noisy_scaled, y_train_scaled, epochs=500, batch_size=32, verbose=0,
                  validation_data=(X_val_noisy_scaled, y_val_scaled))

    y_pred_dnn_scaled = model_dnn.predict(X_test_noisy_scaled)
    y_pred_dnn = scaler_y.inverse_transform(y_pred_dnn_scaled)
    np.save(f'/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/ground_truth_dist_aoa_DNN_y_pred_noise_{noise_level}.npy', y_pred_dnn)
    np.save(f'/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/ground_truth_dist_aoa_DNN_y_test_noise_{noise_level}.npy', y_test)

    # RFR
    model_rfr = RandomForestRegressor(n_estimators=200, random_state=42)
    model_rfr.fit(X_train_noisy_scaled, y_train_scaled)

    y_pred_rfr_scaled = model_rfr.predict(X_test_noisy_scaled)
    y_pred_rfr = scaler_y.inverse_transform(y_pred_rfr_scaled)
    np.save(f'/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/ground_truth_dist_aoa_RFR_y_pred_noise_{noise_level}.npy', y_pred_rfr)
    np.save(f'/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/ground_truth_dist_aoa_RFR_y_test_noise_{noise_level}.npy', y_test)

    # Calculate for DNN
    errors_dnn = np.zeros((y_test.shape[0], 3))
    for t in range(3):
        errors_dnn[:, t] = np.sqrt(np.sum((y_pred_dnn[:, t*3:(t+1)*3] - y_test[:, t*3:(t+1)*3]) ** 2, axis=1))
    variances_dnn = np.var(errors_dnn, axis=0)
    avg_variance_dnn = np.mean(variances_dnn)
    mean_euclidean_error_dnn = np.mean(errors_dnn)

    # Calculate for RFR
    errors_rfr = np.zeros((y_test.shape[0], 3))
    for t in range(3):
        errors_rfr[:, t] = np.sqrt(np.sum((y_pred_rfr[:, t*3:(t+1)*3] - y_test[:, t*3:(t+1)*3]) ** 2, axis=1))
    variances_rfr = np.var(errors_rfr, axis=0)
    avg_variance_rfr = np.mean(variances_rfr)
    mean_euclidean_error_rfr = np.mean(errors_rfr)

    results.append({
        'Noise_Level': noise_level,
        'Mean_Euclidean_Error_DNN': mean_euclidean_error_dnn,
        'Avg_Error_Variance_DNN': avg_variance_dnn,
        'Mean_Euclidean_Error_RFR': mean_euclidean_error_rfr,
        'Avg_Error_Variance_RFR': avg_variance_rfr
    })


# Create DataFrame
results_df = pd.DataFrame(results)
print("DNN and RFR Results:\n", results_df)
# Save DataFrame
results_df.to_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/ground_truth_dist_aoa_results.csv', index=False)


# In[14]:


def plot_dnn_rfr_results_7uav(results_df, save_dir='/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/pictures/ML_methods/7UAV/dist_aoa'):
    sns.set_style("whitegrid")


    palette = sns.color_palette("tab10", n_colors=2)

    # Plot Mean Euclidean Error vs Noise Level
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Noise_Level'], results_df['Mean_Euclidean_Error_DNN'], label='DNN', marker='o', color=palette[0])
    plt.plot(results_df['Noise_Level'], results_df['Mean_Euclidean_Error_RFR'], label='RFR', marker='s', color=palette[1])
    plt.xlabel('Noise Level [m]')
    plt.ylabel('Mean Euclidean Error [m]')
    plt.title('Mean Euclidean Error vs Noise Level (7 UAVs, Distances)')
    plt.legend()
    plt.savefig(f'{save_dir}/dist_dnn_rfr_mean_euclidean_error.png')
    plt.show()
    plt.close()

    # Plot Average Error Variance vs Noise Level
    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Noise_Level'], results_df['Avg_Error_Variance_DNN'], label='DNN', marker='o', color=palette[0])
    plt.plot(results_df['Noise_Level'], results_df['Avg_Error_Variance_RFR'], label='RFR', marker='s', color=palette[1])
    plt.xlabel('Noise Level [m]')
    plt.ylabel('Average Error Variance [m^2]')
    plt.title('Average Error Variance vs Noise Level (7 UAVs, Distances)')
    plt.legend()
    plt.savefig(f'{save_dir}/dist_dnn_rfr_avg_error_variance.png')
    plt.show()
    plt.close()


results_df = pd.read_csv('/Users/marton/DTU_MSC/IV_SEMESTER/Thesis/data/5GHz_7UAV/ground_truth_dist_aoa_results.csv')


plot_dnn_rfr_results_7uav(results_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




