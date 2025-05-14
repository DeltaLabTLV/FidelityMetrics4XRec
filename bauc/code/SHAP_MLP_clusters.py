#!/usr/bin/env python
# coding: utf-8

# # Imports
import pandas as pd
import numpy as np
# import os # Removed, not directly used
import torch
import torch.nn as nn
import shap
from pathlib import Path
import pickle
import warnings
from sklearn.cluster import KMeans # Moved to top

# Specific imports from local modules
from .help_functions import setup_shap_experiment_data # Main setup function
# from .help_functions import * # Avoid star imports if possible, but keep if legacy code relies on it.
# For now, assume specific functions like get_user_recommended_item are brought in by setup_shap_experiment_data's dependencies or are not needed.
# from .recommenders_architecture import MLP # Removed, not essential for revised MLPWrapper
                                        # The revised MLPWrapper does not inherit MLP, but takes an MLP instance.
                                        # So, this import might only be for type hinting if used.

# Configuration
data_name = "ML1M" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "MLP"

# Load data and models using the common setup function
(
    files_path, checkpoints_path, device, 
    kw_dict, train_array, test_array, items_array, pop_array, static_test_data_df,
    recommender_model, top1_train, top1_test, train_data_df, test_data_df, num_items
) = setup_shap_experiment_data(data_name, recommender_name)

# recommender_model IS the trained MLP instance.
# optimizer = torch.optim.Adam(recommender_model.parameters(), lr=0.001) # Removed, unused

# # SHAP

# ### MLP Wrapper for SHAP (Revised to contain the trained model)
class MLPWrapper(nn.Module):
    def __init__(self, trained_mlp_model_instance, cluster_to_items_map, 
                 items_array_global_ref, num_items_global_ref, device_global_ref, kw_dict_ref):
        super(MLPWrapper, self).__init__()
        self.trained_model = trained_mlp_model_instance
        self.cluster_to_items = cluster_to_items_map
        self.items_array_global = items_array_global_ref
        self.device = device_global_ref
        self.num_items = num_items_global_ref
        self.kw_dict = kw_dict_ref

    def preprocess(self, batch_input_np):
        target_item_ids = batch_input_np[:, 0].astype(int)
        cluster_values = batch_input_np[:, 1:]
        num_samples_in_batch = batch_input_np.shape[0]
        n_clusters = cluster_values.shape[1]
        target_items_one_hot = torch.tensor(self.items_array_global[target_item_ids], dtype=torch.float).to(self.device)
        user_history_dense = torch.zeros((num_samples_in_batch, self.num_items), dtype=torch.float).to(self.device)
        for cluster_idx in range(n_clusters):
            if cluster_idx in self.cluster_to_items:
                items_in_cluster = self.cluster_to_items[cluster_idx]
                current_cluster_interactions_for_batch = torch.tensor(cluster_values[:, cluster_idx], dtype=torch.float).to(self.device).unsqueeze(1)
                user_history_dense[:, items_in_cluster] = current_cluster_interactions_for_batch
        return user_history_dense, target_items_one_hot

    def forward(self, batch_of_cluster_inputs_with_target_item_np):
        batch_size_internal = 256  
        outputs_list = []
        num_total_samples = batch_of_cluster_inputs_with_target_item_np.shape[0]
        for i in range(0, num_total_samples, batch_size_internal):
            mini_batch_np = batch_of_cluster_inputs_with_target_item_np[i:i+batch_size_internal]
            user_tensor_dense, target_items_tensor_one_hot = self.preprocess(mini_batch_np)
            output_scores_for_minibatch = self.trained_model(user_tensor_dense, target_items_tensor_one_hot)
            outputs_list.append(output_scores_for_minibatch.detach().cpu().numpy())
        final_outputs_np = np.concatenate(outputs_list)
        return final_outputs_np

# ### Clustering
# K_shap_sampling_bg = 100 # In original MLP script, K was 100 then K=50 for shap.sample. Let's use 50 later.
np.random.seed(3)
# from sklearn.cluster import KMeans # Moved to top
k_clusters = 10
kmeans = KMeans(n_clusters=k_clusters, random_state=3, n_init=10) # Ensure n_init for stability
# Perform clustering on item features (items by user interactions)
# u_train is (num_users, num_items). Transpose to (num_items, num_users) for item clustering.
item_features_for_clustering = np.transpose(train_array) # Use train_array directly
fitted_kmeans = kmeans.fit(item_features_for_clustering)
item_cluster_labels = fitted_kmeans.labels_ # Get cluster labels for each item

item_to_cluster = {item_idx: label for item_idx, label in enumerate(item_cluster_labels)}
cluster_to_items = {label: [] for label in range(k_clusters)}
for item_idx, label in item_to_cluster.items():
    cluster_to_items[label].append(item_idx)

# u_test = torch.tensor(test_array).float() # Removed as test_array is already numpy

# Create user_to_clusters_train_bin (background for SHAP)
user_to_clusters_train = np.zeros((train_array.shape[0], k_clusters))
for cluster_label_idx in range(k_clusters):
    if cluster_label_idx in cluster_to_items and cluster_to_items[cluster_label_idx]: # Check if cluster exists and is not empty
        items_in_this_cluster = cluster_to_items[cluster_label_idx]
        user_to_clusters_train[:, cluster_label_idx] = np.sum(train_array[:, items_in_this_cluster], axis=1) # Use train_array directly
user_to_clusters_train_bin = np.where(user_to_clusters_train > 0, 1, 0)

# Create user_to_clusters_test_bin (data to explain)
user_to_clusters_test = np.zeros((test_array.shape[0], k_clusters))
for cluster_label_idx in range(k_clusters):
    if cluster_label_idx in cluster_to_items and cluster_to_items[cluster_label_idx]: # Check if cluster exists and is not empty
        items_in_this_cluster = cluster_to_items[cluster_label_idx]
        user_to_clusters_test[:, cluster_label_idx] = np.sum(test_array[:, items_in_this_cluster], axis=1) # Use test_array directly
user_to_clusters_test_bin = np.where(user_to_clusters_test > 0, 1, 0)

# Prepare input arrays for SHAP (MLP version prepends target item IDs)
# top1_train and top1_test are dicts: {user_original_id: item_id}, returned by setup
train_user_original_ids = train_data_df.index.tolist() # train_data_df from setup
target_item_ids_for_train_array = [top1_train[uid] for uid in train_user_original_ids]
input_train_for_shap = np.insert(user_to_clusters_train_bin, 0, target_item_ids_for_train_array, axis=1).astype(float)

test_user_original_ids = test_data_df.index.tolist() # test_data_df from setup
target_item_ids_for_test_array = [top1_test[uid] for uid in test_user_original_ids]
input_test_for_shap = np.insert(user_to_clusters_test_bin, 0, target_item_ids_for_test_array, axis=1).astype(float)

# Instantiate the MLPWrapper
wrap_model = MLPWrapper(trained_mlp_model_instance=recommender_model, 
                        cluster_to_items_map=cluster_to_items, 
                        items_array_global_ref=items_array, 
                        num_items_global_ref=num_items, 
                        device_global_ref=device, 
                        kw_dict_ref=kw_dict)

# ### SHAP Execution
K_samples_for_shap_bg = 50 # Original MLP script used K=50 for shap.sample background

# Background data for SHAP explainer (includes the target item ID as the first feature)
sampled_subset_for_explainer = shap.sample(input_train_for_shap, K_samples_for_shap_bg, random_state=42) # Add random_state

# Initialize SHAP KernelExplainer with the wrapper's forward method and the background data
explainer = shap.KernelExplainer(wrap_model.forward, sampled_subset_for_explainer)

# Compute SHAP values for the test set (input_test_for_shap also includes target item ID as first feature)
shap_values_raw_test = explainer.shap_values(input_test_for_shap)

# Process SHAP values for saving
# shap_values_raw_test has shape (num_test_samples, num_features_in_input_test_for_shap)
# num_features_in_input_test_for_shap = 1 (for target_item_id) + k_clusters
# We want to save SHAP values for cluster features only.
if shap_values_raw_test.ndim == 2:
    shap_values_for_cluster_features = shap_values_raw_test[:, 1:] # Exclude SHAP value for the target_item_id feature
else:
    # This case should not happen for KernelExplainer with multiple features if model output is 1D array
    warnings.warn("SHAP values are not 2D as expected. Saving raw SHAP values. Result might be incorrect.")
    shap_values_for_cluster_features = shap_values_raw_test # Fallback, might be wrong shape

# Prepare user IDs for saving (use original user IDs from test_data_df index)
test_user_indices_col = np.array(test_user_original_ids).reshape(-1, 1)

# Combine user IDs with their corresponding SHAP values for cluster features
output_shap_array_to_save = np.hstack((test_user_indices_col, shap_values_for_cluster_features))

# Save item_to_cluster mapping and the processed SHAP values
with open(Path(files_path, f'item_to_cluster_{recommender_name}_{data_name}.pkl'), 'wb') as f:
    pickle.dump(item_to_cluster, f)

with open(Path(files_path, f'shap_values_{recommender_name}_{data_name}.pkl'), 'wb') as f:
    pickle.dump(output_shap_array_to_save, f)

print(f"SHAP values saved for {recommender_name} on {data_name} using MLPWrapper.")

# Cleanup: Remove unused imports from original file if any remain.
# Original imports included:
# from scipy import sparse
# from os import path
# from torch.utils.data import Dataset, DataLoader
# from torch.nn import Linear, ReLU, Sigmoid, Softmax, Module (beyond nn.Module)
# from torch.optim import SGD
# from torch.nn import BCELoss, CrossEntropyLoss
# import torch.nn.functional as F
# These are likely not needed with the current refactored script.
# The import `from .recommenders_architecture import MLP` is also likely not needed if not used for type hint.

