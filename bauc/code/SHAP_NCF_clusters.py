#!/usr/bin/env python
# coding: utf-8

# # Imports


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
from pathlib import Path
import pickle
import warnings
from sklearn.cluster import KMeans

# Specific imports from local modules
from .help_functions import setup_shap_experiment_data
# from .recommenders_architecture import NCF, MLP_model, GMF_model # Not strictly needed

# Configuration
data_name = "Yahoo" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "NCF" 

# Load data and models using the common setup function
(
    files_path, checkpoints_path, device, 
    kw_dict, train_array, test_array, items_array, pop_array, static_test_data_df,
    recommender_model, top1_train, top1_test, train_data_df, test_data_df, num_items
) = setup_shap_experiment_data(data_name, recommender_name)

# recommender_model IS the trained NCF instance.

# # SHAP
# ### NCF Wrapper for SHAP (Definition is already correct from a previous edit)
class NCFWrapper(nn.Module):
    def __init__(self, trained_ncf_model_instance, cluster_to_items_map, 
                 items_array_global_ref, num_items_global_ref, device_global_ref, kw_dict_ref):
        super(NCFWrapper, self).__init__()
        self.trained_model = trained_ncf_model_instance # Instance of recommenders_architecture.NCF
        self.cluster_to_items = cluster_to_items_map
        # items_array_global_ref is the one-hot encoded full item catalog (num_items, num_items)
        self.items_array_global_for_one_hot = torch.tensor(items_array_global_ref, dtype=torch.float).to(device_global_ref)
        self.device = device_global_ref
        self.num_items = num_items_global_ref
        self.kw_dict = kw_dict_ref

    def preprocess(self, batch_input_np): # Expects NumPy array from SHAP
        # batch_input_np has shape (num_samples_in_batch, 1_for_target_item_id + k_clusters)
        target_item_ids = batch_input_np[:, 0].astype(int)
        cluster_values = batch_input_np[:, 1:] # Shape: (num_samples_in_batch, k_clusters)
        num_samples_in_batch = batch_input_np.shape[0]
        n_clusters = cluster_values.shape[1]

        # target_item_vectors for NCF: (num_samples_in_batch, num_items) - one-hot encoded from item IDs
        target_item_vectors_one_hot = self.items_array_global_for_one_hot[target_item_ids]
        
        # user_history_dense: (num_samples_in_batch, num_items)
        user_history_dense = torch.zeros((num_samples_in_batch, self.num_items), dtype=torch.float).to(self.device)
        for cluster_idx in range(n_clusters):
            if cluster_idx in self.cluster_to_items:
                items_in_cluster = self.cluster_to_items[cluster_idx]
                current_cluster_interactions_for_batch = torch.tensor(cluster_values[:, cluster_idx], dtype=torch.float).to(self.device).unsqueeze(1)
                user_history_dense[:, items_in_cluster] = current_cluster_interactions_for_batch
        return user_history_dense, target_item_vectors_one_hot

    def forward(self, batch_of_cluster_inputs_with_target_item_np): # Expects NumPy array from SHAP
        batch_size_internal = 256
        outputs_list = []
        num_total_samples = batch_of_cluster_inputs_with_target_item_np.shape[0]
        for i in range(0, num_total_samples, batch_size_internal):
            mini_batch_np = batch_of_cluster_inputs_with_target_item_np[i:i+batch_size_internal]
            user_tensor_dense, target_item_vectors = self.preprocess(mini_batch_np)
            output_scores_for_minibatch = self.trained_model(user_tensor_dense, target_item_vectors)
            outputs_list.append(output_scores_for_minibatch.detach().cpu().numpy())
        final_outputs_np = np.concatenate(outputs_list)
        return final_outputs_np

# ### Clustering (Standardized)
np.random.seed(3)
k_clusters = 10
kmeans = KMeans(n_clusters=k_clusters, random_state=3, n_init=10)
item_features_for_clustering = np.transpose(train_array)
fitted_kmeans = kmeans.fit(item_features_for_clustering)
item_cluster_labels = fitted_kmeans.labels_
item_to_cluster = {item_idx: label for item_idx, label in enumerate(item_cluster_labels)}
cluster_to_items = {label: [] for label in range(k_clusters)}
for item_idx, label in item_to_cluster.items():
    cluster_to_items[label].append(item_idx)

user_to_clusters_train = np.zeros((train_array.shape[0], k_clusters))
for cluster_label_idx in range(k_clusters):
    if cluster_label_idx in cluster_to_items and cluster_to_items[cluster_label_idx]:
        items_in_this_cluster = cluster_to_items[cluster_label_idx]
        user_to_clusters_train[:, cluster_label_idx] = np.sum(train_array[:, items_in_this_cluster], axis=1)
user_to_clusters_train_bin = np.where(user_to_clusters_train > 0, 1, 0)

user_to_clusters_test = np.zeros((test_array.shape[0], k_clusters))
for cluster_label_idx in range(k_clusters):
    if cluster_label_idx in cluster_to_items and cluster_to_items[cluster_label_idx]:
        items_in_this_cluster = cluster_to_items[cluster_label_idx]
        user_to_clusters_test[:, cluster_label_idx] = np.sum(test_array[:, items_in_this_cluster], axis=1)
user_to_clusters_test_bin = np.where(user_to_clusters_test > 0, 1, 0)

# Prepare input arrays for SHAP (NCF version also prepends target item IDs)
train_user_original_ids = train_data_df.index.tolist()
target_item_ids_for_train_array = [top1_train[uid] for uid in train_user_original_ids]
input_train_for_shap = np.insert(user_to_clusters_train_bin, 0, target_item_ids_for_train_array, axis=1).astype(float)

test_user_original_ids = test_data_df.index.tolist()
target_item_ids_for_test_array = [top1_test[uid] for uid in test_user_original_ids]
input_test_for_shap = np.insert(user_to_clusters_test_bin, 0, target_item_ids_for_test_array, axis=1).astype(float)

# Instantiate the NCFWrapper
wrap_model = NCFWrapper(trained_ncf_model_instance=recommender_model, 
                        cluster_to_items_map=cluster_to_items, 
                        items_array_global_ref=items_array, 
                        num_items_global_ref=num_items, 
                        device_global_ref=device, 
                        kw_dict_ref=kw_dict)

# ### SHAP Execution
K_samples_for_shap_bg = 50
sampled_subset_for_explainer = shap.sample(input_train_for_shap, K_samples_for_shap_bg, random_state=42) # Added random_state
explainer = shap.KernelExplainer(wrap_model.forward, sampled_subset_for_explainer)
shap_values_raw_test = explainer.shap_values(input_test_for_shap)

# Process SHAP values for saving
if shap_values_raw_test.ndim == 2:
    shap_values_for_cluster_features = shap_values_raw_test[:, 1:] # Exclude SHAP value for target_item_id feature
else:
    warnings.warn("SHAP values are not 2D as expected for NCF. Saving raw SHAP values.")
    shap_values_for_cluster_features = shap_values_raw_test # Fallback

test_user_indices_col = np.array(test_user_original_ids).reshape(-1, 1)
output_shap_array_to_save = np.hstack((test_user_indices_col, shap_values_for_cluster_features))

# Save results
with open(Path(files_path, f'item_to_cluster_{recommender_name}_{data_name}.pkl'), 'wb') as f:
    pickle.dump(item_to_cluster, f)

with open(Path(files_path, f'shap_values_{recommender_name}_{data_name}.pkl'), 'wb') as f:
    pickle.dump(output_shap_array_to_save, f)

print(f"SHAP values saved for {recommender_name} on {data_name} using NCFWrapper.")

