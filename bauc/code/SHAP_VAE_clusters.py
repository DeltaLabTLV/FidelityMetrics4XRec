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

# from scipy import sparse # This line will be removed
# from os import path # This line will be removed

# from torch.utils.data import Dataset # This line will be removed
# from torch.utils.data import DataLoader # This line will be removed
# from torch.nn import Linear # This line will be removed
# from torch.nn import ReLU # This line will be removed
# from torch.nn import Sigmoid # This line will be removed
# from torch.nn import Module # This line will be removed
# from torch.optim import SGD # This line will be removed
# from torch.nn import BCELoss # This line will be removed
# from torch.nn import CrossEntropyLoss # This line will be removed
# import torch.nn.functional as F # This line will be removed

# from sklearn.decomposition import NMF # This line will be removed
# from sklearn.preprocessing import LabelEncoder # This line will be removed
# from sklearn.model_selection import train_test_split # This line will be removed
# from scipy import sparse # This line will be removed (duplicate)

# from torch.nn import Softmax # This line will be removed (duplicate)
softmax = nn.Softmax()


# In[2]:


# Import specific functions from .help_functions
from .help_functions import setup_shap_experiment_data, recommender_run 

data_name = "Yahoo" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "VAE"

# Call setup_shap_experiment_data once to get all necessary variables
(
    files_path, checkpoints_path, device, 
    kw_dict, train_array, test_array, items_array, pop_array, static_test_data_df,
    recommender_model, top1_train, top1_test, train_data_df, test_data_df, num_items
) = setup_shap_experiment_data(data_name, recommender_name)


# In[3]:


# output_type_dict = {
#     "VAE":"multiple",
#     "MLP":"single",
#     "NCF": "single"}
# 
# num_users_dict = {
#     "ML1M":6037,
#     "Yahoo":13797, 
#     "Pinterest":19155}
# 
# num_items_dict = {
#     "ML1M":3381,
#     "Yahoo":4604, 
#     "Pinterest":9362}
# 
# 
# recommender_path_dict = {
#     ("ML1M","VAE"):Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
#     ("ML1M","MLP"):Path(checkpoints_path, "MLP1_ML1M_0.0076_256_7.pt"),
#     ("ML1M","NCF"):Path(checkpoints_path, "NCF_ML1M_5e-05_64_16.pt"),
#     
#     ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
#     ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),
#     ("Yahoo","NCF"):Path(checkpoints_path, "NCF_Yahoo_0.001_64_21_0.pt"),
#     
#     ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_12_18_0.0001_256.pt"),
#     ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt"),
#     ("Pinterest","NCF"):Path(checkpoints_path, "NCF2_Pinterest_9e-05_32_9_10.pt"),}
# 
# 
# hidden_dim_dict = {
#     ("ML1M","VAE"): None,
#     ("ML1M","MLP"): 32,
#     ("ML1M","NCF"): 8,
# 
#     ("Yahoo","VAE"): None,
#     ("Yahoo","MLP"):32,
#     ("Yahoo","NCF"):8,
#     
#     ("Pinterest","VAE"): None,
#     ("Pinterest","MLP"):512,
#     ("Pinterest","NCF"): 64,
# }


# In[4]:


# from .help_functions import *


# # SHAP

# ## VAE wrapper for shap

# In[6]:


class WrapperModel(nn.Module):
    def __init__(self, model, item_array, cluster_to_items, item_to_cluster, num_items_arg, device_arg, num_clusters=10):
        super(WrapperModel, self).__init__()
        self.model = model
        self.n_items = num_items_arg
        self.cluster_to_items = cluster_to_items
        self.item_to_cluster = item_to_cluster
        self.item_array = item_array
        self.device = device_arg
        self.n_clusters = num_clusters
    
    def forward(self, input_array):
        batch_size = input_array.shape[0]  # Get the batch size (number of users)
        user_vector_batch = torch.zeros(batch_size, self.n_items).to(self.device)

        for cluster_idx in range(self.n_clusters):
            if cluster_idx in self.cluster_to_items:
                cluster_indices = self.cluster_to_items[cluster_idx]
                if cluster_idx < input_array.shape[1]:
                    user_vector_batch[:, cluster_indices] = torch.from_numpy(input_array[:, cluster_idx]).unsqueeze(1).float().to(self.device)
                else:
                    warnings.warn(f"Cluster index {cluster_idx} is out of bounds for input_array with shape {input_array.shape}. Skipping.")
            # If cluster_idx is not in cluster_to_items, those items remain 0 (or default value)

        model_output_batch = self.model(user_vector_batch)
        
        if model_output_batch.dim() == 1:
            softmax_output_batch = torch.softmax(model_output_batch, dim=0)
        elif model_output_batch.dim() == 2:
            softmax_output_batch = torch.softmax(model_output_batch, dim=1)
        else:
            raise ValueError("Unexpected number of dimensions in model_output_batch")

        cluster_scores_per_user = []

        for user_idx in range(batch_size):
            user_cluster_scores = []
            for cluster_idx_inner, items_in_cluster in self.cluster_to_items.items():
                valid_items_in_cluster = [item for item in items_in_cluster if item < softmax_output_batch.shape[1]]
                if not valid_items_in_cluster:
                    user_cluster_scores.append(torch.tensor(0.0, device=self.device))
                    continue
                
                cluster_scores = softmax_output_batch[user_idx, valid_items_in_cluster]
                avg_score = torch.mean(cluster_scores)
                user_cluster_scores.append(avg_score)
            cluster_scores_per_user.append(torch.stack(user_cluster_scores))

        cluster_scores_per_user = torch.stack(cluster_scores_per_user)

        return cluster_scores_per_user.cpu().detach().numpy()
        
    def predict(self, x):
        x = torch.Tensor(x).to(self.device)
        output = self.forward(x)
        output = torch.Tensor(output).to(self.device)
        return output.cpu().detach().numpy()


# ### Read data

# In[7]:


# train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
# test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
# train_array = train_data.to_numpy()
# test_array = test_data.to_numpy()


# ## Create / Load top recommended item dict 

# In[8]:


# with open(Path(files_path, f'pop_dict_{data_name}.pkl'), 'rb') as f:
#     pop_dict = pickle.load(f)
# pop_array = np.zeros(len(pop_dict))
# for key, value in pop_dict.items():
#     pop_array[key] = value
# 
# static_test_data = pd.read_csv(Path(files_path, f'static_test_data_{data_name}.csv'), index_col=0)
# 
# output_type = output_type_dict[recommender_name]
# num_users = num_users_dict[data_name]
# num_items = num_items_dict[data_name]
# 
# items_array = np.eye(num_items)
# all_items_tensor = torch.Tensor(items_array).to(device)
# 
# hidden_dim = hidden_dim_dict[(data_name, recommender_name)]
# recommender_path = recommender_path_dict[(data_name, recommender_name)]
# 
# kw_dict = {
#     'device': device,
#     'num_items': num_items,
#     'num_features': num_items,
#     'demographic': False,
#     'pop_array': pop_array,
#     'all_items_tensor': all_items_tensor,
#     'static_test_data': static_test_data,
#     'items_array': items_array,
#     'output_type': output_type,
#     'recommender_name': recommender_name
# }
# 
# # Call the centralized load_recommender from help_functions.py
# recommender = load_recommender(data_name, hidden_dim, checkpoints_path, recommender_path, **kw_dict)
# 
# create_dicts = True
# if create_dicts:
#     top1_train = {}
#     top1_test = {}  
#     for i in range(train_array.shape[0]):
#         user_index = int(train_data.index[i])
#         user_tensor = torch.Tensor(train_array[i]).to(device)
#         top1_train[user_index] = int(get_user_recommended_item(user_tensor, recommender, **kw_dict))
#     for i in range(test_array.shape[0]):
#         user_index = int(test_data.index[i])
#         user_tensor = torch.Tensor(test_array[i]).to(device)
#         top1_test[user_index] = int(get_user_recommended_item(user_tensor, recommender, **kw_dict))
#         
#     with open(Path(files_path, f'top1_train_{data_name}_{recommender_name}.pkl'), 'wb') as f:
#         pickle.dump(top1_train, f)
#     with open(Path(files_path, f'top1_test_{data_name}_{recommender_name}.pkl'), 'wb') as f:
#         pickle.dump(top1_test, f)
# else:
#     with open(Path(files_path, f'top1_train_{data_name}_{recommender_name}.pkl'), 'rb') as f:
#         top1_train = pickle.load(f)
#     with open(Path(files_path, f'top1_test_{data_name}_{recommender_name}.pkl'), 'rb') as f:
#         top1_test = pickle.load(f)


# In[12]:


model = recommender
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ### Clustering

# In[13]:


K_shap = 100
u_train = torch.tensor(train_array).float()
v_train = all_items_tensor
user_ids = np.arange(train_array.shape[0])


# In[14]:


np.random.seed(3)
# Cluster items using k-means
from sklearn.cluster import KMeans
import numpy as np
k_clusters = 10

kmeans = KMeans(n_clusters=k_clusters, random_state=3, n_init=10)
clusters = kmeans.fit_predict(np.transpose(u_train.cpu().numpy()))


# In[15]:


item_clusters = kmeans.predict(np.transpose(u_train.cpu().numpy()))

# Create mapping from items to clusters
item_to_cluster = {}
# Create mapping from clusters to items
cluster_to_items = {}
for i, cluster_val in enumerate(item_clusters):
    item_to_cluster[i] = cluster_val
    if cluster_val not in cluster_to_items:
        cluster_to_items[cluster_val] = []
    cluster_to_items[cluster_val].append(i)


# In[16]:


u_test = torch.tensor(test_array).float()


# In[17]:


user_to_clusters = np.zeros((u_test.shape[0], k_clusters))


# In[18]:


for i in range(k_clusters):
    if i in cluster_to_items:
        user_to_clusters[:,i] = np.sum(u_test.cpu().numpy()[:, cluster_to_items[i]], axis=1)


# In[19]:


user_to_clusters_bin =  np.where(user_to_clusters > 0, 1, 0)


# In[20]:


user_to_clusters_train = np.zeros((u_train.shape[0], k_clusters))


# In[21]:


user_to_clusters_test = np.zeros((u_test.shape[0], k_clusters))


# In[22]:


default_value = 0
target_items_test = list(top1_test.values())
target_items_train = list(top1_train.values())


# In[23]:


for i in range(k_clusters):
    if i in cluster_to_items:
        user_to_clusters_train[:,i] = np.sum(u_train.cpu().numpy()[:, cluster_to_items[i]], axis=1)


# In[24]:


user_to_clusters_train_bin = np.where(user_to_clusters_train > 0, 1, 0)


# In[25]:


col2 = list(top1_train.values())
input_train_array = np.insert(user_to_clusters_train_bin, 0, col2, axis=1).astype(int)


# In[26]:


for i in range(k_clusters):
    if i in cluster_to_items:
        user_to_clusters_test[:,i] = np.sum(u_test.cpu().numpy()[:, cluster_to_items[i]], axis=1)


# In[27]:


user_to_clusters_test_bin = np.where(user_to_clusters_test > 0, 1, 0)


# In[28]:


col2 = list(top1_test.values())
input_test_array= np.insert(user_to_clusters_test_bin, 0, col2, axis=1).astype(int)


# In[29]:


wrap_model = WrapperModel(model, items_array, cluster_to_items, item_to_cluster, num_items, device, num_clusters=k_clusters)


# ### SHAP

# In[30]:


K_samples_for_shap = 50


# In[31]:


sampled_subset = shap.sample(user_to_clusters_train_bin, K_samples_for_shap)


# In[32]:


explainer = shap.KernelExplainer(wrap_model.forward, sampled_subset)


# In[ ]:


shap_values_test = explainer.shap_values(user_to_clusters_bin)


# In[ ]:


average_shap = np.mean(np.abs(shap_values_test), axis=0)


# In[ ]:


test_user_indices = test_data_df.index.to_numpy()

if shap_values_test.ndim == 1:
    shap_values_to_save = shap_values_test.reshape(-1, 1)
else:
    shap_values_to_save = shap_values_test

if test_user_indices.ndim == 1:
    test_user_indices_col = test_user_indices.reshape(-1, 1)
else:
    test_user_indices_col = test_user_indices

output_shap_array = np.hstack((test_user_indices_col, shap_values_to_save))


# In[ ]:


with open(Path(files_path,f'item_to_cluster_{recommender_name}_{data_name}.pkl'), 'wb') as f:
    pickle.dump(item_to_cluster, f)


# In[ ]:


with open(Path(files_path,f'shap_values_{recommender_name}_{data_name}.pkl'), 'wb') as f:
    pickle.dump(output_shap_array, f)

