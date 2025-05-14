#!/usr/bin/env python
# coding: utf-8

# ### This notebook produces the metrics for a specific recommendation system and dataset for all the baselines.
# # Imports
# 


import pandas as pd
import numpy as np
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Keep commented
# export_dir = os.getcwd() # Already removed, ensure it stays that way
from pathlib import Path
import pickle
from collections import defaultdict
import time
import torch
# import torch.nn as nn # Removed, not directly used after Explainer class removal
# import copy # Removed, unused
# import torch.nn.functional as F # Removed, unused
# import optuna # Removed, unused
# import logging # Removed, unused
# import matplotlib.pyplot as plt # Removed unused import
# import ipynb # Removed
# import sys # Removed, unused


# In[2]:


data_name = "Yahoo" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "NCF" ### Can be MLP, VAE, NCF

DP_DIR = Path("processed_data", data_name) 
export_dir = Path(os.getcwd()) # Keep: Local variable for path construction
files_path = Path(export_dir.parent, DP_DIR) # Keep: Uses local export_dir
checkpoints_path = Path(export_dir.parent, "checkpoints") # Keep: Uses local export_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


output_type_dict = {
    "VAE":"multiple",
    "MLP":"single",
    "NCF": "single"}

num_users_dict = {
    "ML1M":6037,
    "Yahoo":13797, 
    "Pinterest":19155}

num_items_dict = {
    "ML1M":3381,
    "Yahoo":4604, 
    "Pinterest":9362}


recommender_path_dict = {
    ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","MLP"):Path(checkpoints_path, "MLP_ML1M_0.002_1024_19_8.pt"),
    ("ML1M","NCF"):Path(checkpoints_path, "NCF_ML1M_5e-05_64_16.pt"),
    
    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),
    ("Yahoo","NCF"):Path(checkpoints_path, "NCF_Yahoo_0.001_64_21_0.pt"),
    
    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_12_18_0.0001_256.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt"),
    ("Pinterest","NCF"):Path(checkpoints_path, "NCF2_Pinterest_9e-05_32_9_10.pt"),}


hidden_dim_dict = {
    ("ML1M","VAE"): None,
    ("ML1M","MLP"): 512,
    ("ML1M","NCF"): 8,

    ("Yahoo","VAE"): None,
    ("Yahoo","MLP"):32,
    ("Yahoo","NCF"):8,
    
    ("Pinterest","VAE"): None,
    ("Pinterest","MLP"):512,
    ("Pinterest","NCF"): 64,
}

# LXR_checkpoint_dict = { # REMOVED - This is now centralized in help_functions.py
#     ("ML1M","VAE"): ('LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426.pt',128),
#     ("ML1M","MLP"): ('LXR_ML1M_MLP_19_3_128_13.109692424872248_7.829643365925428.pt',128),
#     ("ML1M","NCF"): ('LXR_ML1M_NCF_17_38_64_14.950042796023537_0.1778309603009678.pt',64),
# 
#     ("Yahoo","VAE"): ('LXR_Yahoo_VAE_neg-1.5pos_combined_19_26_128_18.958765029913238_4.92235962483309.pt',128),
#     ("Yahoo","MLP"):('LXR_Yahoo_MLP_neg-pos_combined_last_29_37_128_12.40692505393434_0.19367009952856118.pt',128),
#     ("Yahoo","NCF"): ('LXR_Yahoo_NCF_neg-pos_combined_loss_14_14_32_16.01464392466348_6.880015038643981.pt', 32),
# 
#     ("Pinterest","VAE"): ('LXR_Pinterest_VAE_comb_4_27_32_6.3443735346179855_1.472868807603448.pt',32),
#     ("Pinterest","MLP"): ('LXR_Pinterest_MLP_0_5_16_10.059416809308486_0.705778173474644.pt',16),
#     ("Pinterest","NCF"): ('LXR_Pinterest_NCF_combined__neg-1.5pos_0_26_32_13.02585523498726_12.8447247971534.pt', 32)
# }


# In[4]:


output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 
num_features = num_items_dict[data_name]

hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
recommender_path = recommender_path_dict[(data_name,recommender_name)]

# lxr_path = LXR_checkpoint_dict[(data_name,recommender_name)][0] # REMOVED - Not needed, uses centralized dict via load_lxr_explainer
# lxr_dim = LXR_checkpoint_dict[(data_name,recommender_name)][1] # REMOVED - Not needed


# ## Data and baselines imports

# In[5]:


train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)


# In[6]:


test_array = static_test_data.iloc[:,:-2].to_numpy()


# In[7]:


with open(Path(files_path, f'jaccard_based_sim_{data_name}.pkl'), 'rb') as f:
    jaccard_dict = pickle.load(f) 


# In[8]:


with open(Path(files_path, f'cosine_based_sim_{data_name}.pkl'), 'rb') as f:
    cosine_dict = pickle.load(f) 


# In[9]:


with open(Path(files_path, f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f) 


# In[10]:


with open(Path(files_path, f'item_to_cluster_{recommender_name}_{data_name}.pkl'), 'rb') as f:
    item_to_cluster = pickle.load(f) 


# In[11]:


with open(Path(files_path, f'shap_values_{recommender_name}_{data_name}.pkl'), 'rb') as f:
    shap_values= pickle.load(f) 


# In[12]:


for i in range(num_items):
    for j in range(i, num_items):
        jaccard_dict[(j,i)]= jaccard_dict[(i,j)]
        cosine_dict[(j,i)]= cosine_dict[(i,j)]


# In[13]:


pop_array = np.zeros(len(pop_dict))
for key, value in pop_dict.items():
    pop_array[key] = value


# In[14]:


kw_dict = {'device':device,
          'num_items': num_items,
           'num_features': num_items, 
            'demographic':False,
          'pop_array':pop_array,
          'all_items_tensor':all_items_tensor,
          'static_test_data':static_test_data,
          'items_array':items_array,
          'output_type':output_type,
          'recommender_name':recommender_name}


# # Configurations

# In[15]:


# sys.path.append('../baselines') # Keep commented

# Import new LIME explanation functions and necessary kernels if not using defaults
from .lime import explain_instance_lime, explain_instance_lire, distance_to_proximity # distance_to_proximity is default in new funcs
# from .lime import LimeBase # REMOVE - No longer need to import LimeBase directly here
# lime = LimeBase(distance_to_proximity) # REMOVE - global LIME instance

from .help_functions import (
    load_recommender, get_user_recommended_item, recommender_run, 
    get_top_k, get_index_in_the_list, get_ndcg, 
    load_lxr_explainer, find_lxr_mask
)
# from ipynb.fs.defs.help_functions import * # Remove
# importlib.reload(ipynb.fs.defs.help_functions) # Remove


# In[16]:


# Call the centralized one from help_functions
recommender = load_recommender(data_name, hidden_dim, checkpoints_path, recommender_path, **kw_dict)


# In[17]:


# class Explainer(nn.Module): # REMOVE local Explainer class
#     def __init__(self, user_size, item_size, hidden_size):
#         super(Explainer, self).__init__()
        
#         self.users_fc = nn.Linear(in_features = user_size, out_features=hidden_size).to(device)
#         self.items_fc = nn.Linear(in_features = item_size, out_features=hidden_size).to(device)
#         self.bottleneck = nn.Sequential(
#             nn.Tanh(),
#             nn.Linear(in_features = hidden_size*2, out_features=hidden_size).to(device),
#             nn.Tanh(),
#             nn.Linear(in_features = hidden_size, out_features=user_size).to(device),
#             nn.Sigmoid()
#         ).to(device)
        
        
#     def forward(self, user_tensor, item_tensor):
#         user_output = self.users_fc(user_tensor.float())
#         item_output = self.items_fc(item_tensor.float())
#         combined_output = torch.cat((user_output, item_output), dim=-1)
#         expl_scores = self.bottleneck(combined_output).to(device)

#         return expl_scores


# In[18]:


# def load_explainer(fine_tuning=False, lambda_pos=None, lambda_neg=None, alpha=None): # REMOVE local load_explainer
#     explainer = Explainer(num_features, num_items, lxr_dim)
#     lxr_checkpoint = torch.load(Path(checkpoints_path, lxr_path))
#     explainer.load_state_dict(lxr_checkpoint)
#     explainer.eval()
#     for param in explainer.parameters():
#         param.requires_grad= False
#     return explainer


# # Baselines functions
# ### Every function produces explanations for a designated baseline, resulting in a dictionary that maps items from the user's history to their explanation scores based on that baseline.

# In[19]:


#popularity mask
def find_pop_mask(x, item_id):
    user_hist = torch.Tensor(x).to(device) # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    item_pop_dict = {}
    
    for i,j in enumerate(user_hist>0):
        if j:
            item_pop_dict[i]=pop_array[i] # add the pop of the item to the dictionary
            
    return item_pop_dict


# In[20]:


#User based similarities using Jaccard
def find_jaccard_mask(x, item_id, user_based_Jaccard_sim):
    user_hist = x # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    item_jaccard_dict = {}
    for i,j in enumerate(user_hist>0):
        if j:
            if (i,item_id) in user_based_Jaccard_sim:
                item_jaccard_dict[i]=user_based_Jaccard_sim[(i,item_id)] # add Jaccard similarity between items
            else:
                item_jaccard_dict[i] = 0            

    return item_jaccard_dict


# In[21]:


#Cosine based similarities between users and items
def find_cosine_mask(x, item_id, item_cosine):
    user_hist = x # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    item_cosine_dict = {}
    for i,j in enumerate(user_hist>0):
        if j:
            if (i,item_id) in item_cosine:
                item_cosine_dict[i]=item_cosine[(i,item_id)]
            else:
                item_cosine_dict[i]=0

    return item_cosine_dict


# In[22]:


# def find_lxr_mask(x, item_tensor, explainer): # REMOVE local find_lxr_mask
    
#     user_hist = x
#     expl_scores = explainer(user_hist, item_tensor)
#     x_masked = user_hist*expl_scores
#     item_sim_dict = {}
#     for i,j in enumerate(x_masked!=0):
#         if j:
#             item_sim_dict[i]=x_masked[i] 
        
#     return item_sim_dict


# In[23]:


def find_fia_mask(user_tensor, item_tensor, item_id, recommender):
    y_pred = recommender_run(user_tensor, recommender, item_tensor, item_id, **kw_dict).to(device)
    items_fia = {}
    user_hist = user_tensor.cpu().detach().numpy().astype(int)
    
    for i in range(num_items):
        if(user_hist[i] == 1):
            user_hist[i] = 0
            user_tensor = torch.FloatTensor(user_hist).to(device)
            y_pred_without_item = recommender_run(user_tensor, recommender, item_tensor, item_id, 'single', **kw_dict).to(device)
            infl_score = y_pred - y_pred_without_item
            items_fia[i] = infl_score
            user_hist[i] = 1

    return items_fia


# In[24]:


def find_shapley_mask(user_tensor, user_id, model, shap_values, item_to_cluster):
    item_shap = {}
    shapley_values = shap_values[shap_values[:, 0].astype(int) == user_id][:,1:]
    user_vector = user_tensor.cpu().detach().numpy().astype(int)

    for i in np.where(user_vector.astype(int) == 1)[0]:
        items_cluster = item_to_cluster[i]
        item_shap[i] = shapley_values.T[int(items_cluster)][0]

    return item_shap  


# In[25]:


def find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model, top_k):
   
    items_accent = defaultdict(float)
    factor = top_k - 1
    user_accent_hist = user_tensor.cpu().detach().numpy().astype(int)

    #Get topk items
    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender_model, **kw_dict).keys())
    
    if top_k == 1:
        # When k=1, return the index of the first maximum value
        top_k_indices = [sorted_indices[0]]
    else:
        top_k_indices = sorted_indices[:top_k]
   

    for iteration, item_k_id in enumerate(top_k_indices):

        # Set topk items to 0 in the user's history
        user_accent_hist[item_k_id] = 0
        user_tensor = torch.FloatTensor(user_accent_hist).to(device)
       
        item_vector = items_array[item_k_id]
        item_tensor = torch.FloatTensor(item_vector).to(device)
              
        # Check influence of the items in the history on this specific item in topk
        fia_dict = find_fia_mask(user_tensor, item_tensor, item_k_id, recommender_model)
         
        # Sum up all differences between influence on top1 and other topk values
        if not iteration:
            for key in fia_dict.keys():
                items_accent[key] *= factor
        else:
            for key in fia_dict.keys():
                items_accent[key] -= fia_dict[key]
       
    for key in items_accent.keys():
        items_accent[key] *= -1    

    return items_accent


# In[26]:


def single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items_global, recommender_model_local, user_id = None, mask_type = None):
    user_hist_size = np.sum(user_vector)

    if mask_type == 'lime':
        POS_sim_items = explain_instance_lime(
            user_vector_np=user_vector,
            item_id_to_explain=item_id,
            recommender_model=recommender_model_local,
            all_items_tensor_global=all_items_tensor,
            kw_dict_global=kw_dict,
            min_pert=50, max_pert=100, num_perturbations_lime=150,
            kernel_fn=distance_to_proximity,
            num_features_to_select=user_hist_size,
            feature_selection_method='highest_weights',
            explanation_method='POS',
            random_state_lime=item_id
        )

    elif mask_type == 'lire':
        POS_sim_items = explain_instance_lire(
            user_vector_np=user_vector,
            item_id_to_explain=item_id,
            recommender_model=recommender_model_local,
            all_items_tensor_global=all_items_tensor,
            train_array_global=train_array,
            kw_dict_global=kw_dict,
            num_perturbations_lire=200,
            proba_lire=0.1,
            kernel_fn=distance_to_proximity,
            num_features_to_select=user_hist_size,
            feature_selection_method='highest_weights',
            explanation_method='POS',
            random_state_lime=item_id
        )

    else:
        if mask_type == 'pop':
            sim_items = find_pop_mask(user_tensor, item_id) # user_tensor is torch.Tensor here
        elif mask_type == 'jaccard':
            sim_items = find_jaccard_mask(user_tensor, item_id, jaccard_dict) # user_tensor is torch.Tensor
        elif mask_type == 'cosine':
            sim_items = find_cosine_mask(user_tensor, item_id, cosine_dict) # user_tensor is torch.Tensor
        elif mask_type == 'shap':
            sim_items = find_shapley_mask(user_tensor, user_id, recommender_model_local, shap_values, item_to_cluster)
        elif mask_type == 'accent':
            sim_items = find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model_local, 5)
        elif mask_type == 'lxr':
            # Load explainer using centralized function
            # It needs: data_name, recommender_name, checkpoints_path, num_items, device
            # These are available globally in this script.
            lxr_explainer_instance = load_lxr_explainer(
                data_name, recommender_name, checkpoints_path, num_items, device
            )
            # Call centralized find_lxr_mask. user_tensor is torch.Tensor, item_tensor is torch.Tensor
            sim_items = find_lxr_mask(user_tensor, item_tensor, lxr_explainer_instance)

        POS_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1],reverse=True))[0:user_hist_size]
        
    return POS_sim_items


# In[27]:


def single_user_metrics(user_vector, user_tensor, item_id, item_tensor, num_of_bins, recommender_model, expl_dict, **kw_dict):
    POS_masked = user_tensor
    NEG_masked = user_tensor
    POS_masked[item_id]=0
    NEG_masked[item_id]=0
    user_hist_size = np.sum(user_vector)
    
    
    bins=[0]+[len(x) for x in np.array_split(np.arange(user_hist_size), num_of_bins, axis=0)]
    
    POS_at_5 = [0]*(len(bins))
    POS_at_10=[0]*(len(bins))
    POS_at_20=[0]*(len(bins))
    
    DEL = [0]*(len(bins))
    INS = [0]*(len(bins))
    NDCG = [0]*(len(bins))

    
    POS_sim_items = expl_dict
    NEG_sim_items  = list(sorted(dict(POS_sim_items).items(), key=lambda item: item[1],reverse=False))
    
    total_items=0
    for i in range(len(bins)):
        total_items += bins[i]
            
        POS_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        
        for j in POS_sim_items[:total_items]:
            POS_masked[j[0]] = 1
        POS_masked = user_tensor - POS_masked # remove the masked items from the user history

        NEG_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        for j in NEG_sim_items[:total_items]:
            NEG_masked[j[0]] = 1
        NEG_masked = user_tensor - NEG_masked # remove the masked items from the user history 
        
        POS_ranked_list = get_top_k(POS_masked, user_tensor, recommender_model, **kw_dict)
        
        if item_id in list(POS_ranked_list.keys()):
            POS_index = list(POS_ranked_list.keys()).index(item_id)+1
        else:
            POS_index = num_items
        NEG_index = get_index_in_the_list(NEG_masked, user_tensor, item_id, recommender_model, **kw_dict)+1

        # for pos:
        POS_at_5[i] = 1 if POS_index <=5 else 0
        POS_at_10[i] = 1 if POS_index <=10 else 0
        POS_at_20[i] = 1 if POS_index <=20 else 0

        # for del:
        DEL[i] = float(recommender_run(POS_masked, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())

        # for ins:
        INS[i] = float(recommender_run(user_tensor-POS_masked, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())

        #for NDCG:
        NDCG[i]= get_ndcg(list(POS_ranked_list.keys()),item_id, **kw_dict)
        
    res = [DEL, INS, NDCG, POS_at_5, POS_at_10, POS_at_20]
    for i in range(len(res)):
        res[i] = np.array(res[i])
        
    return res


# In[28]:


create_dictionaries = True # if it is the first time generating the explanations - assing "True"

if create_dictionaries:
    recommender.eval()
    # Evaluate the model on the test set
    
    pop_expl_dict = {}
    jaccard_expl_dict = {}
    cosine_expl_dict = {}
    lime_expl_dict = {}
    lxr_expl_dict = {}
    accent_expl_dict = {}
    shap_expl_dict = {}
  

    with torch.no_grad():
        for i in range(test_array.shape[0]):
        #for i in range(3):
            if i%500 == 0:
                print(i)
            start_time = time.time()
            user_vector = test_array[i]
            user_tensor = torch.FloatTensor(user_vector).to(device)
            user_id = int(test_data.index[i])

            item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).detach().cpu().numpy())
            item_vector =  items_array[item_id]
            item_tensor = torch.FloatTensor(item_vector).to(device)

            user_vector[item_id] = 0
            user_tensor[item_id] = 0

            recommender.to(device)

            pop_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'pop')
            jaccard_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'jaccard')
            cosine_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'cosine')
            lime_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'lime')
            lxr_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'lxr')
            accent_expl_dict[user_id] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender, mask_type= 'accent')
          #  shap_expl_dict[user_id] = single_user_expl(user_vector, user_tensor,item_id, item_tensor, num_items, recommender, mask_type= 'shap',user_id = user_id)

            
        with open(Path(files_path,f'{recommender_name}_pop_expl_dict.pkl'), 'wb') as handle:
            pickle.dump(pop_expl_dict, handle)

        with open(Path(files_path,f'{recommender_name}_jaccard_expl_dict.pkl'), 'wb') as handle:
            pickle.dump(jaccard_expl_dict, handle)

        with open(Path(files_path,f'{recommender_name}_cosine_expl_dict.pkl'), 'wb') as handle:
            pickle.dump(cosine_expl_dict, handle)

        with open(Path(files_path,f'{recommender_name}_lime_expl_dict.pkl'), 'wb') as handle:
            pickle.dump(lime_expl_dict, handle)
            
        with open(Path(files_path,f'{recommender_name}_lxr_expl_dict.pkl'), 'wb') as handle:
            pickle.dump(lxr_expl_dict, handle)
            
        with open(Path(files_path,f'{recommender_name}_accent_expl_dict.pkl'), 'wb') as handle:
            pickle.dump(accent_expl_dict, handle) 
            
     #   with open(Path(files_path,f'{recommender_name}_shap_expl_dict.pkl'), 'wb') as handle:
        #    pickle.dump(shap_expl_dict, handle)

            
else: #If it is not the first time you run the code - just read the picleks. Dont rewrite them
        with open(Path(files_path,f'{recommender_name}_jaccard_expl_dict.pkl'), 'rb') as handle:
            jaccard_expl_dict = pickle.load(handle) 

        with open(Path(files_path,f'{recommender_name}_cosine_expl_dict.pkl'), 'rb') as handle:
            cosine_expl_dict = pickle.load(handle)

        with open(Path(files_path,f'{recommender_name}_pop_expl_dict.pkl'), 'rb') as handle:
            pop_expl_dict = pickle.load(handle)

        with open(Path(files_path,f'{recommender_name}_lime_expl_dict.pkl'), 'rb') as handle:
            lime_expl_dict = pickle.load(handle)

        with open(Path(files_path,f'{recommender_name}_lxr_expl_dict.pkl'), 'rb') as handle:
            lxr_expl_dict = pickle.load(handle)
            
        with open(Path(files_path,f'{recommender_name}_accent_expl_dict.pkl'), 'rb') as handle:
            accent_expl_dict = pickle.load(handle)

       #with open(Path(files_path,f'{recommender_name}_shap_expl_dict.pkl'), 'rb') as handle:
           # shap_expl_dict = pickle.load(handle)


# In[29]:


def eval_one_expl_type(expl_name):
    '''
    This function aggregates explanations for all test users
    and computes the average metric values across the entire test set.
    '''
    
    print(f' ============ Start explaining {data_name} {recommender_name} by {expl_name} ============')
    with open(Path(files_path,f'{recommender_name}_{expl_name}_expl_dict.pkl'), 'rb') as handle:
        expl_dict = pickle.load(handle)
    recommender.eval()
    # Evaluate the model on the test set

    num_of_bins = 11
    
    users_DEL = np.zeros(num_of_bins)
    users_INS = np.zeros(num_of_bins)
    NDCG = np.zeros(num_of_bins)
    POS_at_5 = np.zeros(num_of_bins)
    POS_at_10 = np.zeros(num_of_bins)
    POS_at_20 = np.zeros(num_of_bins)

    num_of_bins = 10


    with torch.no_grad():
        for i in range(test_array.shape[0]):
            start_time = time.time()
            user_vector = test_array[i]
            user_tensor = torch.FloatTensor(user_vector).to(device)
            user_id = int(test_data.index[i])

            item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).detach().cpu().numpy())
            item_vector =  items_array[item_id]
            item_tensor = torch.FloatTensor(item_vector).to(device)

            user_vector[item_id] = 0
            user_tensor[item_id] = 0

            user_expl = expl_dict[user_id]

            res = single_user_metrics(user_vector, user_tensor, item_id, item_tensor, num_of_bins, recommender, user_expl, **kw_dict)
            users_DEL += res[0]
            users_INS += res[1]
            NDCG += res[2]
            POS_at_5 += res[3]
            POS_at_10 += res[4]
            POS_at_20 += res[5]

            if i%500 == 0:
                prev_time = time.time()
                print("User {}, total time: {:.2f}".format(i,prev_time - start_time))

    a = i+1

    print(f'users_DEL_{expl_name}: ', np.mean(users_DEL)/a)
    print(f'users_INS_{expl_name}: ', np.mean(users_INS)/a)
    print(f'NDCG_{expl_name}: ', np.mean(NDCG)/a)
    print(f'POS_at_5_{expl_name}: ', np.mean(POS_at_5)/a)
    print(f'POS_at_10_{expl_name}: ', np.mean(POS_at_10)/a)
    print(f'POS_at_20_{expl_name}: ', np.mean(POS_at_20)/a)

    print(np.mean(users_DEL)/a , np.mean(users_INS)/a, np.mean(NDCG)/a, np.mean(POS_at_5)/a, np.mean(POS_at_10)/a, np.mean(POS_at_20)/a)


# In[30]:


#expl_names_list = ['accent'] # specify the names of the baselines for which you wish to calculate the metrics values.


# In[31]:


#for expl_name in expl_names_list:
    #eval_one_expl_type(expl_name)


# In[ ]:




