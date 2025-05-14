#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Keep commented
# export_dir = os.getcwd() # Removed top-level, unused one
from pathlib import Path
import pickle
from collections import defaultdict
import torch
# import torch.nn as nn # Removed, not directly used after LXR Explainer local class removal
# import matplotlib.pyplot as plt # Keep commented
# import ipynb # Remove
from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor # Removed, unused
# from openpyxl.cell.cell import MergedCell # Removed, unused and related code is commented

data_name = "ML1M" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "MLP" ### Can be MLP, VAE, NCF

DP_DIR = Path("processed_data", data_name) 
export_dir = Path(os.getcwd()) # Keep local export_dir for checkpoints_path
files_path = Path("/storage/mikhail/PI4Rec", DP_DIR) # Keep absolute path
checkpoints_path = Path(export_dir.parent, "checkpoints") # Keep this structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_type_dict = {
    "VAE":"multiple",
    "MLP":"single",
    "NCF": "single",
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt"),
    ("Pinterest","NCF"):Path(checkpoints_path, "NCF2_Pinterest_9e-05_32_9_10.pt")
}

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
    ("ML1M","MLP"):Path(checkpoints_path, "MLP1_ML1M_0.0076_256_7.pt"),#MLP_ML1M_0.0026_512_14_4.pt
    ("ML1M","NCF"):Path(checkpoints_path, "NCF_ML1M_5e-05_64_16.pt"),
    
    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),
    ("Yahoo","NCF"):Path(checkpoints_path, "NCF_Yahoo_0.001_64_21_0.pt"),
    
    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_12_18_0.0001_256.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt"),
    ("Pinterest","NCF"):Path(checkpoints_path, "NCF2_Pinterest_9e-05_32_9_10.pt"),
}

hidden_dim_dict = {
    ("ML1M","VAE"): None,
    ("ML1M","MLP"): 32,
    ("ML1M","NCF"): 8,

    ("Yahoo","VAE"): None,
    ("Yahoo","MLP"):32,
    ("Yahoo","NCF"):8,
    
    ("Pinterest","VAE"): None,
    ("Pinterest","MLP"):512,
    ("Pinterest","NCF"): 64,
}

output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 

hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
recommender_path = recommender_path_dict[(data_name,recommender_name)]

train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)

test_array = static_test_data.iloc[:,:-2].to_numpy()
with open(Path(files_path, f'jaccard_based_sim_{data_name}.pkl'), 'rb') as f:
    jaccard_dict = pickle.load(f) 
with open(Path(files_path, f'cosine_based_sim_{data_name}.pkl'), 'rb') as f:
    cosine_dict = pickle.load(f) 
with open(Path(files_path, f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f) 
with open(Path(files_path, f'item_to_cluster_{recommender_name}_{data_name}.pkl'), 'rb') as f:
    item_to_cluster = pickle.load(f) 
with open(Path(files_path, f'shap_values_{recommender_name}_{data_name}.pkl'), 'rb') as f:
    shap_values= pickle.load(f) 
for i in range(num_items):
    for j in range(i, num_items):
        jaccard_dict[(j,i)]= jaccard_dict[(i,j)]
        cosine_dict[(j,i)]= cosine_dict[(i,j)]
        pop_array = np.zeros(len(pop_dict))
for key, value in pop_dict.items():
    pop_array[key] = value
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

# import os # Already imported, remove redundant import

#os.chdir('/storage/mikhail/PI4Rec/code') # Remove
# print(os.getcwd()) # Remove

# sys.path.append('../baselines') # Remove this line

from .lime import explain_instance_lime, explain_instance_lire, distance_to_proximity # NEW

from .help_functions import (
    load_recommender, get_user_recommended_item, recommender_run,
    get_top_k, get_index_in_the_list, get_ndcg,
    load_lxr_explainer, find_lxr_mask
)
# from ipynb.fs.defs.help_functions import * # Remove
# importlib.reload(ipynb.fs.defs.help_functions) # Remove

# from .recommenders_architecture import MLP, VAE, NCF, MLP_model, GMF_model # Keep commented
# from ipynb.fs.defs.recommenders_architecture import * # Remove
# importlib.reload(ipynb.fs.defs.recommenders_architecture) # Remove

# Call the centralized one from help_functions
recommender = load_recommender(data_name, hidden_dim, checkpoints_path, recommender_path, **kw_dict)

def find_pop_mask(x, item_id):
    user_hist = torch.Tensor(x).to(device) # remove the positive item we want to explain from the user history
    user_hist[item_id] = 0
    item_pop_dict = {}
    
    for i,j in enumerate(user_hist>0):
        if j:
            item_pop_dict[i]=pop_array[i] # add the pop of the item to the dictionary
            
    return item_pop_dict

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

def find_fia_mask(user_tensor, item_tensor, item_id, recommender):
    y_pred = recommender_run(user_tensor, recommender, item_tensor, item_id, **kw_dict).to(device)
    items_fia = {}
    user_hist = user_tensor.cpu().detach().numpy().astype(int)
    
    for i in range(num_items):
        if(user_hist[i] == 1):
            user_hist[i] = 0
            temp_user_tensor = torch.FloatTensor(user_hist).to(device)
            y_pred_without_item = recommender_run(temp_user_tensor, recommender, item_tensor, item_id, 'single', **kw_dict).to(device)
            infl_score = y_pred - y_pred_without_item
            items_fia[i] = infl_score
            user_hist[i] = 1

    return items_fia

def find_shapley_mask(user_tensor, user_id, model, shap_values_global, item_to_cluster_global):
    item_shap = {}
    current_shap_values = shap_values_global[shap_values_global[:, 0].astype(int) == user_id][:,1:]
    user_vector_np = user_tensor.cpu().detach().numpy().astype(int)
    for i in np.where(user_vector_np == 1)[0]:
        items_cluster = item_to_cluster_global.get(i)
        if items_cluster is not None and int(items_cluster) < current_shap_values.shape[1]:
             item_shap[i] = current_shap_values.T[int(items_cluster)][0]
        else:
            item_shap[i] = 0
    return item_shap  

def find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model, top_k):
    items_accent = defaultdict(float)
    factor = top_k - 1
    user_accent_hist_np = user_tensor.cpu().detach().numpy().astype(int)
    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender_model, **kw_dict).keys())
    top_k_indices = [sorted_indices[0]] if top_k == 1 else sorted_indices[:top_k]
    for iteration, item_k_id in enumerate(top_k_indices):
        user_accent_hist_np_temp = user_accent_hist_np.copy()
        user_accent_hist_np_temp[item_k_id] = 0
        temp_user_tensor = torch.FloatTensor(user_accent_hist_np_temp).to(device)
        temp_item_vector = items_array[item_k_id]
        temp_item_tensor = torch.FloatTensor(temp_item_vector).to(device)
        fia_dict = find_fia_mask(temp_user_tensor, temp_item_tensor, item_k_id, recommender_model)
        if not iteration:
            for key_fia in fia_dict.keys():
                items_accent[key_fia] *= factor
        else:
            for key_fia in fia_dict.keys():
                items_accent[key_fia] -= fia_dict[key_fia]
    for key_accent in items_accent.keys():
        items_accent[key_accent] *= -1    
    return items_accent

def single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items_global, recommender_model_local, user_id = None, mask_type = None):
    user_hist_size = np.sum(user_vector)
    sim_items = {}

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
            sim_items = find_pop_mask(user_vector, item_id) 
        elif mask_type == 'jaccard':
            sim_items = find_jaccard_mask(user_vector, item_id, jaccard_dict)
        elif mask_type == 'cosine':
            sim_items = find_cosine_mask(user_vector, item_id, cosine_dict)
        elif mask_type == 'shap':
            sim_items = find_shapley_mask(user_tensor, user_id, recommender_model_local, shap_values, item_to_cluster)
        elif mask_type == 'accent':
            sim_items = find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model_local, 5)
        elif mask_type == 'lxr':
            lxr_explainer_instance = load_lxr_explainer(
                data_name, recommender_name, checkpoints_path, num_items, device
            )
            sim_items = find_lxr_mask(user_tensor, item_tensor, lxr_explainer_instance)

        POS_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]

    return POS_sim_items

def single_user_metrics(user_vector, user_tensor, item_id, item_tensor, recommender_model, expl_dict, **kw_dict):
    """
    Calculate metrics for a single user with 5 steps of item masking
    Now explicitly includes POS@20 and NEG@20 metrics
    """
    POS_masked = user_tensor.clone()
    NEG_masked = user_tensor.clone()
    POS_masked[item_id] = 0
    NEG_masked[item_id] = 0
    
    # Use 5 steps
    num_steps = 5
    bins = range(1, num_steps + 1)  # [1, 2, 3, 4, 5]
    
    # Initialize metric arrays with 5 elements each
    POS_at_5 = [0] * num_steps
    POS_at_10 = [0] * num_steps
    POS_at_20 = [0] * num_steps  # Explicitly initialize POS@20
    
    NEG_at_5 = [0] * num_steps
    NEG_at_10 = [0] * num_steps
    NEG_at_20 = [0] * num_steps  # Explicitly initialize NEG@20
    
    DEL = [0] * num_steps
    INS = [0] * num_steps
    NDCG = [0] * num_steps
    
    # Get sorted items by importance
    POS_sim_items = expl_dict
    NEG_sim_items = list(sorted(dict(POS_sim_items).items(), key=lambda item: item[1], reverse=False))
    
    # For each step (1 to 5 items)
    for i, num_items_to_mask in enumerate(bins):
        # Create masks for positive and negative cases
        POS_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=kw_dict['device'])
        NEG_masked = torch.zeros_like(user_tensor, dtype=torch.float32, device=kw_dict['device'])
        
        # Apply POS masking
        for j in POS_sim_items[:num_items_to_mask]:
            POS_masked[j[0]] = 1
        POS_masked = user_tensor * (1 - POS_masked)
        
        # Apply NEG masking
        for j in NEG_sim_items[:num_items_to_mask]:
            NEG_masked[j[0]] = 1
        NEG_masked = user_tensor * (1 - NEG_masked)
        
        # Get rankings for both POS and NEG
        POS_ranked_list = get_top_k(POS_masked, user_tensor, recommender_model, **kw_dict)
        NEG_ranked_list = get_top_k(NEG_masked, user_tensor, recommender_model, **kw_dict)
        
        # Calculate POS ranking
        if item_id in list(POS_ranked_list.keys()):
            POS_index = list(POS_ranked_list.keys()).index(item_id) + 1
        else:
            POS_index = kw_dict['num_items']
            
        # Calculate NEG ranking
        if item_id in list(NEG_ranked_list.keys()):
            NEG_index = list(NEG_ranked_list.keys()).index(item_id) + 1
        else:
            NEG_index = kw_dict['num_items']
        
        # Calculate ALL metrics including P@20
        POS_at_5[i] = 1 if POS_index <= 5 else 0
        POS_at_10[i] = 1 if POS_index <= 10 else 0
        POS_at_20[i] = 1 if POS_index <= 20 else 0  # Explicitly calculate POS@20
        
        NEG_at_5[i] = 1 if NEG_index <= 5 else 0
        NEG_at_10[i] = 1 if NEG_index <= 10 else 0
        NEG_at_20[i] = 1 if NEG_index <= 20 else 0  # Explicitly calculate NEG@20
        
        # Calculate other metrics
        DEL[i] = float(recommender_run(POS_masked, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())
        INS[i] = float(recommender_run(user_tensor - POS_masked, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())
        NDCG[i] = get_ndcg(list(POS_ranked_list.keys()), item_id, **kw_dict)
    
    # Return all metrics including P@20
    res = [DEL, INS, NDCG, 
           POS_at_5, POS_at_10, POS_at_20,  # Include POS@20
           NEG_at_5, NEG_at_10, NEG_at_20]  # Include NEG@20
    return [np.array(x) for x in res]

class MetricsBaselines:
    def __init__(self, data_name, recommender_name):
        self.data_name = data_name
        self.recommender_name = recommender_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_data_and_recommender()

    def setup_data_and_recommender(self):
        # Set up all necessary data and variables
        DP_DIR = Path("processed_data", self.data_name)
        self.files_path = Path(export_dir.parent, DP_DIR)
        self.num_users = num_users_dict[self.data_name]
        self.num_items = num_items_dict[self.data_name]
        
        self.test_data = pd.read_csv(Path(self.files_path, f'test_data_{self.data_name}.csv'), index_col=0)
        self.test_array = self.test_data.to_numpy()
        self.items_array = np.eye(self.num_items)
        
        with open(Path(self.files_path, f'pop_dict_{self.data_name}.pkl'), 'rb') as f:
            self.pop_dict = pickle.load(f)
        
        # Load other necessary data (jaccard_dict, cosine_dict, item_to_cluster, shap_values)
        
        self.kw_dict = {
            'device': self.device,
            'num_items': self.num_items,
            'num_features': self.num_items,
            'demographic': False,
            'pop_array': np.array([self.pop_dict.get(i, 0) for i in range(self.num_items)]),
            'all_items_tensor': torch.eye(self.num_items).to(self.device),
            'static_test_data': self.test_data,
            'items_array': self.items_array,
            'output_type': output_type_dict[self.recommender_name],
            'recommender_name': self.recommender_name,
            'files_path': self.files_path
        }
        
        self.recommender = self.load_recommender()

def process_user(user_index, test_array, test_data, recommender, kw_dict):
    try:
        user_vector = test_array[user_index]
        user_tensor = torch.FloatTensor(user_vector).to(kw_dict['device'])
        user_id = int(test_data.index[user_index])

        item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).detach().cpu().numpy())
        item_vector = kw_dict['items_array'][item_id]
        item_tensor = torch.FloatTensor(item_vector).to(kw_dict['device'])

        user_vector[item_id] = 0
        user_tensor[item_id] = 0

        results = {}
        for method in ['pop', 'jaccard', 'cosine', 'lime', 'lxr', 'accent', 'shap']:
            results[method] = single_user_expl(user_vector, user_tensor, item_id, item_tensor, kw_dict['num_items'], recommender, mask_type=method, user_id=user_id if method == 'shap' else None)

        return user_id, results
    except Exception as e:
        print(f"Error processing user {user_id}: {str(e)}")
        return None

def eval_one_expl_type(expl_name):
    print(f' ============ Start explaining {data_name} {recommender_name} by {expl_name} ============')
    
    # Load the appropriate explanation dictionary
    if expl_name == 'PI_base':
        with open(Path(files_path, f'{recommender_name}_PI_base_expl_dict.pkl'), 'rb') as handle:
            expl_dict = pickle.load(handle)
    else:
        with open(Path(files_path,f'{recommender_name}_{expl_name}_expl_dict.pkl'), 'rb') as handle:
            expl_dict = pickle.load(handle)
    
    recommender.eval()
    
    # Initialize arrays for metrics with 5 steps
    num_steps = 5
    users_DEL = np.zeros(num_steps)
    users_INS = np.zeros(num_steps)
    NDCG = np.zeros(num_steps)
    
    POS_at_5 = np.zeros(num_steps)
    POS_at_10 = np.zeros(num_steps)
    POS_at_20 = np.zeros(num_steps)
    
    NEG_at_5 = np.zeros(num_steps)
    NEG_at_10 = np.zeros(num_steps)
    NEG_at_20 = np.zeros(num_steps)

    with torch.no_grad():
        for i in tqdm(range(test_array.shape[0])):
            user_vector = test_array[i]
            user_tensor = torch.FloatTensor(user_vector).to(device)
            user_id = int(test_data.index[i])

            item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).detach().cpu().numpy())
            item_vector = items_array[item_id]
            item_tensor = torch.FloatTensor(item_vector).to(device)

            user_vector[item_id] = 0
            user_tensor[item_id] = 0

            user_expl = expl_dict[user_id]

            res = single_user_metrics(user_vector, user_tensor, item_id, item_tensor, recommender, user_expl, **kw_dict)
            users_DEL += res[0]
            users_INS += res[1]
            NDCG += res[2]
            POS_at_5 += res[3]
            POS_at_10 += res[4]
            POS_at_20 += res[5]
            NEG_at_5 += res[6]
            NEG_at_10 += res[7]
            NEG_at_20 += res[8]

    a = test_array.shape[0]

    print(f'users_DEL_{expl_name}: ', np.mean(users_DEL)/a)
    print(f'users_INS_{expl_name}: ', np.mean(users_INS)/a)
    print(f'NDCG_{expl_name}: ', np.mean(NDCG)/a)
    print(f'POS_at_5_{expl_name}: ', np.mean(POS_at_5)/a)
    print(f'POS_at_10_{expl_name}: ', np.mean(POS_at_10)/a)
    print(f'POS_at_20_{expl_name}: ', np.mean(POS_at_20)/a)
    print(f'NEG_at_5_{expl_name}: ', np.mean(NEG_at_5)/a)
    print(f'NEG_at_10_{expl_name}: ', np.mean(NEG_at_10)/a)
    print(f'NEG_at_20_{expl_name}: ', np.mean(NEG_at_20)/a)

    return {
        'DEL': users_DEL/a,
        'INS': users_INS/a,
        'NDCG': NDCG/a,
        'POS_at_5': POS_at_5/a,
        'POS_at_10': POS_at_10/a,
        'POS_at_20': POS_at_20/a,
        'NEG_at_5': NEG_at_5/a,
        'NEG_at_10': NEG_at_10/a,
        'NEG_at_20': NEG_at_20/a
    }

def run_all_baselines(data_name, recommender_name):
    global num_users, num_items, device, kw_dict, recommender, test_array, test_data, items_array, jaccard_dict, cosine_dict, pop_dict, item_to_cluster, shap_values

    # Update global variables for the current dataset and recommender
    num_users = num_users_dict[data_name]
    num_items = num_items_dict[data_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset-specific files
    DP_DIR = Path("processed_data", data_name)
    files_path = Path(export_dir.parent, DP_DIR)
    test_data = pd.read_csv(Path(files_path, f'test_data_{data_name}.csv'), index_col=0)
    test_array = test_data.to_numpy()
    items_array = np.eye(num_items)

    with open(Path(files_path, f'jaccard_based_sim_{data_name}.pkl'), 'rb') as f:
        jaccard_dict = pickle.load(f)
    with open(Path(files_path, f'cosine_based_sim_{data_name}.pkl'), 'rb') as f:
        cosine_dict = pickle.load(f)
    with open(Path(files_path, f'pop_dict_{data_name}.pkl'), 'rb') as f:
        pop_dict = pickle.load(f)
    with open(Path(files_path, f'item_to_cluster_{recommender_name}_{data_name}.pkl'), 'rb') as f:
        item_to_cluster = pickle.load(f)
    with open(Path(files_path, f'shap_values_{recommender_name}_{data_name}.pkl'), 'rb') as f:
        shap_values = pickle.load(f)

    # Update kw_dict
    kw_dict = {
        'device': device,
        'num_items': num_items,
        'num_features': num_items,
        'demographic': False,
        'pop_array': np.array([pop_dict.get(i, 0) for i in range(num_items)]),
        'all_items_tensor': torch.eye(num_items).to(device),
        'static_test_data': test_data,
        'items_array': items_array,
        'output_type': output_type_dict[recommender_name],
        'recommender_name': recommender_name,
        'files_path': files_path
    }

    # Load recommender
    recommender = load_recommender()

    # Run all baselines
    baselines = [ 'jaccard', 'cosine', 'lime', 'lxr', 'accent', 'shap']
    results = {}

    for baseline in baselines:
        print(f"Running {baseline} baseline for {data_name} {recommender_name}")
        results[baseline] = eval_one_expl_type(baseline)

    return results

# def plot_all_metrics(results, data_name, recommender_name):
#     # Mapping between metric keys and their display names (title, y-label, indicator)
#     metrics_mapping = {
#         'DEL':      ('AUC DEL-P@K', 'DEL@Kₑ', 'Lower is better'),
#         'INS':      ('AUC INS-P@K', 'INS@Kₑ', 'Higher is better'),
#         'NDCG':     ('AUC NDCG-P',  'CNDCG@Kₑ', 'Lower is better'),
#         'POS_at_5': ('AUC POS-P@5', 'POS@20Kₑ', 'Lower is better'),
#         'POS_at_10':('AUC POS-P@10','POS@20Kₑ', 'Lower is better'),
#         'POS_at_20':('AUC POS-P@20','POS@20Kₑ', 'Lower is better'),
#         'NEG_at_5': ('AUC NEG-P@5', 'NEG@20Kₑ', 'Higher is better'),
#         'NEG_at_10':('AUC NEG-P@10','NEG@20Kₑ', 'Higher is better'),
#         'NEG_at_20':('AUC NEG-P@20','NEG@20Kₑ', 'Higher is better')
#     }
    
#     # Style lists
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
#     markers = ['o', 's', '^', 'D', 'v', 'x']
#     linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    
#     # Create plots directory
#     os.makedirs('plots_discrete', exist_ok=True)
    
#     # Plot each metric
#     for metric, (title_name, y_label, indicator) in metrics_mapping.items():
#         plt.figure(figsize=(12, 8))
        
#         for i, (baseline, baseline_metrics) in enumerate(results.items()):
#             if metric not in baseline_metrics:
#                 print(f"Warning: {metric} not found in {baseline} metrics")
#                 continue
            
#             values = baseline_metrics[metric][:5]  # Take only first 5 values
#             x = range(1, len(values) + 1)  # Numbers of masked items (1, 2, 3, 4, 5)
            
#             plt.plot(
#                 x, values,
#                 label=baseline.upper(),
#                 color=colors[i % len(colors)],
#                 linestyle=linestyles[i % len(linestyles)],
#                 marker=markers[i % len(markers)],
#                 markersize=8,
#                 linewidth=2,
#                 markevery=1  # Markers on each value
#             )
        
#         plt.xlabel("Number of Masked Items", fontsize=30)
#         plt.ylabel(y_label, fontsize=30)
#         plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
#         plt.xticks(range(1, 6), fontsize=18)
#         plt.yticks(fontsize=18)
#         plt.legend(
#             loc='best', 
#             fontsize=20,
#             frameon=True,
#             edgecolor='black'
#         )

#         plt.tight_layout()
        
#         # Save the plot
#         safe_display_name = title_name.replace(" ", "_").replace("@", "at")
#         plt.savefig(f'plots_discrete/{safe_display_name}_{data_name}_{recommender_name}d.pdf',
#                     format='pdf', bbox_inches='tight')
#         print(f"Saved plot: {safe_display_name}_{data_name}_{recommender_name}d.pdf")
#         plt.close()

def process_recommender(data_name, recommender_name):
    DP_DIR = Path("processed_data", data_name)
    files_path = Path("/storage/mikhail/PI4Rec", DP_DIR)
    
    num_users = num_users_dict[data_name]
    num_items = num_items_dict[data_name]
    num_features = num_items_dict[data_name]
    
    with open(Path(files_path, f'pop_dict_{data_name}.pkl'), 'rb') as f:
        pop_dict = pickle.load(f)
    pop_array = np.zeros(len(pop_dict))
    for key, value in pop_dict.items():
        pop_array[key] = value

    test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
    static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
    
    test_array = static_test_data.iloc[:,:-2].to_numpy()
    items_array = np.eye(num_items)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_items_tensor = torch.Tensor(items_array).to(device)

    output_type = output_type_dict[recommender_name]
    hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
    recommender_path = recommender_path_dict[(data_name,recommender_name)]

    kw_dict = {
        'device': device,
        'num_items': num_items,
        'demographic': False,
        'num_features': num_features,
        'pop_array': pop_array,
        'all_items_tensor': all_items_tensor,
        'static_test_data': static_test_data,
        'items_array': items_array,
        'output_type': output_type,
        'recommender_name': recommender_name
    }

    recommender = load_recommender()

    print(f"Processing {data_name} dataset with {recommender_name} recommender")
    
    results = {}
    for expl_name in [ 'jaccard', 'cosine', 'lime', 'lxr', 'accent', 'shap']:
        results[expl_name] = eval_one_expl_type(expl_name, data_name, recommender_name, test_array, test_data, items_array, recommender, kw_dict)
    
   # plot_all_metrics(results, data_name, recommender_name) # Call commented out

# from openpyxl import Workbook # Commented out
# from openpyxl.styles import Font, Alignment, Border, Side # Commented out

# def save_results_to_excel(results, filename):
#     wb = Workbook()
    
#     # Create MF recommender sheet
#     ws_mf = wb.active
#     ws_mf.title = "MF Recommender"
    
#     # Create VAE recommender sheet
#     ws_vae = wb.create_sheet(title="VAE Recommender")
    
#     for ws, title in [(ws_mf, "AUC values for explaining an MF recommender."), 
#                       (ws_vae, "AUC values for explaining a VAE recommender.")]:
        
#         # Add title
#         ws['A1'] = f"Table: {title}"
#         ws['A1'].font = Font(bold=True)
#         ws.merge_cells('A1:G1')
        
#         # Add headers
#         headers = ['Method', 'k=5', 'k=10', 'k=20', 'DEL', 'INS', 'NDCG']
#         for col, header in enumerate(headers, start=1):
#             ws.cell(row=3, column=col, value=header).font = Font(bold=True)
        
#         # Add data
#         for row, (method, values) in enumerate(results.items(), start=4):
#             ws.cell(row=row, column=1, value=method)
#             for col, value in enumerate(values, start=2):
#                 ws.cell(row=row, column=col, value=value)
    
#     # Apply some styling
#     for ws in [ws_mf, ws_vae]:
#         for row in ws[f'A3:G{ws.max_row}']:
#             for cell in row:
#                 cell.border = Border(left=Side(style='thin'), 
#                                      right=Side(style='thin'), 
#                                      top=Side(style='thin'), 
#                                      bottom=Side(style='thin'))
    
#     wb.save(filename)

def run_and_format_results(data_name, recommender_name):
    results = {}
    for expl_name in ['jaccard', 'cosine', 'lime', 'shap', 'accent', 'lxr']:
        raw_results = eval_one_expl_type(expl_name)
        
        # Extract POS values
        pos_at_5 = raw_results['POS_at_5'][-1]  # Last value represents 100% of items
        pos_at_10 = raw_results['POS_at_10'][-1]
        pos_at_20 = raw_results['POS_at_20'][-1]
        
        # Format results as per the desired output
        results[expl_name.upper()] = [
            pos_at_5,
            pos_at_10,
            pos_at_20,
            raw_results['DEL'][-1],
            raw_results['INS'][-1],
            raw_results['NDCG'][-1]
        ]
    
    return results

import pandas as pd
import numpy as np
# import seaborn as sns # Commented out
# import matplotlib.pyplot as plt # Already commented out
# from matplotlib.gridspec import GridSpec # Commented out

# def create_comparison_visualizations(all_results, save_dir='./'): 
#     """
#     Creates comprehensive visualizations comparing methods across datasets and recommenders
    
#     Parameters:
#     all_results: dict
#         Format: {(dataset_name, recommender_name): results_dict}
#         where results_dict contains metrics for each explanation method
#     save_dir: str
#         Directory to save the visualization files
#     """
#     # Prepare data for plotting
#     plot_data = []
#     for (dataset, recommender), results in all_results.items():
#         for method, metrics in results.items():
#             for metric_name, values in metrics.items():
#                 if isinstance(values, np.ndarray):
#                     for step, value in enumerate(values, 1):
#                         plot_data.append({
#                             'Dataset': dataset,
#                             'Recommender': recommender,
#                             'Method': method.upper(),
#                             'Metric': metric_name,
#                             'Step': step,
#                             'Value': value
#                         })
    
#     df = pd.DataFrame(plot_data)

#     # 1. Heatmap of method performance across datasets and recommenders
#     plt.figure(figsize=(15, 10))
#     metrics_to_plot = ['DEL', 'INS', 'NDCG']
    
#     for idx, metric in enumerate(metrics_to_plot):
#         plt.subplot(1, 3, idx+1)
#         pivot_data = df[df['Metric'] == metric].groupby(
#             ['Dataset', 'Recommender', 'Method'])['Value'].mean().unstack()
#         sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
#         plt.title(f'{metric} Performance Comparison')
    
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/performance_heatmap.png')
#     plt.close()

#     # 2. Method stability analysis (standard deviation across steps)
#     plt.figure(figsize=(12, 6))
#     stability_data = df.groupby(['Method', 'Metric'])['Value'].std().unstack()
#     stability_data.plot(kind='bar', width=0.8)
#     plt.title('Method Stability Analysis (Standard Deviation Across Steps)')
#     plt.xlabel('Explanation Method')
#     plt.ylabel('Standard Deviation')
#     plt.xticks(rotation=45)
#     plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1))
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/method_stability.png')
#     plt.close()

#     # 3. Radar chart for method comparison
#     def create_radar_chart(data, methods, metrics):
#         angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
#         fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
#         for method in methods:
#             values = [data[(data['Method'] == method) & 
#                           (data['Metric'] == metric)]['Value'].mean() 
#                      for metric in metrics]
#             values += values[:1]
#             angles_plot = np.concatenate([angles, [angles[0]]])
#             ax.plot(angles_plot, values, 'o-', label=method)
#             ax.fill(angles_plot, values, alpha=0.25)
        
#         ax.set_xticks(angles)
#         ax.set_xticklabels(metrics)
#         ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
#         return fig

#     metrics_for_radar = ['DEL', 'INS', 'NDCG', 'POS_at_5', 'POS_at_10']
#     for dataset in df['Dataset'].unique():
#         for recommender in df['Recommender'].unique():
#             data_subset = df[(df['Dataset'] == dataset) & 
#                            (df['Recommender'] == recommender)]
#             fig = create_radar_chart(data_subset, 
#                                    df['Method'].unique(), 
#                                    metrics_for_radar)
#             plt.title(f'{dataset} - {recommender}\nMethod Comparison')
#             plt.savefig(f'{save_dir}/radar_{dataset}_{recommender}.png')
#             plt.close()

#     # 4. Box plots showing distribution of metrics across steps
#     plt.figure(figsize=(15, 10))
#     for idx, metric in enumerate(['DEL', 'INS', 'NDCG'], 1):
#         plt.subplot(1, 3, idx)
#         sns.boxplot(data=df[df['Metric'] == metric], 
#                    x='Method', y='Value', 
#                    hue='Dataset')
#         plt.title(f'{metric} Distribution')
#         plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/metric_distributions.png')
#     plt.close()

#     # 5. Performance improvement over steps
#     plt.figure(figsize=(15, 8))
#     for metric in ['DEL', 'INS', 'NDCG']:
#         plt.subplot(1, 3, metrics_to_plot.index(metric) + 1)
#         for method in df['Method'].unique():
#             data = df[(df['Metric'] == metric) & (df['Method'] == method)]
#             plt.plot(data.groupby('Step')['Value'].mean(), 
#                     marker='o', 
#                     label=method)
#         plt.title(f'{metric} Progress Over Steps')
#         plt.xlabel('Step')
#         plt.ylabel('Value')
#         if metric == 'NDCG':
#             plt.legend(bbox_to_anchor=(1.05, 1))
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/progress_over_steps.png')
#     plt.close()

#     return df

# def generate_summary_report(df, save_dir='./'):
#     """
#     Generates a statistical summary report of the results
#     """
#     # Calculate aggregate statistics
#     summary = df.groupby(['Dataset', 'Recommender', 'Method', 'Metric'])['Value'].agg([
#         'mean', 'std', 'min', 'max'
#     ]).round(3)
    
#     # Save summary to CSV
#     summary.to_csv(f'{save_dir}/summary_statistics.csv')
    
#     # Calculate method rankings
#     rankings = df.groupby(['Dataset', 'Recommender', 'Method', 'Metric'])['Value'].mean().unstack()
#     method_ranks = rankings.rank(ascending=False, method='min')
#     method_ranks.to_csv(f'{save_dir}/method_rankings.csv')
    
#     return summary, method_ranks

# def create_results_table(results, data_name, recommender_name):
#     """
#     Create a comprehensive table of all metrics for each method and masking step
#     """
#     # Initialize the table structure
#     table_data = []
#     metrics = ['DEL', 'INS', 'NDCG', 'POS_at_5', 'POS_at_10', 'POS_at_20']
    
#     for method in results.keys():
#         for step in range(5):  # 5 steps
#             row = {
#                 'Method': method.upper(),
#                 'Step': step + 1,  # 1-based indexing
#                 'Dataset': data_name,
#                 'Recommender': recommender_name
#             }
            
#             # Add all metrics for this method and step
#             for metric in metrics:
#                 row[metric] = results[method][metric][step]
            
#             table_data.append(row)
    
#     # Create DataFrame
#     df = pd.DataFrame(table_data)
    
#     # Save to CSV
#     csv_filename = f'results_{data_name}_{recommender_name}.csv'
#     df.to_csv(csv_filename, index=False)
    
#     # Create and save a formatted Excel file
#     # wb = Workbook() # Already commented
#     # ws = wb.active # Already commented
#     # ws.title = f"{data_name}_{recommender_name}_Results" # Already commented
    
#     # Add title (in row 1)
#     # ws['A1'] = f"Results for {data_name} dataset with {recommender_name} recommender" # Already commented
#     # ws.merge_cells('A1:H1') # Already commented
#     # ws['A1'].font = Font(bold=True) # Already commented
    
#     # Add headers (in row 3)
#     headers = ['Method', 'Step', 'DEL', 'INS', 'NDCG', 'POS@5', 'POS@10', 'POS@20']
#     # for col, header in enumerate(headers, 1): # Already commented
#         # cell = ws.cell(row=3, column=col, value=header) # Already commented
#         # cell.font = Font(bold=True) # Already commented
    
#     # Add data with formatting
#     current_method = None
#     row_num = 4
#     # for _, row in df.iterrows(): # Already commented
#         # if current_method != row['Method']:
#             # current_method = row['Method']
#             # row_num += 1  # Add space between methods
        
#         # ws.cell(row=row_num, column=1, value=row['Method'])
#         # ws.cell(row=row_num, column=2, value=row['Step'])
#         # ws.cell(row=row_num, column=3, value=float(row['DEL']))
#         # ws.cell(row=row_num, column=4, value=float(row['INS']))
#         # ws.cell(row=row_num, column=5, value=float(row['NDCG']))
#         # ws.cell(row=row_num, column=6, value=float(row['POS_at_5']))
#         # ws.cell(row=row_num, column=7, value=float(row['POS_at_10']))
#         # ws.cell(row=row_num, column=8, value=float(row['POS_at_20']))
        
#         # row_num += 1
    
#     # Apply formatting to all cells
#     # for row in ws.iter_rows(min_row=3, max_row=row_num-1): # Already commented
#         # for cell in row: # Already commented
#             # cell.border = Border(
#                 # left=Side(style='thin'),
#                 # right=Side(style='thin'),
#                 # top=Side(style='thin'),
#                 # bottom=Side(style='thin')
#             # )
#             # if isinstance(cell.value, float):
#                 # cell.number_format = '0.000'
    
#     # Adjust column widths (skip merged cells)
#     # column_widths = {} # Already commented
#     # for row in ws.iter_rows(min_row=3):  # Start from row 3 to skip merged cells
#         # for cell in row: # Already commented
#             # if isinstance(cell, MergedCell):
#                 # continue
#             # col = cell.column_letter
#             # width = len(str(cell.value)) + 2
#             # current_width = column_widths.get(col, 0)
#             # column_widths[col] = max(current_width, width)
    
#     # Apply the calculated widths
#     # for col, width in column_widths.items(): # Already commented
#         # ws.column_dimensions[col].width = width
    
#     # Save Excel file
#     # excel_filename = f'results_{data_name}_{recommender_name}.xlsx' # Already commented
#     # wb.save(excel_filename) # Already commented
    
#     # Return DataFrame for further analysis if needed
#     return df

# def print_summary_statistics(df):
#     """
#     Print summary statistics for each method
#     """
#     print("\nSummary Statistics:")
#     print("=" * 80)
    
#     methods = df['Method'].unique()
#     metrics = ['DEL', 'INS', 'NDCG', 'POS_at_5', 'POS_at_10', 'POS_at_20']
    
#     summary_rows = []
#     for method in methods:
#         method_data = df[df['Method'] == method]
#         row = {'Method': method}
        
#         for metric in metrics:
#             row[f'{metric}_Mean'] = method_data[metric].mean()
#             row[f'{metric}_Std'] = method_data[metric].std()
        
#         summary_rows.append(row)
    
#     summary_df = pd.DataFrame(summary_rows)
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.width', None)
#     print(summary_df.round(3))
#     return summary_df

# # Use the functions
# def generate_tables(results, data_name, recommender_name):
#     df = create_results_table(results, data_name, recommender_name)
#     summary_df = print_summary_statistics(df)
#     return df, summary_df

data_names = ["ML1M"]#, ,"Yahoo","Pinterest"
recommender_names = [ "MLP"]#"MLP""VAE", "NCF"

# Create a mapping between explainer names and actual explainer functions
explainer_mapping = {
   # 'pop': find_pop_mask,
    'jaccard': find_jaccard_mask,
    'cosine': find_cosine_mask,
    'lime': explain_instance_lime,
    'lire': explain_instance_lire,
    'lxr': find_lxr_mask,
    'accent': find_accent_mask,
    'shap': find_shapley_mask
}

# Initialize storage for all results
# Initialize storage for all results
all_results = {}

# Create plots directory
plots_dir = Path('plots_discrete')
plots_dir.mkdir(exist_ok=True)

for data_name in data_names:
    DP_DIR = Path("processed_data", data_name)
    files_path = Path("/storage/mikhail/PI4Rec", DP_DIR)

    # Load dataset-specific parameters and data
    num_users = num_users_dict[data_name] 
    num_items = num_items_dict[data_name] 
    num_features = num_items_dict[data_name]
        
    with open(Path(files_path, f'pop_dict_{data_name}.pkl'), 'rb') as f:
        pop_dict = pickle.load(f)
    pop_array = np.zeros(len(pop_dict))
    for key, value in pop_dict.items():
        pop_array[key] = value

    # Load data files
    train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
    test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
    static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
    
    train_array = train_data.to_numpy()
    test_array = test_data.to_numpy()
    items_array = np.eye(num_items)
    all_items_tensor = torch.Tensor(items_array).to(device)
    test_array = static_test_data.iloc[:,:-2].to_numpy()

    for recommender_name in recommender_names:
        print(f"\nProcessing {data_name} dataset with {recommender_name} recommender")
        
        # Set up recommender-specific parameters
        output_type = output_type_dict[recommender_name]
        hidden_dim = hidden_dim_dict[(data_name,recommender_name)]
        recommender_path = recommender_path_dict[(data_name,recommender_name)]

        kw_dict = {
            'device': device,
            'num_items': num_items,
            'demographic': False,
            'num_features': num_features,
            'pop_array': pop_array,
            'all_items_tensor': all_items_tensor,
            'static_test_data': static_test_data,
            'items_array': items_array,
            'output_type': output_type,
            'recommender_name': recommender_name
        }

        recommender = load_recommender()

        # Process each explanation method
        results = {}
        for expl_name in ['jaccard', 'cosine', 'lime', 'lxr', 'accent', 'shap']:
            try:
                results[expl_name] = eval_one_expl_type(expl_name)
                # Take only first 5 values from each metric array
                results[expl_name] = {
                    metric: values[:5] if len(values) >= 5 else values 
                    for metric, values in results[expl_name].items()
                }
            except Exception as e:
                print(f"Error processing {expl_name} for {data_name} {recommender_name}: {str(e)}")
                continue

        # Store results in the overall dictionary
        all_results[(data_name, recommender_name)] = results
        
        # Generate and save plots
        # plot_all_metrics(results, data_name, recommender_name) # Call commented out

print("\nProcessing complete. Results and visualizations have been saved to plots_discrete directory.")

# plot_all_metrics(results, data_name, recommender_name) # Call commented out

# create_results_table(results, data_name, recommender_name) # Call commented out
