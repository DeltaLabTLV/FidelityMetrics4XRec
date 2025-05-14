#!/usr/bin/env python
# coding: utf-8

# ### This notebook includes the framework's functions that are being used in all notebooks.
# # Imports

# In[1]:


import pandas as pd
import numpy as np
# os.environ[\'KMP_DUPLICATE_LIB_OK\'] = \'True\' # Commented out for now
# export_dir = os.getcwd() # Removed, appeared unused
from pathlib import Path
import pickle # Added for setup_shap_experiment_data
import torch
import torch.nn as nn
import os # For os.getcwd() in setup_shap_experiment_data
# Combine imports from .recommenders_architecture
from .recommenders_architecture import MLP, VAE, NCF, MLP_model, GMF_model, Explainer

# Global Configuration Dictionaries
OUTPUT_TYPE_DICT = {
    "VAE": "multiple",
    "MLP": "single",
    "NCF": "single"
}

NUM_USERS_DICT = {
    "ML1M": 6037,
    "Yahoo": 13797,
    "Pinterest": 19155
}

NUM_ITEMS_DICT = {
    "ML1M": 3381,
    "Yahoo": 4604,
    "Pinterest": 9362
}

# Corrected .ptt to .pt for MLP_ML1M for consistency, verify if this is intended
RECOMMENDER_PATH_DICT = {
    ("ML1M", "VAE"): "VAE_ML1M_0.0007_128_10.pt",
    ("ML1M", "MLP"): "MLP_ML1M_0.002_1024_19_8.pt",
    ("ML1M", "NCF"): "NCF_ML1M_5e-05_64_16.pt",
    ("Yahoo", "VAE"): "VAE_Yahoo_0.0001_128_13.pt",
    ("Yahoo", "MLP"): "MLP2_Yahoo_0.0083_128_1.pt",
    ("Yahoo", "NCF"): "NCF_Yahoo_0.001_64_21_0.pt",
    ("Pinterest", "VAE"): "VAE_Pinterest_12_18_0.0001_256.pt",
    ("Pinterest", "MLP"): "MLP_Pinterest_0.0062_512_21_0.pt",
    ("Pinterest", "NCF"): "NCF2_Pinterest_9e-05_32_9_10.pt",
}

HIDDEN_DIM_DICT = {
    ("ML1M", "VAE"): None,
    ("ML1M", "MLP"): 32, # SHAP_MLP used 32, help_functions old load_recommender for MLP implied it via hidden_dim
    ("ML1M", "NCF"): 8,
    ("Yahoo", "VAE"): None,
    ("Yahoo", "MLP"): 32,
    ("Yahoo", "NCF"): 8,
    ("Pinterest", "VAE"): None,
    ("Pinterest", "MLP"): 512,
    ("Pinterest", "NCF"): 64,
}

VAE_BASE_CONFIG = {
    "enc_dims": [512, 128],
    "dropout": 0.5,
    "anneal_cap": 0.2,
    "total_anneal_steps": 200000
}

PINTEREST_VAE_CONFIG = {
    "enc_dims": [256, 64],
    "dropout": 0.5,
    "anneal_cap": 0.2,
    "total_anneal_steps": 200000
}

LXR_CHECKPOINT_DICT = {
    ("ML1M","VAE"): ('LXR_ML1M_VAE_26_38_128_3.185652725834087_1.420642300151426.pt',128),
    ("ML1M","MLP"): ('LXR_ML1M_MLP_19_3_128_13.109692424872248_7.829643365925428.pt',128),
    ("ML1M","NCF"): ('LXR_ML1M_NCF_17_38_64_14.950042796023537_0.1778309603009678.pt',64),
    ("Yahoo","VAE"): ('LXR_Yahoo_VAE_neg-1.5pos_combined_19_26_128_18.958765029913238_4.92235962483309.pt',128),
    ("Yahoo","MLP"):('LXR_Yahoo_MLP_neg-pos_combined_last_29_37_128_12.40692505393434_0.19367009952856118.pt',128),
    ("Yahoo","NCF"): ('LXR_Yahoo_NCF_neg-pos_combined_loss_14_14_32_16.01464392466348_6.880015038643981.pt', 32),
    ("Pinterest","VAE"): ('LXR_Pinterest_VAE_comb_4_27_32_6.3443735346179855_1.472868807603448.pt',32),
    ("Pinterest","MLP"): ('LXR_Pinterest_MLP_0_5_16_10.059416809308486_0.705778173474644.pt',16),
    ("Pinterest","NCF"): ('LXR_Pinterest_NCF_combined__neg-1.5pos_0_26_32_13.02585523498726_12.8447247971534.pt', 32)
}

# # Help Functions

# In[2]:


# a function that samples different train data variation for a diverse training
def sample_indices(data, **kw):
    num_items = kw['num_items']
    pop_array = kw['pop_array']
    
    matrix = np.array(data)[:,:num_items] # keep only items columns, remove demographic features columns
    zero_indices = []
    one_indices = []

    for row in matrix:
        zero_idx = np.where(row == 0)[0]
        one_idx = np.where(row == 1)[0]
        probs = pop_array[zero_idx]
        probs = probs/ np.sum(probs)

        sampled_zero = np.random.choice(zero_idx, p = probs) # sample negative interactions according to items popularity 
        zero_indices.append(sampled_zero)

        sampled_one = np.random.choice(one_idx) # sample positive interactions from user's history
        data.iloc[row, sampled_one] = 0
        one_indices.append(sampled_one)

    data['pos'] = one_indices
    data['neg'] = zero_indices
    return np.array(data)


# In[ ]:


# a function that returns a specific item's rank in user's recommendations list
def get_index_in_the_list(user_tensor, original_user_tensor, item_id, recommender, **kw):
    top_k_list = list(get_top_k(user_tensor, original_user_tensor, recommender, **kw).keys())
    return top_k_list.index(item_id)


# In[ ]:


# returns a dictionary of items and recommendations scores for a specific user
def get_top_k(user_tensor, original_user_tensor, model, **kw):
    all_items_tensor = kw['all_items_tensor']
    num_items = kw['num_items']
    
    item_prob_dict = {}
    output_model = [float(i) for i in recommender_run(user_tensor, model, all_items_tensor, None, 'vector', **kw).cpu().detach().numpy()]
    original_user_vector = np.array(original_user_tensor.cpu())[:num_items]
    catalog = np.ones_like(original_user_vector)- original_user_vector
    output = catalog*output_model
    for i in range(len(output)):
        if catalog[i] > 0:
            item_prob_dict[i]=output[i]
    sorted_items_by_prob  = sorted(item_prob_dict.items(), key=lambda item: item[1],reverse=True)
    return dict(sorted_items_by_prob)


# In[ ]:


# a function that wraps the different recommenders types 
# returns user's scores with respect to a certain item or for all items 
def recommender_run(user_tensor, recommender, item_tensor=None, item_id=None, wanted_output='single', **kw):
    device = kw['device']
    output_type = kw['output_type']
    user_tensor = user_tensor.to(device)
    if item_tensor is not None:
        item_tensor = item_tensor.to(device)
    
    if output_type == 'single':
        if wanted_output == 'single':
            return recommender(user_tensor, item_tensor)
        else:
            return recommender(user_tensor, item_tensor).squeeze()
    else:
        if wanted_output == 'single':
            return recommender(user_tensor).squeeze()[item_id]
        else:
            return recommender(user_tensor).squeeze()


# In[ ]:


def recommender_evaluations(recommender, **kw):
    static_test_data = kw['static_test_data'].copy()
    device = kw['device']
    items_array = kw['items_array']
    num_items = kw['num_items']
    
    # Debug prints removed
    # print(f"Debug shapes in recommender_evaluations:")
    # print(f"static_test_data shape: {static_test_data.shape}")
    # print(f"items_array shape: {items_array.shape}")
    # print(f"num_items: {num_items}")
    
    counter_10 = 0
    counter_50 = 0
    counter_100 = 0
    RR = 0
    PR = 0
    
    temp_test_array = np.array(static_test_data)
    n = temp_test_array.shape[0]
    
    for i in range(n):
        item_id = temp_test_array[i][-2]
        user_tensor = torch.Tensor(temp_test_array[i][:-2]).to(device)
        
        if user_tensor.shape[0] != num_items:
            print(f"Warning: user_tensor shape {user_tensor.shape} doesn't match num_items {num_items}")
            continue
            
        user_tensor[item_id] = 0
        
        if isinstance(recommender, VAE):
            predictions = recommender(user_tensor)
            sorted_indices = torch.argsort(predictions, descending=True)
            index = (sorted_indices == item_id).nonzero().item() + 1
        else:
            index = get_index_in_the_list(user_tensor, user_tensor, item_id, recommender, **kw) + 1
            
        if index <= 10:
            counter_10 += 1
        if index <= 50:
            counter_50 += 1
        if index <= 100:
            counter_100 += 1
        RR += np.reciprocal(index)
        PR += index/num_items
        
    return counter_10/n, counter_50/n, counter_100/n, RR/n, PR*100/n


# In[ ]:


# get user's top recommended item
def get_user_recommended_item(user_tensor, recommender, **kw):
    all_items_tensor = kw['all_items_tensor']
    num_items = kw['num_items']
    user_res = recommender_run(user_tensor, recommender, all_items_tensor, None, 'vector', **kw)[:num_items]
    user_tensor = user_tensor[:num_items]
    user_catalog = torch.ones_like(user_tensor)-user_tensor
    user_recommenations = torch.mul(user_res, user_catalog)
    return(torch.argmax(user_recommenations))


# In[ ]:


# calculate the ndcg score of the restored recommendations list after perturbating the user's data.
def get_ndcg(ranked_list, target_item, **kw):
    device = kw['device']
    if target_item not in ranked_list:
        return 0.0

    target_idx = torch.tensor(ranked_list.index(target_item), device=device)
    dcg = torch.reciprocal(torch.log2(target_idx + 2))

    return dcg.item()


# In[ ]:


def get_ndcg_negative(ranked_list, target_item, **kw):
    device = kw['device']
    if target_item not in ranked_list:
        return 1.0  # Best case for negative item
    target_idx = ranked_list.index(target_item)
    dcg = 1 / torch.log2(torch.tensor(len(ranked_list) - target_idx + 1, dtype=torch.float, device=device))
    return dcg.item()


# In[ ]:


# from .recommenders_architecture import MLP, VAE, NCF, MLP_model, GMF_model # REMOVE DUPLICATE IMPORT
# importlib.reload(ipynb.fs.defs.recommenders_architecture) # Removed
# from ipynb.fs.defs.recommenders_architecture import * # Removed


# In[ ]:


def load_recommender(data_name, hidden_dim, checkpoints_path, recommender_path_filename, **kw_dict): # recommender_path is now just filename
    recommender_name = kw_dict['recommender_name']
    device = kw_dict['device']
    
    # Создаем модель в зависимости от типа
    if recommender_name == 'MLP':
        recommender = MLP(hidden_dim, **kw_dict)
    elif recommender_name == 'VAE':
        if data_name == "Pinterest":
            config_to_use = PINTEREST_VAE_CONFIG
        else:
            config_to_use = VAE_BASE_CONFIG
        recommender = VAE(config_to_use, **kw_dict)
    elif recommender_name == 'NCF':
        MLP_temp = MLP_model(hidden_size=hidden_dim, num_layers=3, **kw_dict)
        GMF_temp = GMF_model(hidden_size=hidden_dim, **kw_dict)
        recommender = NCF(factor_num=hidden_dim, num_layers=3, dropout=0.5, 
                         model='NeuMF-pre', GMF_model=GMF_temp, 
                         MLP_model=MLP_temp, **kw_dict)
    else:
        raise ValueError(f"Unknown recommender type: {recommender_name}")
    
    # Загружаем веса модели
    full_recommender_path = Path(checkpoints_path, recommender_path_filename) # Construct full path here
    recommender_checkpoint = torch.load(full_recommender_path, 
                                      map_location=device)
    recommender.load_state_dict(recommender_checkpoint)
    recommender.to(device)
    recommender.eval()
    
    # Отключаем градиенты
    for param in recommender.parameters():
        param.requires_grad = False
        
    return recommender


# In[ ]:


# metrics calculations (will be used in all metrics notebooks)
def single_user_metrics(user_vector, user_tensor, item_id, item_tensor, num_of_bins, recommender_model, expl_dict, **kw_dict):
    device = kw_dict['device']
    
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


# ## LXR Related

# In[ ]:


class LXR_loss(nn.Module):
    def __init__(self, lambda_pos, lambda_neg, alpha):
        super(LXR_loss, self).__init__()
        
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.alpha = alpha
        
        
    def forward(self, user_tensors, items_tensors, items_ids, pos_masks):
        neg_masks = torch.sub(torch.ones_like(pos_masks), pos_masks)
        x_masked_pos = user_tensors * pos_masks
        x_masked_neg = user_tensors * neg_masks
        if output_type=='single':
            x_masked_res_pos = recommender_run(x_masked_pos, recommender, items_tensors, item_id=items_ids, wanted_output = 'single', **kw_dict)
            x_masked_res_neg = recommender_run(x_masked_neg, recommender, items_tensors, item_id=items_ids, wanted_output = 'single', **kw_dict)
        else:
            x_masked_res_pos_before = recommender_run(x_masked_pos, recommender, items_tensors, item_id=items_ids, wanted_output = 'vector', **kw_dict)
            x_masked_res_neg_before = recommender_run(x_masked_neg, recommender, items_tensors, item_id=items_ids, wanted_output = 'vector', **kw_dict)
            rows=torch.arange(len(items_ids))
            x_masked_res_pos = x_masked_res_pos_before[rows, items_ids] 
            x_masked_res_neg = x_masked_res_neg_before[rows, items_ids] 
        
            
        pos_loss = -torch.mean(torch.log(x_masked_res_pos))
        neg_loss = torch.mean(torch.log(x_masked_res_neg))
        l1 = x_masked_pos[x_masked_pos>0].mean()
        combined_loss = self.lambda_pos*pos_loss + self.lambda_neg*neg_loss + self.alpha*l1
        return combined_loss, pos_loss, neg_loss, l1


# New function for SHAP experiment setup
def setup_shap_experiment_data(data_name_input, recommender_name_input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    current_script_path = Path(os.getcwd()) # Assuming scripts are in bauc/code/
    base_project_path = current_script_path.parent # path to bauc/

    dp_dir_name = Path("processed_data", data_name_input)
    files_path = Path(base_project_path, dp_dir_name)
    checkpoints_path = Path(base_project_path, "checkpoints")

    # Load dataframes and convert to numpy arrays
    train_data_df = pd.read_csv(Path(files_path, f'train_data_{data_name_input}.csv'), index_col=0)
    test_data_df = pd.read_csv(Path(files_path, f'test_data_{data_name_input}.csv'), index_col=0)
    train_array = train_data_df.to_numpy()
    test_array = test_data_df.to_numpy()

    with open(Path(files_path, f'pop_dict_{data_name_input}.pkl'), 'rb') as f:
        pop_dict_loaded = pickle.load(f)
    
    num_items_val = NUM_ITEMS_DICT[data_name_input]
    pop_array = np.zeros(num_items_val)
    for key, value in pop_dict_loaded.items():
        # Ensure key is treated as an integer for indexing
        key_int = int(key)
        if key_int < num_items_val: # Basic bounds check
            pop_array[key_int] = value

    static_test_data_df = pd.read_csv(Path(files_path, f'static_test_data_{data_name_input}.csv'), index_col=0)

    output_type = OUTPUT_TYPE_DICT[recommender_name_input]
    items_array = np.eye(num_items_val)
    all_items_tensor = torch.Tensor(items_array).to(device)

    hidden_dim = HIDDEN_DIM_DICT.get((data_name_input, recommender_name_input)) # Use .get for safety if a combo is missing
    recommender_path_filename = RECOMMENDER_PATH_DICT.get((data_name_input, recommender_name_input))
    
    if recommender_path_filename is None:
        raise ValueError(f"Recommender path not found for {data_name_input}, {recommender_name_input}")

    kw_dict = {
        'device': device,
        'num_items': num_items_val,
        'num_features': num_items_val, 
        'demographic': False, 
        'pop_array': pop_array,
        'all_items_tensor': all_items_tensor,
        'static_test_data': static_test_data_df, 
        'items_array': items_array,
        'output_type': output_type,
        'recommender_name': recommender_name_input
    }

    recommender_model = load_recommender(data_name_input, hidden_dim, checkpoints_path, recommender_path_filename, **kw_dict)

    top1_train_path = Path(files_path, f'top1_train_{data_name_input}_{recommender_name_input}.pkl')
    top1_test_path = Path(files_path, f'top1_test_{data_name_input}_{recommender_name_input}.pkl')
    
    # Forcibly recreate dictionaries as per original SHAP scripts logic
    top1_train = {}
    for i in range(train_array.shape[0]):
        user_idx_original = int(train_data_df.index[i])
        user_tensor = torch.Tensor(train_array[i]).to(device)
        top1_train[user_idx_original] = int(get_user_recommended_item(user_tensor, recommender_model, **kw_dict))
    
    with open(top1_train_path, 'wb') as f:
        pickle.dump(top1_train, f)

    top1_test = {}
    for i in range(test_array.shape[0]):
        user_idx_original = int(test_data_df.index[i])
        user_tensor = torch.Tensor(test_array[i]).to(device)
        top1_test[user_idx_original] = int(get_user_recommended_item(user_tensor, recommender_model, **kw_dict))
            
    with open(top1_test_path, 'wb') as f:
        pickle.dump(top1_test, f)

    return (
        files_path, checkpoints_path, device, 
        kw_dict, train_array, test_array, items_array, pop_array, static_test_data_df,
        recommender_model, top1_train, top1_test, train_data_df, test_data_df, num_items_val
    )


def load_lxr_explainer(
    data_name_str, 
    recommender_name_str, 
    checkpoints_path_global,    
    num_items_val,              
    device_val                  
):
    if not isinstance(checkpoints_path_global, Path):
        checkpoints_path_global = Path(checkpoints_path_global)
        
    # Use the global LXR_CHECKPOINT_DICT from this module
    lxr_filename, lxr_hidden_dim = LXR_CHECKPOINT_DICT[(data_name_str, recommender_name_str)]
    
    explainer_model = Explainer(num_items_val, num_items_val, lxr_hidden_dim, device_val)
    
    actual_checkpoint_path = checkpoints_path_global / lxr_filename
    # Ensure checkpoint is loaded to the correct device, especially if trained on GPU and loading on CPU
    lxr_checkpoint_state = torch.load(actual_checkpoint_path, map_location=device_val)
    explainer_model.load_state_dict(lxr_checkpoint_state)
    explainer_model.eval()
    for param in explainer_model.parameters():
        param.requires_grad = False
    return explainer_model


def find_lxr_mask(user_history_tensor, item_target_tensor, lxr_explainer_model_instance):
    """
    Generates an explanation mask using a trained LXR explainer model.

    Args:
        user_history_tensor (torch.Tensor): Tensor representing the user's interaction history.
        item_target_tensor (torch.Tensor): Tensor representing the target item.
        lxr_explainer_model_instance (torch.nn.Module): Trained LXR explainer model instance.

    Returns:
        dict: A dictionary mapping item_id to its explanation score.
              Scores are the product of the user's history and the explainer's output.
    """
    expl_scores = lxr_explainer_model_instance(user_history_tensor, item_target_tensor)
    x_masked = user_history_tensor * expl_scores  # Element-wise product
    
    item_sim_dict = {}
    for i in range(x_masked.shape[0]):  # Iterate over all items
        score_val = x_masked[i].item()
        item_sim_dict[i] = score_val
        
    return item_sim_dict

