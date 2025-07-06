import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pathlib import Path

from .models import VAE, MLP, NCF, MLP_model, GMF_model

def sample_indices(data, **kw):
    num_items = kw['num_items']
    pop_array = kw['pop_array']
    
    matrix = np.array(data)[:,:num_items] # keep only items columns, remove demographic features columns
    zero_indices = []
    one_indices = []

    for i, row in enumerate(matrix):
        zero_idx = np.where(row == 0)[0]
        one_idx = np.where(row == 1)[0]
        
        if len(one_idx) == 0:
            continue

        probs = pop_array[zero_idx]
        probs = probs / np.sum(probs)

        sampled_zero = np.random.choice(zero_idx, p=probs) # sample negative interactions according to items popularity 
        zero_indices.append(sampled_zero)

        sampled_one = np.random.choice(one_idx) # sample positive interactions from user's history
        
        # Create a copy to avoid SettingWithCopyWarning
        data_copy = data.copy()
        data_copy.iloc[i, sampled_one] = 0
        data = data_copy

        one_indices.append(sampled_one)

    data['pos'] = one_indices
    data['neg'] = zero_indices
    return np.array(data)

def get_index_in_the_list(user_tensor, original_user_tensor, item_id, recommender, **kw):
    top_k_list = list(get_top_k(user_tensor, original_user_tensor, recommender, **kw).keys())
    return top_k_list.index(item_id)

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

def recommender_evaluations(recommender, **kw):
    static_test_data = kw['static_test_data'].copy()
    device = kw['device']
    items_array = kw['items_array']
    num_items = kw['num_items']
    
    print(f"Debug shapes in recommender_evaluations:")
    print(f"static_test_data shape: {static_test_data.shape}")
    print(f"items_array shape: {items_array.shape}")
    print(f"num_items: {num_items}")
    
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

def get_user_recommended_item(user_tensor, recommender, **kw):
    all_items_tensor = kw['all_items_tensor']
    num_items = kw['num_items']
    user_res = recommender_run(user_tensor, recommender, all_items_tensor, None, 'vector', **kw)[:num_items]
    user_tensor = user_tensor[:num_items]
    user_catalog = torch.ones_like(user_tensor)-user_tensor
    user_recommenations = torch.mul(user_res, user_catalog)
    return(torch.argmax(user_recommenations))

def get_ndcg(ranked_list, target_item, **kw):
    device = kw['device']
    if target_item not in ranked_list:
        return 0.0

    target_idx = torch.tensor(ranked_list.index(target_item), device=device)
    dcg = torch.reciprocal(torch.log2(target_idx + 2))

    return dcg.item()

def get_ndcg_negative(ranked_list, target_item, **kw):
    device = kw['device']
    if target_item not in ranked_list:
        return 1.0  # Best case for negative item
    target_idx = ranked_list.index(target_item)
    dcg = 1 / torch.log2(torch.tensor(len(ranked_list) - target_idx + 1, dtype=torch.float, device=device))
    return dcg.item()

def load_recommender(data_name, hidden_dim, recommender_path, **kw_dict):
    recommender_name = kw_dict['recommender_name']
    device = kw_dict['device']
    
    # Создаем модель в зависимости от типа
    if recommender_name == 'MLP':
        recommender = MLP(hidden_dim, **kw_dict)
    elif recommender_name == 'VAE':
        VAE_config = {
            "enc_dims": [512,128],
            "dropout": 0.5,
            "anneal_cap": 0.2,
            "total_anneal_steps": 200000
        }
        Pinterest_VAE_config = {
            "enc_dims": [256,64],
            "dropout": 0.5,
            "anneal_cap": 0.2,
            "total_anneal_steps": 200000
        }
        
        if data_name == "Pinterest":
            recommender = VAE(Pinterest_VAE_config, **kw_dict)
        else:
            recommender = VAE(VAE_config, **kw_dict)
    elif recommender_name == 'NCF':
        MLP_temp = MLP_model(hidden_size=hidden_dim, num_layers=3, **kw_dict)
        GMF_temp = GMF_model(hidden_size=hidden_dim, **kw_dict)
        recommender = NCF(factor_num=hidden_dim, num_layers=3, dropout=0.5, 
                         model='NeuMF-pre', GMF_model=GMF_temp, 
                         MLP_model=MLP_temp, **kw_dict)
    else:
        raise ValueError(f"Unknown recommender type: {recommender_name}")
    
    # Загружаем веса модели
    recommender_checkpoint = torch.load(recommender_path,
                                      map_location=device)
    recommender.load_state_dict(recommender_checkpoint)
    recommender.to(device)
    recommender.eval()
    
    # Отключаем градиенты
    for param in recommender.parameters():
        param.requires_grad = False
        
    return recommender

def single_user_metrics(user_vector, user_tensor, item_id, item_tensor, num_of_bins, recommender_model, expl_dict, **kw_dict):
    device = kw_dict['device']
    num_items = kw_dict['num_items']
    
    POS_masked = user_tensor.clone()
    NEG_masked = user_tensor.clone()
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
            
        POS_masked_loop = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        
        for j in POS_sim_items[:total_items]:
            POS_masked_loop[j[0]] = 1
        POS_masked_loop = user_tensor - POS_masked_loop # remove the masked items from the user history

        NEG_masked_loop = torch.zeros_like(user_tensor, dtype=torch.float32, device=device)
        for j in NEG_sim_items[:total_items]:
            NEG_masked_loop[j[0]] = 1
        NEG_masked_loop = user_tensor - NEG_masked_loop # remove the masked items from the user history 
        
        POS_ranked_list = get_top_k(POS_masked_loop, user_tensor, recommender_model, **kw_dict)
        
        if item_id in list(POS_ranked_list.keys()):
            POS_index = list(POS_ranked_list.keys()).index(item_id)+1
        else:
            POS_index = num_items
        NEG_index = get_index_in_the_list(NEG_masked_loop, user_tensor, item_id, recommender_model, **kw_dict)+1

        # for pos:
        POS_at_5[i] = 1 if POS_index <=5 else 0
        POS_at_10[i] = 1 if POS_index <=10 else 0
        POS_at_20[i] = 1 if POS_index <=20 else 0

        # for del:
        DEL[i] = float(recommender_run(POS_masked_loop, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())

        # for ins:
        INS[i] = float(recommender_run(user_tensor-POS_masked_loop, recommender_model, item_tensor, item_id, **kw_dict).detach().cpu().numpy())

        #for NDCG:
        NDCG[i]= get_ndcg(list(POS_ranked_list.keys()),item_id, **kw_dict)
        
    res = [DEL, INS, NDCG, POS_at_5, POS_at_10, POS_at_20]
    for i in range(len(res)):
        res[i] = np.array(res[i])
        
    return res

class LXR_loss(nn.Module):
    def __init__(self, lambda_pos, lambda_neg, alpha):
        super(LXR_loss, self).__init__()
        
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.alpha = alpha
        
        
    def forward(self, user_tensors, items_tensors, items_ids, pos_masks, recommender, **kw_dict):
        output_type = kw_dict['output_type']
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