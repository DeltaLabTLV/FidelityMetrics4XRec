import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
from collections import defaultdict
import time
import torch
import torch.nn as nn
import sys
import shap
from sklearn.cluster import KMeans

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from src.models import MLP, VAE, NCF, GMF_model, MLP_model
from src.utils import get_user_recommended_item, recommender_run, get_top_k, get_index_in_the_list, get_ndcg
from src.lime import LimeBase, distance_to_proximity, get_lime_args, get_lire_args
from src.explainers import find_accent_mask, find_shapley_mask
from src.config import recommender_path_dict, hidden_dim_dict, LXR_checkpoint_dict, checkpoints_path

def load_recommender(recommender_name, data_name, kw_dict):
    hidden_dim = hidden_dim_dict[(data_name, recommender_name)]
    recommender_path = recommender_path_dict[(data_name, recommender_name)]

    VAE_config = {
        "enc_dims": [512, 128], "dropout": 0.5, "anneal_cap": 0.2, "total_anneal_steps": 200000
    }
    Pinterest_VAE_config = {
        "enc_dims": [256, 64], "dropout": 0.5, "anneal_cap": 0.2, "total_anneal_steps": 200000
    }

    if recommender_name == 'MLP':
        recommender = MLP(hidden_dim, **kw_dict)
    elif recommender_name == 'VAE':
        if data_name == "Pinterest":
            recommender = VAE(Pinterest_VAE_config, **kw_dict)
        else:
            recommender = VAE(VAE_config, **kw_dict)
    elif recommender_name == 'NCF':
        MLP_temp = MLP_model(hidden_size=hidden_dim, num_layers=3, **kw_dict)
        GMF_temp = GMF_model(hidden_size=hidden_dim, **kw_dict)
        recommender = NCF(factor_num=hidden_dim, num_layers=3, dropout=0.5, model='NeuMF-pre', GMF_model=GMF_temp, MLP_model=MLP_temp, **kw_dict)
    else:
        raise ValueError(f"Unknown recommender type: {recommender_name}")

    recommender_checkpoint = torch.load(recommender_path)
    recommender.load_state_dict(recommender_checkpoint)
    recommender.eval()
    for param in recommender.parameters():
        param.requires_grad = False
    return recommender

class MLPWrapper(MLP):
    def __init__(self, hidden_size, cluster_to_items, **kw):
        super().__init__(hidden_size=hidden_size, device=kw['device'], num_items=kw['num_items'])
        self.cluster_to_items = cluster_to_items
        self.items_array = kw['items_array']
        self.device = kw['device']
        self.num_items = kw['num_items']

    def preprocess(self, batch):
        items = batch[:, 0]
        clusters = batch[:, 1:]
        n_clusters = clusters.shape[1]
        items_tensor = torch.Tensor(self.items_array[items]).to(self.device)
        user_tensor = torch.zeros((len(batch), self.num_items), dtype=torch.float).to(self.device)
        for cluster in range(n_clusters):
            cluster_indices = torch.tensor(clusters[:, cluster], dtype=torch.float).to(self.device)
            user_tensor[:, self.cluster_to_items[cluster]] = cluster_indices.unsqueeze(1)
        return user_tensor, items_tensor

    def forward(self, batch):
        batch_size = 256
        outputs = []
        for i in range(0, len(batch), batch_size):
            mini_batch = batch[i:i+batch_size]
            user_tensor, items_tensor = self.preprocess(mini_batch)
            output = super().forward(user_tensor, items_tensor)
            outputs.append(torch.diag(output).detach().cpu().numpy())
        return np.concatenate(outputs)

class Explainer(nn.Module):
    def __init__(self, user_size, item_size, hidden_size, device):
        super(Explainer, self).__init__()
        self.users_fc = nn.Linear(in_features = user_size, out_features=hidden_size).to(device)
        self.items_fc = nn.Linear(in_features = item_size, out_features=hidden_size).to(device)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features = hidden_size*2, out_features=hidden_size).to(device),
            nn.Tanh(),
            nn.Linear(in_features = hidden_size, out_features=user_size).to(device),
            nn.Sigmoid()
        ).to(device)

    def forward(self, user_tensor, item_tensor):
        user_output = self.users_fc(user_tensor.float())
        item_output = self.items_fc(item_tensor.float())
        combined_output = torch.cat((user_output, item_output), dim=-1)
        expl_scores = self.bottleneck(combined_output)
        return expl_scores

def load_explainer(num_features, num_items, lxr_dim, lxr_path, device):
    explainer = Explainer(num_features, num_items, lxr_dim, device)
    lxr_checkpoint = torch.load(lxr_path)
    explainer.load_state_dict(lxr_checkpoint)
    explainer.eval()
    for param in explainer.parameters():
        param.requires_grad= False
    return explainer

def find_pop_mask(x, item_id, pop_array):
    user_hist = torch.Tensor(x)
    user_hist[item_id] = 0
    item_pop_dict = {}
    for i,j in enumerate(user_hist>0):
        if j:
            item_pop_dict[i]=pop_array[i]
    return item_pop_dict

def find_jaccard_mask(x, item_id, user_based_Jaccard_sim):
    user_hist = x
    user_hist[item_id] = 0
    item_jaccard_dict = {}
    for i,j in enumerate(user_hist>0):
        if j:
            if (i,item_id) in user_based_Jaccard_sim:
                item_jaccard_dict[i]=user_based_Jaccard_sim[(i,item_id)]
            else:
                item_jaccard_dict[i] = 0            
    return item_jaccard_dict

def find_cosine_mask(x, item_id, item_cosine):
    user_hist = x
    user_hist[item_id] = 0
    item_cosine_dict = {}
    for i,j in enumerate(user_hist>0):
        if j:
            if (i,item_id) in item_cosine:
                item_cosine_dict[i]=item_cosine[(i,item_id)]
            else:
                item_cosine_dict[i]=0
    return item_cosine_dict

def find_lime_mask(x, item_id, recommender, all_items_tensor, kw_dict, num_samples=10, method = 'POS'):
    # Ensure user_hist is a torch.Tensor for downstream .clone() usage
    if not isinstance(x, torch.Tensor):
        user_hist = torch.FloatTensor(x)
    else:
        user_hist = x.clone() if x.is_leaf else x
    user_hist[item_id] = 0

    lime = LimeBase(distance_to_proximity)
    lime.kernel_fn = distance_to_proximity  # ensure kernel is set
    neighborhood_data, neighborhood_labels, distances, item_id = get_lime_args(
        user_hist, item_id, recommender, all_items_tensor,
        min_pert=50, max_pert=100, num_of_perturbations=150, seed=item_id, **kw_dict)
    if method == 'POS':
        most_pop_items = lime.explain_instance_with_data(
            neighborhood_data, neighborhood_labels, distances, item_id, num_samples, 'highest_weights', pos_neg='POS')
    elif method == 'NEG':
        most_pop_items = lime.explain_instance_with_data(
            neighborhood_data, neighborhood_labels, distances, item_id, num_samples, 'highest_weights', pos_neg='NEG')
    else:
        most_pop_items = []
    return most_pop_items

def find_lxr_mask(x, item_tensor, explainer):
    user_hist = x
    expl_scores = explainer(user_hist, item_tensor)
    x_masked = user_hist*expl_scores
    item_sim_dict = {}
    for i,j in enumerate(x_masked.squeeze()):
        if j > 0:
            item_sim_dict[i]=x_masked.squeeze()[i].item()
    return item_sim_dict

def single_user_expl(user_vector, user_tensor, item_id, item_tensor, recommender_model, kw_dict, pop_array, jaccard_dict, cosine_dict, explainer, mask_type = None, user_id=None):
    user_hist_size = np.sum(user_vector)
    if mask_type == 'lime':
        sim_items = find_lime_mask(user_vector, item_id, recommender_model, kw_dict['all_items_tensor'], kw_dict, num_samples=user_hist_size)
    elif mask_type == 'pop':
        sim_items = find_pop_mask(user_tensor, item_id, pop_array)
    elif mask_type == 'jaccard':
        sim_items = find_jaccard_mask(user_tensor, item_id, jaccard_dict)
    elif mask_type == 'cosine':
        sim_items = find_cosine_mask(user_tensor, item_id, cosine_dict)
    elif mask_type == 'lxr':
        sim_items = find_lxr_mask(user_tensor, item_tensor, explainer)
    elif mask_type == 'accent':
        sim_items = find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model, 5, **kw_dict)
    elif mask_type == 'shap':
        sim_items = find_shapley_mask(user_tensor, user_id, recommender_model,
                                      shap_values=kw_dict['shap_values'],
                                      item_to_cluster=kw_dict['item_to_cluster'])

    if isinstance(sim_items, list):
        sorted_sim_items = sorted(sim_items, key=lambda item: item[1], reverse=True)
    else:
        sorted_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1],reverse=True))
    return sorted_sim_items

def main():
    parser = argparse.ArgumentParser(description='Create explanation dictionaries.')
    parser.add_argument('--dataset', type=str, default='ML1M', help='Dataset to use (ML1M, Yahoo, Pinterest)')
    parser.add_argument('--recommender', type=str, default='MLP', help='Recommender to use (MLP, VAE, NCF)')
    args = parser.parse_args()

    data_name = args.dataset
    recommender_name = args.recommender

    DP_DIR = Path("data/processed", data_name)
    files_path = Path(os.getcwd(), DP_DIR)
    # checkpoints_path is now imported from config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_type_dict = {"VAE":"multiple", "MLP":"single", "NCF": "single"}
    num_items_dict = {"ML1M":3381, "Yahoo":4604, "Pinterest":9362}
    
    # LXR_checkpoint_dict is now imported from config

    num_items = num_items_dict[data_name]
    lxr_path_name, lxr_dim = LXR_checkpoint_dict[(data_name, recommender_name)]
    lxr_path = Path(checkpoints_path, lxr_path_name)

    train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
    test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
    static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
    with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
        pop_dict = pickle.load(f)
    
    items_array = np.eye(num_items)
    all_items_tensor = torch.Tensor(items_array).to(device)
    test_array = static_test_data.iloc[:,:-2].to_numpy()

    with open(Path(files_path, f'jaccard_based_sim_{data_name}.pkl'), 'rb') as f:
        jaccard_dict = pickle.load(f) 
    with open(Path(files_path, f'cosine_based_sim_{data_name}.pkl'), 'rb') as f:
        cosine_dict = pickle.load(f) 

    for i in range(num_items):
        for j in range(i, num_items):
            jaccard_dict[(j,i)]= jaccard_dict.get((i,j), 0)
            cosine_dict[(j,i)]= cosine_dict.get((i,j), 0)

    pop_array = np.zeros(len(pop_dict))
    for key, value in pop_dict.items():
        if key < len(pop_array):
            pop_array[key] = value

    kw_dict = {'device':device, 'num_items': num_items, 'num_features': num_items, 
               'demographic':False, 'pop_array':pop_array, 'all_items_tensor':all_items_tensor,
               'static_test_data':static_test_data, 'items_array':items_array,
               'output_type':output_type_dict[recommender_name], 'recommender_name':recommender_name}

    recommender = load_recommender(recommender_name, data_name, kw_dict)
    explainer = load_explainer(num_items, num_items, lxr_dim, lxr_path, device)

    if recommender_name == 'MLP':
        print("Clustering for SHAP...")
        u_train = torch.tensor(train_data.to_numpy()).float()
        kmeans = KMeans(n_clusters=10)
        item_clusters = kmeans.fit_predict(np.transpose(u_train))
        item_to_cluster = {}
        cluster_to_items = {}
        for i, cluster in enumerate(item_clusters):
            item_to_cluster[i] = cluster
            if cluster not in cluster_to_items:
                cluster_to_items[cluster] = []
            cluster_to_items[cluster].append(i)
        
        with open(Path(files_path, f'item_to_cluster_{recommender_name}_{data_name}.pkl'), 'wb') as f:
            pickle.dump(item_to_cluster, f)

        print("Calculating SHAP values...")
        u_test = torch.tensor(test_array).float()
        user_to_clusters = np.zeros((u_test.shape[0], 10))
        for i in cluster_to_items.keys():
            user_to_clusters[:, i] = np.sum(u_test.cpu().detach().numpy().T[cluster_to_items[i]], axis=0)
        user_to_clusters_bin = np.where(user_to_clusters > 0, 1, 0)
        
        top1_test = {}
        for i in range(test_array.shape[0]):
            user_index = int(test_data.index[i])
            user_tensor = torch.Tensor(test_array[i]).to(device)
            top1_test[user_index] = int(get_user_recommended_item(user_tensor, recommender, **kw_dict))
        
        col2 = list(top1_test.values())
        input_test_array = np.insert(user_to_clusters_bin, 0, col2, axis=1).astype(int)
        
        wrap_model = MLPWrapper(hidden_dim_dict[(data_name, recommender_name)], cluster_to_items, **kw_dict)
        
        u_train_numpy = train_data.to_numpy()
        user_to_clusters_train = np.zeros((u_train_numpy.shape[0], 10))
        for i in cluster_to_items.keys():
            user_to_clusters_train[:, i] = np.sum(u_train_numpy.T[cluster_to_items[i]], axis=0)
        user_to_clusters_train_bin = np.where(user_to_clusters_train > 0, 1, 0)
        
        top1_train = {}
        for i in range(train_data.shape[0]):
            user_index = int(train_data.index[i])
            user_tensor = torch.Tensor(train_data.iloc[i].to_numpy()).to(device)
            top1_train[user_index] = int(get_user_recommended_item(user_tensor, recommender, **kw_dict))
            
        col2_train = list(top1_train.values())
        input_train_array = np.insert(user_to_clusters_train_bin, 0, col2_train, axis=1).astype(int)
        
        sampled_subset = shap.sample(input_train_array, 50)
        explainer_shap = shap.KernelExplainer(wrap_model.forward, sampled_subset)
        shap_values_test = explainer_shap.shap_values(input_test_array)
        
        row_test_indices = np.arange(test_array.shape[0]) + train_data.shape[0]
        col1 = row_test_indices
        input_test_array_with_ids = np.insert(shap_values_test[:, 1:], 0, col1, axis=1)
        
        with open(Path(files_path, f'shap_values_{recommender_name}_{data_name}.pkl'), 'wb') as f:
            pickle.dump(input_test_array_with_ids, f)
        
        kw_dict['shap_values'] = input_test_array_with_ids
        kw_dict['item_to_cluster'] = item_to_cluster

    explanation_dictionaries = {
        'pop': {},
        'jaccard': {},
        'cosine': {},
        'lime': {},
        'lxr': {},
        'accent': {},
        'shap': {}
    }

    with torch.no_grad():
        for i in range(test_array.shape[0]):
            if i % 100 == 0:
                print(f"Processing user {i}/{test_array.shape[0]}")

            user_vector = test_array[i]
            user_tensor = torch.FloatTensor(user_vector).to(device)
            user_id = int(test_data.index[i])

            item_id = int(get_user_recommended_item(user_tensor, recommender, **kw_dict).detach().cpu().numpy())
            item_vector = items_array[item_id]
            item_tensor = torch.FloatTensor(item_vector).to(device)

            user_vector[item_id] = 0
            user_tensor[item_id] = 0

            recommender.to(device)

            for mask_type in explanation_dictionaries.keys():
                if mask_type == 'shap' and recommender_name != 'MLP':
                    continue
                explanation_dictionaries[mask_type][user_id] = single_user_expl(
                    user_vector, user_tensor, item_id, item_tensor, recommender, kw_dict,
                    pop_array, jaccard_dict, cosine_dict, explainer, mask_type=mask_type, user_id=user_id
                )

    print("Saving dictionaries...")
    output_path = Path(files_path, f'{recommender_name}_explanation_dicts.pkl')
    with open(output_path, 'wb') as handle:
        pickle.dump(explanation_dictionaries, handle)
    
    print("Dictionaries created successfully.")

if __name__ == '__main__':
    main()