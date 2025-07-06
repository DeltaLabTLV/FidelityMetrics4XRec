import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import get_top_k, recommender_run
from src.lime import LimeBase, get_lime_args, get_lire_args, distance_to_proximity

class Explainer(nn.Module):
    def __init__(self, user_size, item_size, hidden_size, device):
        super(Explainer, self).__init__()
        self.device = device
        self.users_fc = nn.Linear(in_features=user_size, out_features=hidden_size).to(device)
        self.items_fc = nn.Linear(in_features=item_size, out_features=hidden_size).to(device)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size).to(device),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=user_size).to(device),
            nn.Sigmoid()
        ).to(device)

    def forward(self, user_tensor, item_tensor):
        user_output = self.users_fc(user_tensor.float())
        item_output = self.items_fc(item_tensor.float())
        combined_output = torch.cat((user_output, item_output), dim=-1)
        expl_scores = self.bottleneck(combined_output).to(self.device)
        return expl_scores

def load_explainer(LXR_checkpoint_dict, data_name, recommender_name, checkpoints_path, num_items, num_features, device):
    lxr_path, lxr_dim = LXR_checkpoint_dict[(data_name, recommender_name)]
    explainer = Explainer(num_features, num_items, lxr_dim, device)
    lxr_checkpoint = torch.load(Path(checkpoints_path, lxr_path))
    explainer.load_state_dict(lxr_checkpoint)
    explainer.eval()
    for param in explainer.parameters():
        param.requires_grad = False
    return explainer

def find_pop_mask(x, item_id, pop_array, **kw):
    user_hist = torch.Tensor(x).to(kw['device'])
    user_hist[item_id] = 0
    item_pop_dict = {}
    for i, j in enumerate(user_hist > 0):
        if j:
            item_pop_dict[i] = pop_array[i]
    return item_pop_dict

def find_jaccard_mask(x, item_id, user_based_Jaccard_sim, **kw):
    user_hist = x.copy()
    user_hist[item_id] = 0
    item_jaccard_dict = {}
    for i, j in enumerate(user_hist > 0):
        if j:
            if (i, item_id) in user_based_Jaccard_sim:
                item_jaccard_dict[i] = user_based_Jaccard_sim[(i, item_id)]
            else:
                item_jaccard_dict[i] = 0
    return item_jaccard_dict

def find_cosine_mask(x, item_id, item_cosine, **kw):
    user_hist = x.copy()
    user_hist[item_id] = 0
    item_cosine_dict = {}
    for i, j in enumerate(user_hist > 0):
        if j:
            if (i, item_id) in item_cosine:
                item_cosine_dict[i] = item_cosine[(i, item_id)]
            else:
                item_cosine_dict[i] = 0
    return item_cosine_dict

def find_lime_mask(x, item_id, recommender, item_tensor, **kw_dict):
    user_hist = x.clone()
    user_hist[item_id] = 0
    lime = LimeBase(distance_to_proximity)
    neighborhood_data, neighborhood_labels, distances, item_id = get_lime_args(user_hist, item_id, recommender, item_tensor, **kw_dict)

    # Request a larger number of features to get a good pool for filtering
    explanation_unfiltered = lime.explain_instance_with_data(
        neighborhood_data, neighborhood_labels, distances, item_id,
        200, feature_selection='highest_weights', pos_neg='POS'
    )

    # Filter the explanation to only include items from the user's original history
    user_hist_mask = (user_hist > 0).cpu().numpy()
    filtered_explanation = [item for item in explanation_unfiltered if user_hist_mask[item[0]]]

    return filtered_explanation

def find_lire_mask(x, item_id, recommender, **kw_dict):
    user_hist = x.copy()
    user_hist[item_id] = 0
    lime = LimeBase(distance_to_proximity)
    neighborhood_data, neighborhood_labels, distances, item_id = get_lire_args(user_hist, item_id, recommender, **kw_dict)
    most_pop_items = lime.explain_instance_with_data(
        neighborhood_data, neighborhood_labels, distances, item_id,
        200, feature_selection='highest_weights', pos_neg='POS'
    )
    return most_pop_items

def find_lxr_mask(x, item_tensor, explainer, **kw):
    user_hist = x
    if isinstance(user_hist, np.ndarray):
        user_hist = torch.FloatTensor(user_hist)
    if isinstance(item_tensor, np.ndarray):
        item_tensor = torch.FloatTensor(item_tensor)
    if isinstance(item_tensor, int):
        # Convert to one-hot tensor
        num_items = user_hist.shape[0] if hasattr(user_hist, 'shape') else len(user_hist)
        one_hot = torch.zeros(num_items)
        one_hot[item_tensor] = 1.0
        item_tensor = one_hot
    expl_scores = explainer(user_hist, item_tensor)
    # Ensure both tensors are on the same device
    user_hist = user_hist.to(expl_scores.device)
    x_masked = user_hist * expl_scores
    item_sim_dict = {}
    for i, j in enumerate(x_masked != 0):
        if j:
            item_sim_dict[i] = x_masked[i].item()
    return item_sim_dict

def find_fia_mask(user_tensor, item_tensor, item_id, recommender, **kw_dict):
    y_pred = recommender_run(user_tensor, recommender, item_tensor, item_id, **kw_dict).to(kw_dict['device'])
    items_fia = {}
    user_hist = user_tensor.cpu().detach().numpy().astype(int)
    num_items = kw_dict['num_items']

    for i in range(num_items):
        if (user_hist[i] == 1):
            user_hist[i] = 0
            temp_user_tensor = torch.FloatTensor(user_hist).to(kw_dict['device'])
            y_pred_without_item = recommender_run(temp_user_tensor, recommender, item_tensor, item_id, 'single', **kw_dict).to(kw_dict['device'])
            infl_score = y_pred - y_pred_without_item
            items_fia[i] = infl_score.item()
            user_hist[i] = 1
    return items_fia

def find_shapley_mask(user_tensor, user_id, model, shap_values, item_to_cluster, **kw):
    item_shap = {}
    shapley_values_user = shap_values[shap_values[:, 0].astype(int) == user_id][:, 1:]
    user_vector = user_tensor.cpu().detach().numpy().astype(int)

    for i in np.where(user_vector.astype(int) == 1)[0]:
        items_cluster = item_to_cluster[i]
        item_shap[i] = shapley_values_user.T[int(items_cluster)][0]
    return item_shap

def find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model, top_k, **kw_dict):
    items_accent = defaultdict(float)
    factor = top_k - 1
    user_accent_hist = user_tensor.cpu().detach().numpy().astype(int)
    items_array = kw_dict['items_array']

    sorted_indices = list(get_top_k(user_tensor, user_tensor, recommender_model, **kw_dict).keys())
    
    if top_k == 1:
        top_k_indices = [sorted_indices[0]]
    else:
        top_k_indices = sorted_indices[:top_k]

    for iteration, item_k_id in enumerate(top_k_indices):
        user_accent_hist[item_k_id] = 0
        temp_user_tensor = torch.FloatTensor(user_accent_hist).to(kw_dict['device'])
        
        item_vector = items_array[item_k_id]
        temp_item_tensor = torch.FloatTensor(item_vector).to(kw_dict['device'])
              
        fia_dict = find_fia_mask(temp_user_tensor, temp_item_tensor, item_k_id, recommender_model, **kw_dict)
         
        if not iteration:
            for key, value in fia_dict.items():
                items_accent[key] = value * factor
        else:
            for key, value in fia_dict.items():
                items_accent[key] -= value
       
    for key in items_accent.keys():
        items_accent[key] *= -1    

    return items_accent

def single_user_expl(user_vector, user_tensor, item_id, item_tensor, num_items, recommender_model, user_id=None, mask_type=None, **kwargs):
    user_hist_size = np.sum(user_vector)

    if mask_type == 'lime':
        return find_lime_mask(user_vector, item_id, recommender_model, **kwargs)
    elif mask_type == 'lire':
        return find_lire_mask(user_vector, item_id, recommender_model, **kwargs)
    else:
        if mask_type == 'pop':
            sim_items = find_pop_mask(user_tensor, item_id, **kwargs)
        elif mask_type == 'jaccard':
            sim_items = find_jaccard_mask(user_tensor, item_id, **kwargs)
        elif mask_type == 'cosine':
            sim_items = find_cosine_mask(user_tensor, item_id, **kwargs)
        elif mask_type == 'shap':
            sim_items = find_shapley_mask(user_tensor, user_id, recommender_model, **kwargs)
        elif mask_type == 'accent':
            sim_items = find_accent_mask(user_tensor, user_id, item_tensor, item_id, recommender_model, 5, **kwargs)
        elif mask_type == 'lxr':
            explainer = load_explainer(**kwargs)
            sim_items = find_lxr_mask(user_tensor, item_tensor, explainer, **kwargs)
        
        return list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]