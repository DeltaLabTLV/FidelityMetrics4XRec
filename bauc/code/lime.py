#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import pandas as pd # Removed, unused
import numpy as np
import torch
# import torch.nn as nn # Removed, nn components not directly used in this file
# from torch.utils.data import Dataset # Removed, unused
# from torch.utils.data import DataLoader # Removed, unused
# from torch.nn import Linear # Removed, unused
# from torch.nn import functional as F # Removed, unused
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from .help_functions import recommender_run


# In[2]:


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     max_iter=15,
                                     eps = 2.220446049250313e-7,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)                
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weights = np.asarray(weights)
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances_list,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None,
                                   pos_neg = 'POS'):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances_list)
        labels_column = neighborhood_labels[:,label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        if pos_neg =='POS':
            return sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: x[1], reverse=True)
        elif pos_neg =='NEG':
            return sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: x[1], reverse=False)
        elif pos_neg == 'ABS':
            return sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=False)
        else: 
            return('Unfamiliar method')


# In[4]:


def distance_to_proximity(distances_list):
    return [1-distances_list[i]/sum(distances_list) for i in range(len(distances_list))]


# In[5]:


def gaussian_kernel(distances, sigma=1):
    kernel = [np.exp(-distances[i]**2 / (2 * sigma**2)) for i in range(len(distances))]
    return kernel


# In[9]:


# preturbate user:
# choose random seed
# for number of  perturbations
    # choose random number of perturbations
        # choose randomly slots in user vector and change the slot value 
    # save the perturbated user and the distance (=number of perturbations) and the model's lables for this user


def get_lime_args(user_vec, item_id, model, item_tensor_all_items, min_pert = 10, max_pert = 20, num_of_perturbations = 5, seed = 0, **kw_dict_passed):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use device from kw_dict
    device = kw_dict_passed['device']
    output_type = kw_dict_passed['output_type'] # output_type is used by recommender_run
    
    # user_vec is a numpy array. Create a copy to modify.
    current_user_vec = user_vec.copy()
    current_user_vec[item_id] = 0 # Mask item to explain in the base vector for perturbations
    
    neighborhood_data = [current_user_vec] # First sample is the (modified) original
    user_tensor = torch.Tensor(current_user_vec).to(device)
    
    # Get predictions for all items for the base user vector
    base_user_labels = recommender_run(user_tensor, model, item_tensor_all_items, item_id=None, wanted_output='vector', **kw_dict_passed).cpu().detach().numpy()
    neighborhood_labels = [base_user_labels]
    distances = [0.0] # Distance of original to itself is 0
    
    np.random.seed(seed)

    for _ in range(num_of_perturbations):
        neighbor = current_user_vec.copy()
        dist_val = np.random.randint(min_pert, high=max_pert + 1) # +1 because high is exclusive
        
        # Ensure pos and neg don't exceed available 1s and 0s
        num_ones_in_neighbor = np.sum(neighbor)
        num_zeros_in_neighbor = len(neighbor) - num_ones_in_neighbor

        pos_flips = min(np.random.randint(0, high=dist_val + 1), num_ones_in_neighbor)
        neg_flips = min(dist_val - pos_flips, num_zeros_in_neighbor)
        
        # Actual distance might be less if dist_val was too large
        actual_dist = pos_flips + neg_flips
        if actual_dist == 0 and num_of_perturbations > 0 : # Avoid zero distance unless it's the only sample
            # if no flips happened, force at least one flip if possible
            if num_ones_in_neighbor > 0:
                pos_flips = 1; neg_flips = 0; actual_dist=1;
            elif num_zeros_in_neighbor > 0:
                neg_flips = 1; pos_flips = 0; actual_dist=1;
            # if user_vec is all 0s or all 1s (after initial item_id mask), dist could still be 0.

        if pos_flips > 0:
            pos_locations = np.random.choice(np.where(neighbor==1)[0], size=pos_flips, replace=False)
            neighbor[pos_locations] = 0
        if neg_flips > 0:
            neg_locations = np.random.choice(np.where(neighbor==0)[0], size=neg_flips, replace=False)
            neighbor[neg_locations] = 1
            
        neighborhood_data.append(neighbor)
        distances.append(float(actual_dist))
        
        neighbor_tensor = torch.Tensor(neighbor).to(device)
        neighbor_labels = recommender_run(neighbor_tensor, model, item_tensor_all_items, item_id=None, wanted_output='vector', **kw_dict_passed).cpu().detach().numpy()
        neighborhood_labels.append(neighbor_labels)
        
    neighborhood_data_np = np.array(neighborhood_data)
    neighborhood_labels_np = np.array(neighborhood_labels)
    
    return neighborhood_data_np, neighborhood_labels_np, distances, item_id
    


# In[10]:


def get_lire_args(user_vec, item_id, model, item_tensor_all_items, train_array, num_of_perturbations, proba = 0.1, seed = 0, **kw_dict_passed):
    device = kw_dict_passed['device']
    num_features = kw_dict_passed['num_items'] # Assuming num_features is num_items

    current_user_vec = user_vec.copy()
    current_user_vec[item_id] = 0
    user_tensor = torch.Tensor(current_user_vec).to(device)
    
    if torch.sum(user_tensor) == 0 and num_of_perturbations > 0:
        # If user vector is all zeros (after masking item_id), LIME might not be meaningful
        # Return structure that leads to zero importance explanations
        # Shape of neighborhood_data: (num_perturbations + 1, num_features)
        # Shape of neighborhood_labels: (num_perturbations + 1, num_features) (scores for all items)
        return (
            np.zeros((num_of_perturbations + 1, num_features)),
            np.zeros((num_of_perturbations + 1, num_features)),
            [0.0] * (num_of_perturbations + 1),
            item_id
        )
    
    np.random.seed(seed)
    # Ensure train_array is numpy for std calculation
    stds = np.std(np.asarray(train_array), axis=0)
    stds[stds == 0] = 1e-6 # Avoid division by zero if a feature has no variance
    
    # Expand original user vector for batch perturbation
    expanded_users = user_tensor.expand(num_of_perturbations, num_features).clone() # Use clone for safety
    
    # Generate noise based on std deviations
    # Original LIRE paper suggests sampling z_i from N(0,1) then scaling by sigma_i for feature i.
    # The code had: item_perturbation = nn.init.normal_(torch.zeros(num_of_perturbations, 1, device=device), 0, stds[item])
    # then hstacked. This is equivalent to sampling all features for all perturbations at once.
    noise_std_normal = torch.randn(num_of_perturbations, num_features, device=device)
    scaled_noise = noise_std_normal * torch.tensor(stds, device=device, dtype=torch.float) # Broadcast stds
    
    # Apply noise based on probability mask
    # Perturb only non-zero features of the original user vector, as per original code intent with (users != 0.)
    # However, LIRE paper also perturbs zero-valued features. Let's stick to original code's apparent intent.
    # Creating mask for features that were originally non-zero in the user_tensor
    # user_tensor is 1D. expanded_users is (num_perturbations, num_features)
    # We need mask for elements of expanded_users that were non-zero in user_tensor
    original_non_zero_mask = (user_tensor != 0.0).expand_as(expanded_users)
    
    random_proba_mask = torch.rand(num_of_perturbations, num_features, device=device) < proba
    final_perturb_mask = random_proba_mask & original_non_zero_mask
    
    perturbed_features = scaled_noise * final_perturb_mask.float()
    
    # Add perturbations to expanded users
    perturbed_user_vectors = expanded_users + perturbed_features
    perturbed_user_vectors = torch.clamp(perturbed_user_vectors, 0, 1) # Ensure values are [0,1]
    
    # Prepend original user_tensor (after masking item_id)
    neighborhood_data_torch = torch.vstack((user_tensor.unsqueeze(0), perturbed_user_vectors))
    
    distances = []
    neighborhood_labels_list = []
    
    # Get model predictions for all perturbed samples + original
    for i in range(num_of_perturbations + 1):
        p_user_tensor = neighborhood_data_torch[i, :]
        # Calculate L1 distance from the original (masked) user_tensor
        dist_val = torch.sum(torch.abs(user_tensor - p_user_tensor)).item()
        distances.append(dist_val)
        
        p_labels = recommender_run(p_user_tensor, model, item_tensor_all_items, item_id=None, wanted_output='vector', **kw_dict_passed).cpu().detach().numpy()
        neighborhood_labels_list.append(p_labels)
    
    neighborhood_data_np = neighborhood_data_torch.cpu().numpy()
    neighborhood_labels_np = np.array(neighborhood_labels_list)
    
    return neighborhood_data_np, neighborhood_labels_np, distances, item_id

# New centralized explanation functions:
def explain_instance_lime(
    user_vector_np, item_id_to_explain, recommender_model, 
    all_items_tensor_global, kw_dict_global,
    min_pert=10, max_pert=20, num_perturbations_lime=5,
    kernel_fn=distance_to_proximity, num_features_to_select=10, 
    feature_selection_method='auto', explanation_method='POS', 
    random_state_lime=None
):
    user_hist_for_lime = user_vector_np.copy()
    lime_explainer = LimeBase(kernel_fn, random_state=random_state_lime)
    neighborhood_data, neighborhood_labels, distances, item_id_label = get_lime_args(
        user_hist_for_lime, item_id_to_explain, recommender_model, all_items_tensor_global,
        min_pert=min_pert, max_pert=max_pert, 
        num_of_perturbations=num_perturbations_lime, 
        seed=item_id_to_explain, **kw_dict_global
    )
    explanation = lime_explainer.explain_instance_with_data(
        neighborhood_data, neighborhood_labels, distances, 
        label=item_id_label, 
        num_features=num_features_to_select, 
        feature_selection=feature_selection_method,
        pos_neg=explanation_method
    )
    return explanation

def explain_instance_lire(
    user_vector_np, item_id_to_explain, recommender_model,
    all_items_tensor_global, train_array_global, kw_dict_global,
    num_perturbations_lire=200, proba_lire=0.1,
    kernel_fn=distance_to_proximity, num_features_to_select=10, 
    feature_selection_method='auto', explanation_method='POS',
    random_state_lime=None
):
    user_hist_for_lire = user_vector_np.copy()
    lime_explainer = LimeBase(kernel_fn, random_state=random_state_lime)
    neighborhood_data, neighborhood_labels, distances, item_id_label = get_lire_args(
        user_hist_for_lire, item_id_to_explain, recommender_model, all_items_tensor_global,
        train_array_global, 
        num_of_perturbations=num_perturbations_lire, 
        seed=item_id_to_explain, proba=proba_lire, **kw_dict_global
    )
    explanation = lime_explainer.explain_instance_with_data(
        neighborhood_data, neighborhood_labels, distances, 
        label=item_id_label,
        num_features=num_features_to_select,
        feature_selection=feature_selection_method,
        pos_neg=explanation_method
    )
    return explanation

