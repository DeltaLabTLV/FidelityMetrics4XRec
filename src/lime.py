import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from src.utils import recommender_run

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""

    def __init__(self, kernel_fn, verbose=False, random_state=None):
        """
        Args:
            kernel_fn: function that transforms an array of distances into an array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to generate random numbers.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data."""
        x_vector = weighted_data
        alphas, _, coefs = lars_path(
            x_vector,
            weighted_labels,
            max_iter=15,
            eps=2.220446049250313e-7,
            method='lasso',
            verbose=False
        )
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
                clf.fit(data[:, used_features + [feature]], labels, sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]], labels, sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True, random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)
            coef = clf.coef_
            if hasattr(data, "tocsc") and hasattr(data, "shape") and len(data.shape) == 2 and hasattr(coef, "shape"):
                # Handle sparse data
                import scipy as sp
                if sp.sparse.issparse(data):
                    coef = sp.sparse.csr_matrix(clf.coef_)
                    weighted_data = coef.multiply(data[0])
                    sdata = len(weighted_data.data)
                    argsort_data = np.abs(weighted_data.data).argsort()
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
            # Dense data
            weighted_data = coef * data[0]
            feature_weights = sorted(
                zip(range(data.shape[1]), weighted_data),
                key=lambda x: np.abs(x[1]),
                reverse=True
            )
            return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weights = np.asarray(weights)
            weighted_data = ((data - np.average(data, axis=0, weights=weights)) * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights)) * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data, weighted_labels)
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
            return self.feature_selection(data, labels, weights, num_features, n_method)

    def explain_instance_with_data(
        self,
        neighborhood_data,
        neighborhood_labels,
        distances_list,
        label,
        num_features,
        feature_selection='auto',
        model_regressor=None,
        pos_neg='POS'
    ):
        """Takes perturbed data, labels and distances, returns explanation."""
        weights = self.kernel_fn(distances_list)
        # Handle case where neighborhood_labels has only one column
        if neighborhood_labels.shape[1] == 1:
            labels_column = neighborhood_labels[:, 0]
        else:
            labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(
            neighborhood_data, labels_column, weights, num_features, feature_selection
        )
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features], labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features], labels_column, sample_weight=weights
        )
        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])
        if pos_neg == 'POS':
            return sorted(zip(used_features, easy_model.coef_), key=lambda x: x[1], reverse=True)
        elif pos_neg == 'NEG':
            return sorted(zip(used_features, easy_model.coef_), key=lambda x: x[1], reverse=False)
        elif pos_neg == 'ABS':
            return sorted(zip(used_features, easy_model.coef_), key=lambda x: np.abs(x[1]), reverse=False)
        else:
            return 'Unfamiliar method'

def distance_to_proximity(distances_list):
    return [1 - distances_list[i] / sum(distances_list) for i in range(len(distances_list))]

def gaussian_kernel(distances, sigma=1):
    kernel = [np.exp(-distances[i] ** 2 / (2 * sigma ** 2)) for i in range(len(distances))]
    return kernel

def get_lime_args(user_vec, item_id, model, item_tensor, min_pert=10, max_pert=20, num_of_perturbations=5, seed=0, **kw):
    output_type = kw['output_type']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_vec[item_id] = 0
    neighborhood_data = [user_vec]
    user_tensor = torch.Tensor(user_vec).to(device)
    if output_type == 'single':
        result = recommender_run(user_tensor, model, item_tensor, None, 'vector', **kw).cpu().detach().numpy()
        if result.shape == ():  # 0-d array (scalar)
            user_labels = [float(result)]
        else:
            user_labels = [float(i) for i in result]
    else:
        user_labels = model(user_tensor)[0].tolist()
    neighborhood_labels = [user_labels]
    distances = [0]
    np.random.seed(seed)
    for perturbation in range(num_of_perturbations):
        neighbor = user_vec.clone()
        dist = np.random.randint(min_pert, high=max_pert)
        pos = min(np.random.randint(0, high=dist), np.sum(user_vec.cpu().numpy()))
        neg = dist - pos
        neg_locations = np.random.choice(np.where(neighbor.cpu().numpy() == 0)[0], int(neg))
        pos_locations = np.random.choice(np.where(neighbor.cpu().numpy() == 1)[0], int(pos))
        for l in neg_locations:
            neighbor[l] = 1
        for l in pos_locations:
            neighbor[l] = 0
        neighborhood_data.append(neighbor)
        distances.append(dist)
        if output_type == 'single':
            result = recommender_run(torch.Tensor(neighbor), model, item_tensor, None, 'vector', **kw).cpu().detach().numpy()
            if result.shape == ():  # 0-d array (scalar)
                labels = [float(result)]
            else:
                labels = [float(i) for i in result]
            neighborhood_labels.append(labels)
        else:
            neighborhood_labels.append(model(torch.Tensor(neighbor).to(device))[0].tolist())
    neighborhood_data = np.array([t.cpu().numpy() for t in neighborhood_data])
    neighborhood_labels = np.array(neighborhood_labels)
    return neighborhood_data, neighborhood_labels, distances, item_id

def get_lire_args(user_vec, item_id, model, item_tensor, train_array, num_of_perturbations, proba=0.1, seed=0, **kw):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_vec[item_id] = 0
    user_tensor = torch.Tensor(user_vec).to(device)
    # Check if all items are masked
    if torch.sum(user_tensor) == 0:
        # Return data that will result in zero importance for all items
        return (
            np.zeros((num_of_perturbations + 1, len(user_vec))),
            np.zeros((num_of_perturbations + 1, len(user_vec))),
            [0] * (num_of_perturbations + 1),
            item_id
        )
    np.random.seed(seed)
    stds = np.std(train_array, axis=0)
    num_features = kw['num_features']
    users = user_tensor.expand(num_of_perturbations, num_features).detach().clone()
    neighborhood_data = torch.zeros(num_of_perturbations, 1, device=device)
    for item in range(num_features):
        item_perturbation = nn.init.normal_(torch.zeros(num_of_perturbations, 1, device=device), 0, stds[item])
        neighborhood_data = torch.hstack((neighborhood_data, item_perturbation))
    neighborhood_data = neighborhood_data[:, 1:]
    rd_mask = torch.zeros(num_of_perturbations, num_features, device=device).uniform_() > (1. - proba)
    neighborhood_data = neighborhood_data * rd_mask * (users != 0.)
    neighborhood_data = users + neighborhood_data
    neighborhood_data = torch.clamp(neighborhood_data, 0, 1)
    neighborhood_data = torch.vstack((user_tensor, neighborhood_data))
    distances = []
    neighborhood_labels = []
    for perturbation in range(num_of_perturbations + 1):
        neighbor = neighborhood_data[perturbation, :]
        distances.append(torch.sum(torch.abs(torch.sub(user_tensor, neighbor))).item())
        labels = [float(i) for i in recommender_run(neighbor, model, item_tensor, None, 'vector', **kw).cpu().detach().numpy()]
        neighborhood_labels.append(labels)
    neighborhood_data = np.array(torch.abs(neighborhood_data).cpu().detach().numpy())
    neighborhood_labels = np.array(neighborhood_labels)
    return neighborhood_data, neighborhood_labels, distances, item_id