import argparse
import logging
import json
from pathlib import Path
import sys

import numpy as np
import optuna
import torch
import torch.optim as optim
from sklearn.model_selection import KFold

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import load_and_preprocess_data
from src.models import MLP, VAE, NCF, GMF_model, MLP_model
from src.utils import (recommender_evaluations, sample_indices,
                         get_user_recommended_item, load_recommender)


def MLP_objective(trial, kw_dict, train_data, static_test_data, items_array, checkpoints_path, data_name):
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    beta = trial.suggest_float('beta', 0, 4)
    epochs = 10

    model = MLP(hidden_dim, **kw_dict)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    hr10 = []

    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))

    for epoch in range(epochs):
        train_matrix = sample_indices(train_data.copy(), **kw_dict)
        perm = np.random.permutation(num_training)
        loss = []
        train_pos_loss = []
        train_neg_loss = []

        if epoch != 0 and epoch % 10 == 0:
            lr = 0.1 * lr
            optimizer.lr = lr

        for b in range(num_batches):
            optimizer.zero_grad()
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx, :-2]).to(kw_dict['device'])

            batch_pos_idx = train_matrix[batch_idx, -2]
            batch_neg_idx = train_matrix[batch_idx, -1]

            batch_pos_items = torch.Tensor(items_array[batch_pos_idx]).to(kw_dict['device'])
            batch_neg_items = torch.Tensor(items_array[batch_neg_idx]).to(kw_dict['device'])

            pos_output = torch.diagonal(model(batch_matrix, batch_pos_items))
            neg_output = torch.diagonal(model(batch_matrix, batch_neg_items))

            pos_loss = torch.mean((torch.ones_like(pos_output) - pos_output) ** 2)
            neg_loss = torch.mean((neg_output) ** 2)

            batch_loss = pos_loss + beta * neg_loss
            batch_loss.backward()
            optimizer.step()

            loss.append(batch_loss.item())
            train_pos_loss.append(pos_loss.item())
            train_neg_loss.append(neg_loss.item())

        train_losses.append(np.mean(loss))
        torch.save(model.state_dict(),
                   Path(checkpoints_path, f'MLP_{data_name}_{round(lr, 4)}_{batch_size}_{trial.number}_{epoch}.pt'))

        model.eval()
        test_matrix = np.array(static_test_data)
        test_tensor = torch.Tensor(test_matrix[:, :-2]).to(kw_dict['device'])

        test_pos = test_matrix[:, -2]
        row_indices = np.arange(test_matrix.shape[0])
        test_tensor[row_indices, test_pos] = 0

        hit_rate_at_10, _, _, _, _ = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10)
        test_losses.append(-hit_rate_at_10)

        if epoch > 5:
            if test_losses[-2] <= test_losses[-1] and test_losses[-3] <= test_losses[-2] and test_losses[-4] <= \
                    test_losses[-3]:
                return max(hr10)

    return max(hr10)


def VAE_objective(trial, kw_dict, train_array, static_test_data, checkpoints_path, data_name):
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    epochs = 50

    if data_name == "Pinterest":
        VAE_config = {
            "enc_dims": [256, 64],
            "dropout": 0.5,
            "anneal_cap": 0.2,
            "total_anneal_steps": 200000
        }
    else:
        VAE_config = {
            "enc_dims": [512, 128],
            "dropout": 0.5,
            "anneal_cap": 0.2,
            "total_anneal_steps": 200000
        }
    model = VAE(VAE_config, **kw_dict)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    hr10 = []
    best_hr10 = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        if epoch != 0 and epoch % 10 == 0:
            lr = 0.1 * lr
            optimizer.lr = lr

        model.train_one_epoch(train_array, optimizer, batch_size)
        torch.save(model.state_dict(),
                   Path(checkpoints_path, f'VAE_{data_name}_{trial.number}_{epoch}_{round(lr, 4)}_{batch_size}.pt'))

        model.eval()
        hit_rate_at_10, _, _, _, _ = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10)

        if hit_rate_at_10 > best_hr10:
            best_hr10 = hit_rate_at_10
            epochs_without_improvement = 0
            torch.save(model.state_dict(),
                       Path(checkpoints_path, f'VAE_{data_name}_{trial.number}_best_{round(lr, 4)}_{batch_size}.pt'))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 10:
            break

    return best_hr10


def NCF_objective(trial, kw_dict, train_data, items_array, checkpoints_path, data_name):
    lr = trial.suggest_float('learning_rate', 0.0005, 0.005)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    beta = trial.suggest_float('beta', 0, 4)
    epochs = 20

    # These paths need to be updated or passed as arguments
    mlp_checkpoint_path = Path(checkpoints_path, 'MLP_model_ML1M_0.0001_64_27.pt')
    gmf_checkpoint_path = Path(checkpoints_path, 'GMF_best_ML1M_0.0001_32_17.pt')

    MLP = MLP_model(hidden_size=8, num_layers=3, **kw_dict)
    MLP.load_state_dict(torch.load(mlp_checkpoint_path))
    MLP.train()

    GMF = GMF_model(hidden_size=8, **kw_dict)
    GMF.load_state_dict(torch.load(gmf_checkpoint_path))
    GMF.train()

    model = NCF(factor_num=8, num_layers=3, dropout=0.5, model='NeuMF-pre', GMF_model=GMF, MLP_model=MLP, **kw_dict)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    hr10 = []
    test_losses = []

    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))

    for epoch in range(epochs):
        train_matrix = sample_indices(train_data.copy(), **kw_dict)
        perm = np.random.permutation(num_training)

        if epoch != 0 and epoch % 10 == 0:
            lr = 0.1 * lr
            optimizer.lr = lr

        for b in range(num_batches):
            optimizer.zero_grad()
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx, :-2]).to(kw_dict['device'])

            batch_pos_idx = train_matrix[batch_idx, -2]
            batch_neg_idx = train_matrix[batch_idx, -1]

            batch_pos_items = torch.Tensor(items_array[batch_pos_idx]).to(kw_dict['device'])
            batch_neg_items = torch.Tensor(items_array[batch_neg_idx]).to(kw_dict['device'])

            pos_output = model(batch_matrix, batch_pos_items)
            neg_output = model(batch_matrix, batch_neg_items)

            pos_loss = -torch.log(pos_output).mean()
            neg_loss = -torch.log(torch.ones_like(neg_output) - neg_output).mean()

            batch_loss = pos_loss + beta * neg_loss
            if batch_loss < torch.inf:
                batch_loss.backward()
                optimizer.step()

        torch.save(model.state_dict(),
                   Path(checkpoints_path, f'NCF_{data_name}_{round(lr, 5)}_{batch_size}_{trial.number}_{epoch}.pt'))

        model.eval()
        hit_rate_at_10, _, _, _, _ = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10)
        test_losses.append(-hit_rate_at_10)

        if epoch > 5:
            if test_losses[-2] <= test_losses[-1] and test_losses[-3] <= test_losses[-2] and test_losses[-4] <= \
                    test_losses[-3]:
                return max(hr10)

    return max(hr10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ML1M', help='Dataset to use (ML1M, Yahoo, Pinterest)')
    parser.add_argument('--recommender', type=str, default='MLP', help='Recommender to train (MLP, VAE, NCF)')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    args = parser.parse_args()

    data_name = args.dataset
    recommender_name = args.recommender

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f"{recommender_name}_{data_name}_Optuna.log", mode="w"),
                            logging.StreamHandler()
                        ])

    # Load data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files_path = Path("data/processed", data_name)
    (train_data, test_data, static_test_data, pop_dict,
     train_array, test_array_static, items_array, all_items_tensor,
     pop_array) = load_and_preprocess_data(data_name, files_path, device)

    num_items = test_data.shape[1]
    output_type_dict = {"VAE": "multiple", "MLP": "single", "NCF": "single"}

    kw_dict = {'device': device,
               'num_items': num_items,
               'num_features': num_items,
               'demographic': False,
               'pop_array': pop_array,
               'all_items_tensor': all_items_tensor,
               'static_test_data': static_test_data,
               'items_array': items_array,
               'output_type': output_type_dict[recommender_name],
               'recommender_name': recommender_name}

    checkpoints_path = Path("checkpoints")

    # Optuna study
    study = optuna.create_study(direction='maximize')
    logging.info("Start optimization.")

    if recommender_name == 'MLP':
        study.optimize(lambda trial: MLP_objective(trial, kw_dict, train_data, static_test_data, items_array,
                                                   checkpoints_path, data_name), n_trials=args.n_trials)
    elif recommender_name == 'VAE':
        study.optimize(
            lambda trial: VAE_objective(trial, kw_dict, train_array, static_test_data, checkpoints_path, data_name),
            n_trials=args.n_trials)
    elif recommender_name == 'NCF':
        study.optimize(
            lambda trial: NCF_objective(trial, kw_dict, train_data, items_array, checkpoints_path, data_name),
            n_trials=args.n_trials)

    print("Best hyperparameters: {}".format(study.best_params))
    print("Best metric value: {}".format(study.best_value))


if __name__ == '__main__':
    main()