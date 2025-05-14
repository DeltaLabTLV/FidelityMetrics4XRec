#!/usr/bin/env python
# coding: utf-8

# # Imports

import pandas as pd
import numpy as np
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Keep commented
# export_dir = os.getcwd() # Keep commented/removed top-level one if it was a mistake
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import copy
import optuna
import logging
# import ipynb # Remove
# import matplotlib.pyplot as plt # Keep commented
from sklearn.model_selection import KFold

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

data_name = "ML1M" ### Can be ML1M, Yahoo, Pinterest
recommender_name = "MLP" ## Can be MLP, VAE, NCF

DP_DIR = Path("processed_data", data_name) 
export_dir = Path(os.getcwd()).parent # Reinstated: This is the local definition for paths below
files_path = Path(export_dir, DP_DIR)
checkpoints_path = Path(export_dir, "checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_type = output_type_dict[recommender_name] ### Can be single, multiple
num_users = num_users_dict[data_name] 
num_items = num_items_dict[data_name] 

# Standard import changed to specific
from .help_functions import sample_indices, recommender_evaluations, VAE_BASE_CONFIG, PINTEREST_VAE_CONFIG
# importlib.reload(ipynb.fs.defs.help_functions) # Remove
# from ipynb.fs.defs.help_functions import * # Remove


# ## Data imports and preprocessing

train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)
with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
    pop_dict = pickle.load(f)
    
train_array = train_data.to_numpy()
test_array = test_data.to_numpy()
items_array = np.eye(num_items)
all_items_tensor = torch.Tensor(items_array).to(device)

for row in range(static_test_data.shape[0]):
    static_test_data.iloc[row, static_test_data.iloc[row,-2]]=0
test_array = static_test_data.iloc[:,:-2].to_numpy()

pop_array = np.zeros(len(pop_dict))
for key, value in pop_dict.items():
    pop_array[key] = value


# # Recommenders Import

# Standard import
from .recommenders_architecture import MLP, VAE, NCF, MLP_model, GMF_model
# importlib.reload(ipynb.fs.defs.recommenders_architecture) # Remove
# from ipynb.fs.defs.recommenders_architecture import * # Remove


# # Define the dict

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


# # Training

# ## MLP Train function

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

def MLP_objective(trial):
    
    lr = trial.suggest_float('learning_rate', 0.001, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    beta = trial.suggest_float('beta', 0, 4)
    epochs = 10
    model = MLP(hidden_dim, **kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    print(f'======================== new run - {recommender_name} ========================')
    logger.info(f'======================== new run - {recommender_name} ========================')
    
    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))

    
    for epoch in range(epochs):
        train_matrix = sample_indices(train_data.copy(), **kw_dict)
        perm = np.random.permutation(num_training)
        loss = []
        train_pos_loss=[]
        train_neg_loss=[]
        if epoch!=0 and epoch%10 == 0:
            lr = 0.1*lr
            optimizer.lr = lr
        
        for b in range(num_batches):
            optimizer.zero_grad()
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]    
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx,:-2]).to(device)

            batch_pos_idx = train_matrix[batch_idx,-2]
            batch_neg_idx = train_matrix[batch_idx,-1]
            
            batch_pos_items = torch.Tensor(items_array[batch_pos_idx]).to(device)
            batch_neg_items = torch.Tensor(items_array[batch_neg_idx]).to(device)
            
            pos_output = torch.diagonal(model(batch_matrix, batch_pos_items))
            neg_output = torch.diagonal(model(batch_matrix, batch_neg_items))
            
            pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
            neg_loss = torch.mean((neg_output)**2)
            
            batch_loss = pos_loss + beta*neg_loss
            batch_loss.backward()
            optimizer.step()
            
            loss.append(batch_loss.item())
            train_pos_loss.append(pos_loss.item())
            train_neg_loss.append(neg_loss.item())
            
        print(f'train pos_loss = {np.mean(train_pos_loss)}, neg_loss = {np.mean(train_neg_loss)}')    
        train_losses.append(np.mean(loss))
        torch.save(model.state_dict(), Path(checkpoints_path, f'MLP_{data_name}_{round(lr,4)}_{batch_size}_{trial.number}_{epoch}.pt'))


        model.eval()
        test_matrix = np.array(static_test_data)
        test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
        
        test_pos = test_matrix[:,-2]
        test_neg = test_matrix[:,-1]
        
        row_indices = np.arange(test_matrix.shape[0])
        test_tensor[row_indices,test_pos] = 0
        
        pos_items = torch.Tensor(items_array[test_pos]).to(device)
        neg_items = torch.Tensor(items_array[test_neg]).to(device)
        
        pos_output = torch.diagonal(model(test_tensor, pos_items).to(device))
        neg_output = torch.diagonal(model(test_tensor, neg_items).to(device))
        
        pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
        neg_loss = torch.mean((neg_output)**2)
        print(f'test pos_loss = {pos_loss}, neg_loss = {neg_loss}')
        
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10)
        print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
        
        test_losses.append(-hit_rate_at_10)
        if epoch>5:
            if test_losses[-2]<=test_losses[-1] and test_losses[-3]<=test_losses[-2] and test_losses[-4]<=test_losses[-3]:
                logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                train_losses_dict[trial.number] = train_losses
                test_losses_dict[trial.number] = test_losses
                HR10_dict[trial.number] = hr10
                return max(hr10)
            
    logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)


# ## VAE Train function

# ### Define the configs (they are defined once again inside the load recommender function in the "help funcion" notebook

# VAE_config= { # REMOVE LOCAL DEFINITION
# "enc_dims": [512,128],
# "dropout": 0.5,
# "anneal_cap": 0.2,
# "total_anneal_steps": 200000}

# Pinterest_VAE_config= { # REMOVE LOCAL DEFINITION
# "enc_dims": [256,64],
# "dropout": 0.5,
# "anneal_cap": 0.2,
# "total_anneal_steps": 200000}


# ### Function

def cross_validate_vae(n_splits=5):
    # Combine data for splitting
    all_data = pd.concat([train_data, test_data])
    
    # Добавляем столбцы pos и neg, если их нет
    if all_data.shape[1] == num_items:
        # Для каждого пользователя находим один случайный положительный и отрицательный предмет
        pos_items = []
        neg_items = []
        for _, user_data in all_data.iterrows():
            # Находим индексы положительных и отрицательных взаимодействий
            pos_indices = np.where(user_data > 0)[0]
            neg_indices = np.where(user_data == 0)[0]
            
            # Выбираем случайный положительный и отрицательный предмет
            if len(pos_indices) > 0:
                pos_items.append(np.random.choice(pos_indices))
            else:
                pos_items.append(0)  # fallback
                
            if len(neg_indices) > 0:
                neg_items.append(np.random.choice(neg_indices))
            else:
                neg_items.append(0)  # fallback
        
        all_data['pos'] = pos_items
        all_data['neg'] = neg_items
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # Split data
        fold_train = all_data.iloc[train_idx]
        fold_test = all_data.iloc[test_idx]
        
        # Создаем копию kw_dict и обновляем static_test_data
        fold_kw_dict = kw_dict.copy()
        fold_kw_dict['static_test_data'] = fold_test
        
        # Initialize and train model
        model = VAE(VAE_BASE_CONFIG, **kw_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print(f"Fold shapes - Train: {fold_train.shape}, Test: {fold_test.shape}")
        
        # Training loop
        best_hr10 = 0
        for epoch in range(50):
            # Тренируем только на данных взаимодействий (без pos/neg столбцов)
            loss = model.train_one_epoch(fold_train.iloc[:, :num_items].to_numpy(), 
                                       optimizer, batch_size=128)
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                hit_rate_at_10, _, _, _, _ = recommender_evaluations(model, **fold_kw_dict)
                
                if hit_rate_at_10 > best_hr10:
                    best_hr10 = hit_rate_at_10
        
        fold_results.append(best_hr10)
        print(f"Fold {fold + 1} best HR@10: {best_hr10:.4f}")
    
    print("\nCross-validation results:")
    print(f"Mean HR@10: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    return fold_results

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

def VAE_objective(trial):
    lr = trial.suggest_float('learning_rate', 0.0001, 0.005, log=True)
    # enc_dims_choice = trial.suggest_categorical('enc_dims', [[512, 128], [256, 128], [1024, 256, 128]])
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.7)
    anneal_cap = trial.suggest_float('anneal_cap', 0.1, 0.5)
    # total_anneal_steps = trial.suggest_int('total_anneal_steps', 100000, 300000)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    beta = trial.suggest_float('beta', 0.5, 2.0)  # For beta-VAE, if applicable
    epochs = 10

    # Use imported VAE configs
    if data_name == "Pinterest":
        current_vae_config = PINTEREST_VAE_CONFIG.copy() # Use copy to modify safely if needed
    else:
        current_vae_config = VAE_BASE_CONFIG.copy()
    
    # Override with Optuna-suggested params
    # current_vae_config["enc_dims"] = enc_dims_choice # If you want to tune enc_dims
    current_vae_config["dropout"] = dropout_rate
    current_vae_config["anneal_cap"] = anneal_cap
    # current_vae_config["total_anneal_steps"] = total_anneal_steps


    model = VAE(current_vae_config, **kw_dict) # Pass the chosen config
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    update_count = 0
    train_losses = []
    test_losses = []
    hr10_list = [] # To store HR@10 for each epoch

    print(f'======================== new run - {recommender_name} ========================')
    logger.info(f'======================== new run - {recommender_name} ========================')

    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()
            
            if (b + 1) * batch_size >= num_training:
                batch_idx = np.random.permutation(num_training)[b * batch_size:]
            else:
                batch_idx = np.random.permutation(num_training)[b * batch_size: (b + 1) * batch_size]
            
            batch_matrix = torch.FloatTensor(train_array[batch_idx]).to(device)
            
            if model.total_anneal_steps > 0:
                anneal = min(model.anneal_cap, update_count / model.total_anneal_steps)
            else:
                anneal = model.anneal_cap
            
            recon_batch, mu, logvar = model(batch_matrix)
            loss = model.loss_function_per_user(recon_batch, batch_matrix, mu, logvar, anneal, beta) # Pass beta if using beta-VAE
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            update_count +=1
        
        train_losses.append(epoch_loss / num_batches)
        logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}')
        torch.save(model.state_dict(), Path(checkpoints_path, f'VAE_{data_name}_{lr:.4f}_{batch_size}_{trial.number}_{epoch}.pt'))

        model.eval()
        # Evaluation using recommender_evaluations
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
        hr10_list.append(hit_rate_at_10)
        
        # test_loss based on -HR@10, consistent with other objectives
        current_test_loss = -hit_rate_at_10 
        test_losses.append(current_test_loss)

        logger.info(f'Epoch {epoch+1}, HR@10: {hit_rate_at_10:.4f}, HR@50: {hit_rate_at_50:.4f}, MRR: {MRR:.4f}')
        # print(f'Epoch {epoch+1}, HR@10: {hit_rate_at_10:.4f}, HR@50: {hit_rate_at_50:.4f}, MRR: {MRR:.4f}')

        # Comment out the problematic NDCG/Recall loop
        # ndcg_list = []
        # recall_list = []
        # best_ndcg = 0
        # for K in [5, 10, 20, 50, 100]:
        #     # These functions (ndcg_score_fn, recall_score_fn) are not defined
        #     # ndcg_k = ndcg_score_fn(model, static_test_data, train_data, K, device, **kw_dict) 
        #     # recall_k = recall_score_fn(model, static_test_data, train_data, K, device, **kw_dict)
        #     # logger.info(f"NDCG@{K} : {ndcg_k:.4f}")
        #     # logger.info(f"Recall@{K} : {recall_k:.4f}")
        #     # ndcg_list.append(ndcg_k)
        #     # recall_list.append(recall_k)
        #     # if K == 20 and ndcg_k > best_ndcg: # Example: track best NDCG@20
        #     #     best_ndcg = ndcg_k
        #     pass # Placeholder for the commented block
            
        # test_losses.append(np.mean(ndcg_list)) # This was problematic

        if epoch > 5: # Adjusted early stopping to use HR@10 based test_losses
            if test_losses[-2] <= test_losses[-1] and test_losses[-3] <= test_losses[-2] and test_losses[-4] <= test_losses[-3]:
                logger.info(f'Early stop for VAE at trial {trial.number}, epoch {epoch+1}. Best HR@10: {max(hr10_list):.4f} (Test Loss: {min(test_losses):.4f})')
                # train_losses_dict[trial.number] = train_losses # Ensure these dicts are defined globally if used like this
                # test_losses_dict[trial.number] = test_losses
                # HR10_dict[trial.number] = hr10_list
                return max(hr10_list) # Return the metric Optuna tries to maximize

    logger.info(f'Finished VAE trial {trial.number}. Best HR@10: {max(hr10_list):.4f} (Test Loss: {min(test_losses):.4f})')
    # train_losses_dict[trial.number] = train_losses
    # test_losses_dict[trial.number] = test_losses
    # HR10_dict[trial.number] = hr10_list
    return max(hr10_list) # Return the metric Optuna tries to maximize


def analyze_checkpoints(best_trial_num):
    """
    Analyzes the best trial to determine gold, silver, and bronze checkpoints
    based on model performance at different epochs.
    
    Args:
        best_trial_num: The number of the best trial from optimization
        
    Returns:
        tuple: Epochs for gold, silver, and bronze checkpoints
    """
    hr10_values = HR10_dict[best_trial_num]
    max_hr10 = max(hr10_values)
    best_epoch = hr10_values.index(max_hr10)
    
    # Gold - best performing epoch
    gold_hr10 = max_hr10
    gold_epoch = best_epoch
    
    # Silver - ~60% of maximum performance
    silver_target = 0.6 * max_hr10
    silver_epoch = next(i for i, x in enumerate(hr10_values) 
                       if x >= silver_target)
    silver_hr10 = hr10_values[silver_epoch]
    
    # Bronze - early epoch with lower performance
    bronze_epoch = min(5, len(hr10_values)-1)  # take epoch 5 or earlier
    bronze_hr10 = hr10_values[bronze_epoch]
    
    print("\nCheckpoint Analysis:")
    print(f"Gold   - Epoch {gold_epoch}, HR@10: {gold_hr10:.4f}")
    print(f"Silver - Epoch {silver_epoch}, HR@10: {silver_hr10:.4f}")
    print(f"Bronze - Epoch {bronze_epoch}, HR@10: {bronze_hr10:.4f}")
    
    return gold_epoch, silver_epoch, bronze_epoch

def analyze_convergence(evaluation_metrics):
    best_epochs = []
    best_values = []
    
    for trial_values in evaluation_metrics['HR@10'].values():
        best_value = max(trial_values)
        best_epoch = trial_values.index(best_value)
        best_epochs.append(best_epoch)
        best_values.append(best_value)
    
    print(f"Statistics of best results achievement:")
    print(f"Mean epoch to best result: {np.mean(best_epochs):.1f}")
    print(f"Median epoch to best result: {np.median(best_epochs):.1f}")
    print(f"Mean best HR@10: {np.mean(best_values):.4f}")
    print(f"Best HR@10: {max(best_values):.4f}")
    print(f"Worst HR@10: {min(best_values):.4f}")


# ## NCF training functions

train_losses_dict = {}
test_losses_dict = {}
HR10_dict = {}

## PAY ATTENTION to define manualy the MLP_model and GMF_model checkpoints which will be used inside the NCF

def NCF_objective(trial):
    lr = trial.suggest_float('learning_rate', 0.0005, 0.005)
    batch_size = trial.suggest_categorical('batch_size', [32,64,128])
    beta = trial.suggest_float('beta',0, 4)
    epochs = 20
    MLP = MLP_model(hidden_size=8, num_layers=3, **kw_dict)
    # EDIT HERE
    MLP_checkpoint = torch.load(Path(checkpoints_path, 'MLP_model_ML1M_0.0001_64_27.pt'))
    MLP.load_state_dict(MLP_checkpoint)
    MLP.train()
    GMF = GMF_model(hidden_size=8, **kw_dict)
    # & EDIT HERE
    GMF_checkpoint = torch.load(Path(checkpoints_path, 'GMF_best_ML1M_0.0001_32_17.pt'))
    GMF.load_state_dict(GMF_checkpoint)
    GMF.train()
    model = NCF(factor_num=8, num_layers=3, dropout=0.5, model= 'NeuMF-pre', GMF_model= GMF, MLP_model=MLP, **kw_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    hr10 = []
    print(f'======================== new run - {recommender_name} ========================')
    logger.info(f'======================== new run - {recommender_name} ========================')
    
    num_training = train_data.shape[0]
    num_batches = int(np.ceil(num_training / batch_size))

    
    for epoch in range(epochs):
        train_matrix = sample_indices(train_data.copy(), **kw_dict)
        perm = np.random.permutation(num_training)
        loss = []
        train_pos_loss=[]
        train_neg_loss=[]
        if epoch!=0 and epoch%10 == 0:
            lr = 0.1*lr
            optimizer.lr = lr
        
        for b in range(num_batches):
            optimizer.zero_grad()
            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]    
            batch_matrix = torch.FloatTensor(train_matrix[batch_idx,:-2]).to(device)

            batch_pos_idx = train_matrix[batch_idx,-2]
            batch_neg_idx = train_matrix[batch_idx,-1]
            
            batch_pos_items = torch.Tensor(items_array[batch_pos_idx]).to(device)
            batch_neg_items = torch.Tensor(items_array[batch_neg_idx]).to(device)
            
            pos_output = model(batch_matrix, batch_pos_items)
            neg_output = model(batch_matrix, batch_neg_items)

            pos_loss = -torch.log(pos_output).mean()
            neg_loss = -torch.log(torch.ones_like(neg_output)-neg_output).mean()

            batch_loss = pos_loss + beta*neg_loss
            if batch_loss<torch.inf:
                batch_loss.backward()
                optimizer.step()
            
            loss.append(batch_loss.item())
            train_pos_loss.append(pos_loss.item())
            train_neg_loss.append(neg_loss.item())
            
        print(f'train pos_loss = {np.mean(train_pos_loss)}, neg_loss = {np.mean(train_neg_loss)}')    
        train_losses.append(np.mean(loss))
        torch.save(model.state_dict(), Path(checkpoints_path, f'{recommender_name}2_{data_name}_{round(lr,5)}_{batch_size}_{trial.number}_{epoch}.pt'))


        model.eval()
        test_matrix = np.array(static_test_data)
        test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
        
        test_pos = test_matrix[:,-2]
        test_neg = test_matrix[:,-1]
        
        row_indices = np.arange(test_matrix.shape[0])
        test_tensor[row_indices,test_pos] = 0
        
        pos_items = torch.Tensor(items_array[test_pos]).to(device)
        neg_items = torch.Tensor(items_array[test_neg]).to(device)
        
        pos_output = model(test_tensor, pos_items).to(device)
        neg_output = model(test_tensor, neg_items).to(device)
        
        pos_loss = -torch.log(pos_output).mean()
        neg_loss = -torch.log(torch.ones_like(neg_output)-neg_output).mean()
        print(f'test pos_loss = {pos_loss}, neg_loss = {neg_loss}')
        
        hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
        hr10.append(hit_rate_at_10)
        print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
                   
        
        test_losses.append(-hit_rate_at_10)
        if epoch>5:
            if test_losses[-2]<=test_losses[-1] and test_losses[-3]<=test_losses[-2] and test_losses[-4]<=test_losses[-3]:
                logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                train_losses_dict[trial.number] = train_losses
                test_losses_dict[trial.number] = test_losses
                HR10_dict[trial.number] = hr10
                return max(hr10)
            
    logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
    train_losses_dict[trial.number] = train_losses
    test_losses_dict[trial.number] = test_losses
    HR10_dict[trial.number] = hr10
    return max(hr10)

logger = logging.getLogger()

logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler(f"{recommender_name}_{data_name}_Optuna.log", mode="w"))

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

study = optuna.create_study(direction='maximize')

logger.info("Start optimization.")
study.optimize(MLP_objective, n_trials=20)

with open(f"{recommender_name}_{data_name}_Optuna.log") as f:
    assert f.readline().startswith("A new study created")
    assert f.readline() == "Start optimization.\n"
    
    
# Print best hyperparameters and corresponding metric value
print("Best hyperparameters: {}".format(study.best_params))
print("Best metric value: {}".format(study.best_value))


# # Evaluations

# ## Load the trained recommender

recommender_path_dict = {
    ("ML1M","VAE"): Path(checkpoints_path, "VAE_ML1M_0.0007_128_10.pt"),
    ("ML1M","MLP"):Path(checkpoints_path, "MLP_ML1M_0.002_1024_19_8.pt"),
    ("ML1M","NCF"):Path(checkpoints_path, "NCF_ML1M_5e-05_64_16.pt"),

    ("Yahoo","VAE"): Path(checkpoints_path, "VAE_Yahoo_0.0001_128_13.pt"),
    ("Yahoo","MLP"):Path(checkpoints_path, "MLP2_Yahoo_0.0083_128_1.pt"),
    ("Yahoo","NCF"):Path(checkpoints_path, "NCF_Yahoo_0.001_64_21_0.pt"),
    
    ("Pinterest","VAE"): Path(checkpoints_path, "VAE_Pinterest_12_18_0.0001_256.pt"),
    ("Pinterest","MLP"):Path(checkpoints_path, "MLP_Pinterest_0.0062_512_21_0.pt"),
    ("Pinterest","NCF"):Path(checkpoints_path, "NCF2_Pinterest_9e-05_32_9_10.pt")}


hidden_dim_dict = {
    ("ML1M","VAE"): None,
    ("ML1M","MLP"): 32,
    ("ML1M","NCF"): 8,
    
    ("Yahoo","VAE"): None,
    ("Yahoo","MLP"):32,
    ("Yahoo","NCF"):8,
    
    ("Pinterest","VAE"): None,
    ("Pinterest","MLP"):512,
    ("Pinterest","NCF"): 64}

hidden_dim = hidden_dim_dict[(data_name, recommender_name)]
recommender_path = recommender_path_dict[(data_name, recommender_name)]

model = load_recommender(data_name, hidden_dim, checkpoints_path, recommender_path, **kw_dict)


# ## plot the distribution of top recommended item accross all users

# plot the distribution of top recommended item accross all users
# topk_train = {}
# for i in range(len(train_array)):
#     vec = train_array[i]
#     tens = torch.Tensor(vec).to(device)
#     topk_train[i] = int(get_user_recommended_item(tens, model).cpu().detach().numpy())

# plt.hist(topk_train.values(), bins=1000)
# plt.plot(np.array(list(pop_dict.keys())), np.array(list(pop_dict.values()))*100, alpha=0.2)
# plt.show()

# topk_test = {}
# for i in range(len(test_array)):
#     vec = test_array[i]
#     tens = torch.Tensor(vec).to(device)
#     topk_test[i] = int(get_user_recommended_item(tens, model).cpu().detach().numpy())

# plt.hist(topk_test.values(), bins=400)
# plt.plot(np.array(list(pop_dict.keys())), np.array(list(pop_dict.values()))*200, alpha=0.2)
# plt.show() 

# hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model)

# print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)

