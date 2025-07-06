import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path

def load_and_preprocess_data(data_name, files_path, device):
    """
    Loads and preprocesses the data for a given dataset.

    Args:
        data_name (str): The name of the dataset (e.g., 'ML1M').
        files_path (Path): The path to the directory containing the data files.
        device (torch.device): The device to move tensors to.

    Returns:
        tuple: A tuple containing all the necessary data structures:
               (train_data, test_data, static_test_data, pop_dict,
                train_array, test_array, items_array, all_items_tensor,
                pop_array)
    """
    # Load data from CSV files
    train_data = pd.read_csv(Path(files_path, f'train_data_{data_name}.csv'), index_col=0)
    test_data = pd.read_csv(Path(files_path, f'test_data_{data_name}.csv'), index_col=0)
    static_test_data = pd.read_csv(Path(files_path, f'static_test_data_{data_name}.csv'), index_col=0)

    # Load popularity dictionary
    with open(Path(files_path, f'pop_dict_{data_name}.pkl'), 'rb') as f:
        pop_dict = pickle.load(f)

    # Convert to numpy arrays
    train_array = train_data.to_numpy()
    test_array = test_data.to_numpy()

    # Create items array and tensor
    num_items = test_data.shape[1]
    items_array = np.eye(num_items)
    all_items_tensor = torch.Tensor(items_array).to(device)

    # Preprocess static test data
    for row in range(static_test_data.shape[0]):
        # Ensure the index is within bounds
        if static_test_data.iloc[row, -2] < static_test_data.shape[1]:
            static_test_data.iloc[row, static_test_data.iloc[row, -2]] = 0
    
    test_array_static = static_test_data.iloc[:, :-2].to_numpy()


    # Create popularity array
    pop_array = np.zeros(len(pop_dict))
    for key, value in pop_dict.items():
        if key < len(pop_array):
            pop_array[key] = value

    return (train_data, test_data, static_test_data, pop_dict,
            train_array, test_array_static, items_array, all_items_tensor,
            pop_array)