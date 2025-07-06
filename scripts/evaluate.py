import argparse
import json
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import load_and_preprocess_data
from src.metrics import eval_one_expl_type, create_results_table
from src.utils import load_recommender
from src.visualization import plot_all_metrics, plot_continuous_metric_distributions
from src.config import recommender_path_dict, hidden_dim_dict

def run_all_baselines(data_name, recommender_name, test_array, test_data, items_array, recommender, kw_dict, files_path, metric_type, steps):
    baselines = ['jaccard', 'cosine', 'lime', 'lxr', 'accent', 'shap']
    results = {}

    for baseline in baselines:
        if baseline == 'shap' and recommender_name != 'MLP':
            continue
        print(f"Running {baseline} baseline for {data_name} {recommender_name} ({metric_type})")
        results[baseline] = eval_one_expl_type(
            baseline, data_name, recommender_name, test_array, test_data,
            items_array, recommender, kw_dict, files_path,
            metric_type=metric_type, steps=steps, mask_by='history'
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ML1M', help='Dataset to use (ML1M, Yahoo, Pinterest)')
    parser.add_argument('--recommender', type=str, default='MLP', help='Recommender to evaluate (MLP, VAE, NCF)')
    args = parser.parse_args()

    data_name = args.dataset
    recommender_name = args.recommender

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files_path = Path("data/processed", data_name)
    checkpoints_path = Path("checkpoints")

    # Load data
    (train_data, test_data, static_test_data, pop_dict,
     train_array, test_array, items_array, all_items_tensor,
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
               'recommender_name': recommender_name,
               'min_pert': 50,
               'max_pert': 100,
               'num_of_perturbations': 150}

    recommender_path = recommender_path_dict[(data_name, recommender_name)]
    hidden_dim = hidden_dim_dict[(data_name, recommender_name)]
    recommender = load_recommender(data_name, hidden_dim, recommender_path, **kw_dict)

    # Discrete evaluation
    discrete_results = run_all_baselines(data_name, recommender_name, test_array, test_data, items_array, recommender, kw_dict, files_path, 'discrete', 5)
    df_discrete = create_results_table(discrete_results, data_name, recommender_name)
    print("Discrete Metrics Results:")
    print(df_discrete)
    plot_all_metrics(discrete_results, data_name, recommender_name, metric_type='discrete', steps=5)

    # Continuous evaluation
    continuous_results = run_all_baselines(data_name, recommender_name, test_array, test_data, items_array, recommender, kw_dict, files_path, 'continuous', 11)
    df_continuous = create_results_table(continuous_results, data_name, recommender_name)
    print("\nContinuous Metrics Results:")
    print(df_continuous)
    plot_all_metrics(continuous_results, data_name, recommender_name, metric_type='continuous', steps=11)
    
    plot_continuous_metric_distributions(continuous_results, data_name, recommender_name)


if __name__ == '__main__':
    main()