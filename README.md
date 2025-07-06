# Refining Fidelity Metrics for Explainable Recommendations

This repository contains the implementation and extended experimental results for our SIGIR 2025 paper on refined counterfactual metrics for assessing explanation fidelity in recommender systems.

## Repository Structure

```
.
├── src/
│   ├── config.py                 # Configuration files
│   ├── data_processing.py        # Scripts for data processing
│   ├── explainers.py             # Implementation of explainer models
│   ├── lime.py                   # LIME implementation
│   ├── metrics.py                # Implementation of our refined metrics
│   ├── models.py                 # Architectures of recommender models
│   ├── utils.py                  # Utility functions
│   └── visualization.py          # Code for generating visualizations
├── scripts/
│   ├── create_dictionaries.py    # Dictionary creation for baselines
│   ├── download_data.py          # Script to download datasets
│   ├── evaluate.py               # Script to evaluate models
│   ├── run_pipeline.py           # Main pipeline to run experiments
│   └── train.py                  # Script to train models
├── notebooks/
│   ├── LXR_training.ipynb        # Training code for LXR explainer
│   ├── SHAP_MLP_clusters.ipynb   # SHAP analysis for MLP recommender
│   ├── SHAP_NCF_clusters.ipynb   # SHAP analysis for NCF recommender
│   ├── SHAP_VAE_clusters.ipynb   # SHAP analysis for VAE recommender
│   ├── lime.ipynb                # LIME implementation examples
│   ├── metrics_discrete.ipynb    # Our refined metrics implementation examples
│   ├── metrics_continuous.ipynb  # Baseline metrics implementation examples
│   └── visualization.ipynb       # Notebook for visualizations
├── data/                         # Folder for datasets
├── checkpoints/                  # Folder for model checkpoints
└── results/                      # Folder for experiment results
```

## Overview

Our refined counterfactual metrics address three key limitations of existing AUC-based metrics by:
- Evaluating only the most relevant Ke explaining features
- Excluding contradictory elements that suppress recommendations
- Using fixed-length perturbations for consistent evaluation

## Installation

This project uses Python 3.8+ and PyTorch. To set up the environment, run:

```bash
pip install --upgrade pip
pip install pandas
pip install torch
pip install optuna
pip install matplotlib
pip install scipy
pip install scikit-learn
pip install wandb
pip install shap
pip install seaborn
pip install openpyxl
pip install tqdm
```

Key dependencies include:
- PyTorch - Deep learning framework
- Pandas & NumPy - Data manipulation
- scikit-learn - Machine learning utilities
- SHAP - Model interpretability
- Optuna - Hyperparameter optimization
- WandB - Experiment tracking

## How to Run

1.  **Download Data**: Run `python scripts/download_data.py` to download the required datasets.
2.  **Train Models**: Run `python scripts/train.py --model_name VAE --dataset_name ML1M` to train a recommender model.
3.  **Run Full Pipeline**: Use `python scripts/run_pipeline.py` to run the full pipeline of training, evaluation, and explanation.

## Experiments and Results

### Datasets
We evaluate our metrics on three datasets:
- MovieLens 1M
- Yahoo! Music
- Pinterest

### Metrics Implementation
Our refined metrics are implemented in `src/metrics.py` and include:
- POS@Kr,Ke: Positive Perturbations metric
- CNDCG@Ke: Counterfactual NDCG
- INS@Ke: Insertion metric
- DEL@Ke: Deletion metric

### Baseline Methods
We compare against several baseline explanation methods:
- LIME (`notebooks/lime.ipynb`)
- SHAP (`notebooks/SHAP_*_clusters.ipynb`)
- ACCENT
- LXR (our method, `notebooks/LXR_training.ipynb`)

### Additional Results
Beyond the results presented in the paper, this repository includes:
- Extended ablation studies
- Per-dataset analysis
- Additional metrics not included in the paper
- Detailed explanation examples

You can find these results in the `notebooks` directory and their output cells.

### Visualizations and Extended Analysis
We've compiled additional visualizations, graphs, and analysis in this [Google Slides document](https://docs.google.com/presentation/d/1gz8pIA8P-lRpmXvMsfHNwkxcZ7aNNFZsAiIKgqf7960/edit#slide=id.p1), which provides a more comprehensive view of our findings across all datasets and recommendation models.

### Model Checkpoints
To facilitate reproduction of our results, we provide pre-trained model checkpoints and explainer models in this [Google Drive folder](https://drive.google.com/drive/folders/15YlS9QbVXvXrnFUe1OWGCpBwHZhG9fTz?usp=drive_link). These include:
- Recommender models (MF, VAE, NCF) for all datasets
- Trained explainer models
- Serialized data structures for efficient evaluation
