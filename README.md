# Refining Fidelity Metrics for Explainable Recommendations

This repository contains the implementation and extended experimental results for our SIGIR 2025 paper on refined counterfactual metrics for assessing explanation fidelity in recommender systems.

## Repository Structure

```
.
├── code/
│   ├── LXR_training.ipynb            # Training code for LXR explainer
│   ├── SHAP_MLP_clusters.ipynb       # SHAP analysis for MLP recommender
│   ├── SHAP_NCF_clusters.ipynb       # SHAP analysis for NCF recommender
│   ├── SHAP_VAE_clusters.ipynb       # SHAP analysis for VAE recommender
│   ├── create_dictionaries.ipynb     # Dictionary creation for baselines
│   ├── help_functions.ipynb          # Utility functions
│   ├── lime.ipynb                    # LIME implementation
│   ├── metrics_discrete.ipynb        # Our refined metrics implementation
│   ├── metrics_continuous.ipynb      # Baseline metrics implementation
│   ├── recommenders_architecture.ipynb # Model architectures
│   └── recommenders_training.ipynb    # Model training procedures
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
pip install ipynb
pip install scipy
pip install scikit-learn
pip install wandb
pip install shap
pip install seaborn
pip install openpyxl
pip install tqdm
pip install import-ipynb
```

Key dependencies include:
- PyTorch - Deep learning framework
- Pandas & NumPy - Data manipulation
- scikit-learn - Machine learning utilities
- SHAP - Model interpretability
- Optuna - Hyperparameter optimization
- WandB - Experiment tracking
- import-ipynb - For importing functions between notebooks

## Experiments and Results

### Datasets
We evaluate our metrics on three datasets:
- MovieLens 1M
- Yahoo! Music 
- Pinterest

### Metrics Implementation
Our refined metrics include:
- POS@Kr,Ke: Positive Perturbations metric
- CNDCG@Ke: Counterfactual NDCG
- INS@Ke: Insertion metric
- DEL@Ke: Deletion metric

### Baseline Methods
We compare against several baseline explanation methods:
- LIME (lime.ipynb)
- SHAP (SHAP_*_clusters.ipynb)
- ACCENT 
- LXR (our method, LXR_training.ipynb)

### Additional Results
Beyond the results presented in the paper, this repository includes:
- Extended ablation studies
- Per-dataset analysis
- Additional metrics not included in the paper
- Detailed explanation examples

You can find these results in the metrics notebooks and their output cells.

### Visualizations and Extended Analysis
We've compiled additional visualizations, graphs, and analysis in this [Google Slides document](https://docs.google.com/presentation/d/1gz8pIA8P-lRpmXvMsfHNwkxcZ7aNNFZsAiIKgqf7960/edit#slide=id.p1), which provides a more comprehensive view of our findings across all datasets and recommendation models.

