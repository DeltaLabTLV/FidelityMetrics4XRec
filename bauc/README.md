# Fidelity Metrics for Explainable Recommender Systems

This project explores fidelity metrics for various explanation methods applied to recommender systems. It is based on the work from the paper "Beyond AUC: Towards a Deeper Understanding of the Quality of Explanations for Recommender Systems".

The codebase allows for:
1.  Training different recommender system models (MLP, VAE, NCF).
2.  Generating explanations for recommendations using various techniques (LIME, SHAP, LXR, FIA, etc.).
3.  Calculating fidelity and other metrics to evaluate the quality of these explanations.

## Project Structure

-   `bauc/code/`: Contains all Python scripts for training, explanation, and evaluation.
    -   `help_functions.py`: Core utility functions, including data loading, model loading, and evaluation metrics.
    -   `recommenders_architecture.py`: Definitions of the recommender system model architectures.
    -   `recommenders_training.py`: Script for training the recommender models.
    -   `LXR_training.py`: Script for training the LXR explainer.
    -   `SHAP_*.py`: Scripts for generating SHAP explanations for different models.
    -   `lime.py`: Implementation of LIME and LIRE.
    -   `metrics_continous.py` / `metrics_discrete.py`: Scripts for calculating and evaluating various explanation metrics.
    -   `create_dictionaries.py`: Script for pre-calculating and saving various data dictionaries (e.g., similarities, popularities).
-   `processed_data/<dataset_name>/`: (Needs to be created by the user or scripts) Will contain processed data files.
-   `checkpoints/`: (Needs to be created by the user or scripts) Will contain saved model checkpoints.
-   `requirements.txt`: Python dependencies.
-   `README.md`: This file.

## Setup

### 1. Clone the Repository

If you haven't already, clone this repository:
```bash
git clone https://github.com/DeltaLabTLV/FidelityMetrics4XRec.git
cd FidelityMetrics4XRec
```

### 2. Create Python Environment and Install Dependencies

It is highly recommended to use a virtual environment (e.g., venv or conda).

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```
You might need to install `python3-venv` if it's not available: `sudo apt update && sudo apt install python3-venv`.

### 3. Data Preparation

The Python scripts in this project operate on preprocessed data files for datasets like ML1M, Yahoo, and Pinterest.

**Input Data Format:**
The scripts expect base data files such as `train_data_<dataset_name>.csv` and `test_data_<dataset_name>.csv`.
-   **Obtaining Base Processed Files:** You will need to first download the raw datasets (e.g., MovieLens 1M, Yahoo Music, Pinterest). The conversion of these raw datasets into the required base CSV format (e.g., user-item interaction tables) is a preliminary step. For guidance on this initial preprocessing, you may need to refer to the original paper or the `Anonymous-beyond-auc/bauc` repository, which might provide scripts or instructions for this specific transformation.
-   **Directory Structure:** Place these base processed CSV files (once obtained) into a directory structure like:
    `processed_data/<dataset_name>/` (e.g., `FidelityMetrics4XRec/processed_data/ML1M/train_data_ML1M.csv`)

**Generating Derived Data Dictionaries (Using this project's scripts):**
Several Python scripts in this project rely on derived data dictionaries (e.g., item popularities, similarities). The `bauc/code/create_dictionaries.py` script is provided to generate these from your base processed CSV files.
-   Before running analyses that depend on these dictionaries, execute:
    ```bash
    cd bauc/code
    python create_dictionaries.py
    ```
    (Ensure you set `data_name` within `create_dictionaries.py` and confirm it can correctly read your base processed files from the `../../processed_data/<dataset_name>/` relative path, or adjust paths accordingly). These derived files (e.g., `pop_dict_ML1M.pkl`) will typically be saved by the script into the same `processed_data/<dataset_name>/` directory.

**Path Adjustments in Scripts:**
The Python scripts (`.py` files in `bauc/code/`) contain path variables (e.g., `files_path`, `DP_DIR`, `export_dir`, `checkpoints_path`) that point to data and checkpoint locations. You **must** review and adjust these paths within the scripts to match your local setup.
-   `files_path` should point to your data directory (e.g., `FidelityMetrics4XRec/processed_data/<dataset_name>/`).
-   `checkpoints_path` should point to your checkpoints directory (e.g., `FidelityMetrics4XRec/checkpoints/`).
    Scripts often define these relative to their `export_dir` (which is usually `Path(os.getcwd()).parent` when running from `bauc/code/`, thus pointing to `FidelityMetrics4XRec/`).

### 4. Create Directories for Checkpoints

Create a directory for saving model checkpoints:
```bash
mkdir checkpoints
```
The scripts in `bauc/code/` will expect this directory to exist at the root of the `FidelityMetrics4XRec` project (i.e., sibling to the `bauc` directory). Some scripts might define `checkpoints_path` relative to their own location or a parent directory. You might need to adjust path definitions like `checkpoints_path = Path(export_dir.parent, "checkpoints")` or `checkpoints_path = Path(export_dir, "checkpoints")` in the scripts to ensure they point to `FidelityMetrics4XRec/checkpoints/`.

## How to Run

The Python scripts for this project are located in the `bauc/code/` directory. **To run them, first navigate to this directory:**
```bash
cd bauc/code
```
Then, you can execute the scripts using `python <script_name>.py`. This ensures that relative imports within the scripts (e.g., `from .help_functions import ...`) work correctly.

Many scripts have global variables near the top (e.g., `data_name`, `recommender_name`). You will need to modify these to select the dataset and model for your experiments.

### 1. Precompute Dictionaries (Using this project's scripts)

The `bauc/code/create_dictionaries.py` script precomputes and saves various data dictionaries (like item similarities and popularities) from your base CSV data files. These dictionaries are then used by other analysis scripts.
```bash
# Ensure you are in the FidelityMetrics4XRec/bauc/code/ directory
python create_dictionaries.py
```
(Remember to set `data_name` inside the script and verify that file paths for reading base data and writing dictionaries are correctly configured, typically to use `../../processed_data/<dataset_name>/`).

### 2. Train Recommender Models

Use `recommenders_training.py` to train the recommender models.
```bash
python recommenders_training.py
```
(Set `data_name` and `recommender_name` in the script. Trained models will be saved in the `checkpoints/` directory).

### 3. Train LXR Explainer (if using LXR)

Use `LXR_training.py` to train the LXR explainer.
```bash
python LXR_training.py
```
(Set `data_name` and `recommender_name`. Ensure the corresponding recommender model is already trained).

### 4. Generate SHAP Explanations (if using SHAP)

Use the `SHAP_*.py` scripts (e.g., `SHAP_MLP_clusters.py`) to generate and save SHAP values.
```bash
python SHAP_MLP_clusters.py
```
(Set `data_name`. Ensure the corresponding recommender model is trained and data/paths are correctly set).

### 5. Calculate Explanation Metrics

Use `metrics_continous.py` or `metrics_discrete.py` to calculate fidelity and other metrics for the generated explanations.
```bash
python metrics_continous.py
```
(Set `data_name` and `recommender_name`. This script loads trained models, explanation data (like SHAP values or LXR models), and computes metrics).

## Key Scripts and Their Roles:

-   **`recommenders_training.py`**:
    -   Trains MLP, VAE, or NCF recommenders.
    -   Uses Optuna for hyperparameter tuning.
    -   Saves trained model checkpoints.
-   **`help_functions.py`**:
    -   `load_recommender()`: Loads trained recommender models.
    -   `recommender_evaluations()`: Evaluates recommender performance (HR@k, MRR, etc.).
    -   `single_user_metrics()`: Core function (likely within metrics scripts) for computing fidelity for a single user.
    -   `setup_shap_experiment_data()`: Helper to load data for SHAP experiments.
    -   `load_lxr_explainer()`: Loads trained LXR models.
-   **`lime.py`**:
    -   Contains `LimeBase` class and functions like `explain_instance_lime` and `explain_instance_lire` for generating LIME/LIRE explanations.
-   **`SHAP_MLP_clusters.py` (and similar for NCF, VAE)**:
    -   Loads a trained recommender.
    -   Prepares data by clustering items.
    -   Uses the `shap` library with a model wrapper to compute SHAP values for item cluster importance.
    -   Saves SHAP values.
-   **`metrics_continous.py` / `metrics_discrete.py`**:
    -   Main scripts for evaluating explanation quality.
    -   Load recommender models and pre-computed explanations (SHAP, LXR) or generate them on-the-fly (LIME, FIA).
    -   Implement various fidelity metrics (e.g., fidelity+, fidelity-, PGI, PGU) and other evaluation measures.
    -   Iterate over users in the test set to compute these metrics.
    -   Aggregate and save results.

## Customization

-   **Datasets**: To use a new dataset, you'll need to preprocess it into the expected format (user-item interaction matrices/lists) and update data loading paths and parameters (number of users/items) in the scripts.
-   **Recommenders**: New recommender architectures can be added to `recommenders_architecture.py` and integrated into the training and evaluation pipelines.
-   **Explainers**: New explanation methods can be implemented and evaluated within the metrics scripts.
-   **Paths**: **Crucially, review and update file paths** in all relevant scripts to match your local environment and data locations. Variables to check include `files_path`, `checkpoints_path`, `export_dir`, and `DP_DIR`.

## Original Repository and Paper

This project builds upon the concepts and codebase from:
-   Paper: "Beyond AUC: Towards a Deeper Understanding of the Quality of Explanations for Recommender Systems"
-   Original Repository: `https://github.com/Anonymous-beyond-auc/bauc` (Note: This link might be for an anonymized version. The actual public repository may differ.)

Please refer to the original paper for theoretical background and detailed experimental setup.
