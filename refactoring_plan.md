# Refactoring Plan: Unifying Discrete and Continuous Metrics

### Understanding the Goal

The main objective is to integrate the logic from `notebooks/metrics_continous.ipynb` and `notebooks/metrics_discrete.ipynb` into `src/metrics.py` and `src/visualization.py`. This will allow us to calculate and plot both types of metrics (continuous and discrete) from a single, unified codebase.

### Proposed Refactoring Plan

I'll introduce a `metric_type` parameter across the relevant functions, which can be set to either `'discrete'` or `'continuous'` to control the behavior.

Here’s a diagram illustrating the proposed workflow:

```mermaid
graph TD
    subgraph src/metrics.py
        A[eval_one_expl_type(..., metric_type, steps)] --> B{single_user_metrics};
        B -- discrete --> C[Mask 1 to N items];
        B -- continuous --> D[Mask X% of items];
        C --> E[Calculate Metrics];
        D --> E;
        E --> F[Return Results];
    end

    subgraph src/visualization.py
        G[plot_all_metrics(results, ..., metric_type)] --> H{Check metric_type};
        H -- discrete --> I[Plot vs "Number of Masked Items"];
        H -- continuous --> J[Plot vs "Masked Items Percentage"];
    end

    subgraph Main Script (e.g., scripts/evaluate.py)
        K[main()] --> L1[eval_one_expl_type(..., metric_type='discrete')];
        K --> L2[eval_one_expl_type(..., metric_type='continuous')];
        L1 --> M1[Results_discrete];
        L2 --> M2[Results_continuous];
        M1 --> N1[plot_all_metrics(..., metric_type='discrete')];
        M2 --> N2[plot_all_metrics(..., metric_type='continuous')];
    end

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#f9f,stroke:#333,stroke-width:2px
```

### Detailed Changes

#### `src/metrics.py`

1.  **Refactor `single_user_metrics`:**
    *   I will replace the `masking_percentages` parameter with `metric_type: str` and `steps: int`.
    *   Based on `metric_type`, the function will either generate masking steps as percentages (continuous) or a range of integers (discrete).
    *   I will also add a `mask_by` parameter (`'history'` or `'explanation'`) to handle the logic from `single_user_metrics_by_explanation`.

2.  **Refactor `eval_one_expl_type`:**
    *   This function will also get the `metric_type`, `steps`, and `mask_by` parameters to pass down to `single_user_metrics`.

3.  **Cleanup:**
    *   I will remove `single_user_metrics_by_explanation` and `eval_one_expl_type_by_explanation` as their functionality will be merged into the main functions.

#### `src/visualization.py`

1.  **Consolidate Plotting Functions:**
    *   I will consolidate `plot_all_metrics`, `plot_all_metrics_percentage`, and `plot_all_metrics_by_explanation` into a single, more powerful `plot_all_metrics` function.
    *   This function will accept a `metric_type` parameter to determine the x-axis label and data.
    *   I'll also add a `metrics_to_plot` parameter to allow plotting only a subset of metrics, which will cover the functionality of the current `plot_continuous_metrics` function.

2.  **Cleanup:**
    *   The now-redundant plotting functions will be removed to streamline the code.