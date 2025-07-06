import os
import numpy as np
import matplotlib.pyplot as plt

def plot_all_metrics(results, data_name, recommender_name, metric_type='discrete', steps=5, metrics_to_plot=None):
    """
    Plots all metrics for all baselines.
    """
    all_metrics = {
        'DEL': ('DEL@Ke. Lower values are better', 'DEL@Kₑ', 'Lower is better'),
        'INS': ('INS@Ke. Higher values are better', 'INS@Kₑ', 'Higher is better'),
        'NDCG': ('CDCG@Ke. Lower values are better', 'CNDCG@Kₑ', 'Lower is better'),
        'POS_at_20': ('POSat20. Lower values are better', 'POSat20', 'Lower is better')
    }
    
    if metrics_to_plot:
        metrics_mapping = {k: v for k, v in all_metrics.items() if k in metrics_to_plot}
    else:
        metrics_mapping = all_metrics

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'x']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    
    os.makedirs('results/plots', exist_ok=True)

    for metric, (title_name, y_label, _) in metrics_mapping.items():
        plt.figure(figsize=(12, 8))
        
        # Create a common x-axis for all baselines
        if metric_type == 'continuous':
            x_values = np.linspace(0, 1, steps)
        
        for i, (baseline, baseline_metrics) in enumerate(results.items()):
            if metric not in baseline_metrics:
                continue
            
            values = baseline_metrics[metric][:steps]
            
            if metric_type == 'discrete':
                x_values = range(1, len(values) + 1)
                if metric == 'INS':
                    plt.xlabel("Number of Added Items", fontsize=30)
                else:
                    plt.xlabel("Number of Masked Items", fontsize=30)
                plt.xticks(range(1, len(values) + 1), fontsize=18)
            else: # continuous
                if metric == 'INS':
                    plt.xlabel("Percentage of Added Items", fontsize=30)
                else:
                    plt.xlabel("Masked Items Percentage", fontsize=30)
                plt.xticks(np.arange(0, 1.1, 0.2), fontsize=18)
            
            plt.plot(
                x_values, values, label=baseline.upper(), color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)], marker=markers[i % len(markers)],
                markersize=8, linewidth=2, markevery=1
            )
        
        all_values = np.concatenate([b[metric][:steps] for b in results.values() if metric in b])
        y_min, y_max = all_values.min(), all_values.max()
        y_range = y_max - y_min
        plt.ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

        plt.ylabel(y_label, fontsize=30)
        plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
        plt.yticks(fontsize=18)
        plt.legend(loc='best', fontsize=20, frameon=True, edgecolor='black')
        plt.title(title_name, fontsize=30)
        plt.tight_layout()
        
        safe_display_name = title_name.replace(" ", "_").replace("@", "at")
        filename = f'results/plots/{safe_display_name}_{data_name}_{recommender_name}_{metric_type}.pdf'
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"Saved plot: {filename}")
        plt.close()

import seaborn as sns
def plot_continuous_metric_distributions(results, data_name, recommender_name):
    """
    Creates box plots showing the distribution of DEL, INS, NDCG across methods.
    """
    import pandas as pd
    metrics_to_plot = ['DEL', 'INS', 'NDCG']
    plot_data = []
    for method, metrics in results.items():
        for metric in metrics_to_plot:
            if metric in metrics:
                values = metrics[metric]
                for step, value in enumerate(values, 1):
                    plot_data.append({
                        'Method': method.upper(),
                        'Metric': metric,
                        'Step': step,
                        'Value': value
                    })
    if not plot_data:
        print("No data found for continuous metrics.")
        return
    df = pd.DataFrame(plot_data)
    import matplotlib.pyplot as plt
    import os
    os.makedirs('results/plots', exist_ok=True)
    plt.figure(figsize=(15, 10))
    for idx, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(1, 3, idx)
        sns.boxplot(data=df[df['Metric'] == metric], x='Method', y='Value')
        plt.title(f'{metric} Distribution')
        plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = f'results/plots/metric_distributions_{data_name}_{recommender_name}.pdf'
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"Saved box plot for continuous metrics: {out_path}")
    plt.close()