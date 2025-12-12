
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set style
sns.set_style("whitegrid")
sns.set_context("talk")

MODELS = ['gaussian_mf', 'poisson_mf', 'hpf_cavi', 'hpf_pytorch']
PRED_BASE_DIR = 'data/predictions'
OUTPUT_DIR = 'reports/figures/forecast_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MSE': mse, 'R2': r2}

def plot_true_vs_pred_box(df, model_name, ax):
    """Boxplot of predictions for each true integer rating"""
    sns.boxplot(x='y_true_int', y='y_pred', data=df, ax=ax, palette="viridis")
    ax.set_title(f'{model_name}: Preds vs True')
    ax.set_xlabel('True Rating')
    ax.set_ylabel('Predicted Rating')
    # Add diagonal line
    ax.plot([0, 5], [0, 5], ls="--", c=".3")

def plot_residuals(df, model_name, ax):
    """Distribution of residuals"""
    residuals = df['y_true'] - df['y_pred']
    sns.histplot(residuals, kde=True, ax=ax, color='blue', alpha=0.6)
    ax.set_title(f'{model_name}: Residuals (True - Pred)')
    ax.set_xlabel('Residual')

def plot_pred_hist_by_true_value(df, model_name, output_dir):
    """
    For each integer true value, plot a histogram of predictions.
    Annotate with count and proportion.
    """
    unique_trues = sorted(df['y_true_int'].unique())
    n_plots = len(unique_trues)
    total_samples = len(df)
    
    # Create a figure with a grid of subplots
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    
    for i, true_val in enumerate(unique_trues):
        ax = axes[i]
        subset = df[df['y_true_int'] == true_val]
        count = len(subset)
        prop = count / total_samples * 100
        
        sns.histplot(subset['y_pred'], ax=ax, bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f'True Rating: {true_val}\nCount: {count} ({prop:.1f}%)')
        ax.set_xlabel('Predicted Value')
        ax.set_ylabel('Frequency')
        
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
        
    fig.suptitle(f'{model_name}: Predictions by True Value', fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{model_name}_pred_hist_by_true.png'), dpi=300)
    plt.close(fig)

def main():
    results = []
    
    # Setup global figures
    fig_box, axes_box = plt.subplots(2, 2, figsize=(16, 12))
    axes_box = axes_box.flatten()
    
    fig_res, axes_res = plt.subplots(2, 2, figsize=(16, 12))
    axes_res = axes_res.flatten()

    for idx, model in enumerate(MODELS):
        file_path = os.path.join(PRED_BASE_DIR, model, 'test_predictions.csv')
        
        if not os.path.exists(file_path):
            print(f"Warning: Predictions for {model} not found at {file_path}")
            continue
            
        print(f"Analyzing {model}...")
        df = pd.read_csv(file_path)
        
        # Clip predictions to reasonable range for analysis [0, 6]? or just raw?
        # Let's keep raw but maybe note if they are wild.
        
        # Round true ratings to int just in case
        df['y_true_int'] = df['y_true'].round().astype(int)
        
        metrics = compute_metrics(df['y_true'], df['y_pred'])
        metrics['Model'] = model
        results.append(metrics)
        
        # Plots
        if idx < len(axes_box):
            plot_true_vs_pred_box(df, model, axes_box[idx])
            plot_residuals(df, model, axes_res[idx])
        
        # New Histogram Plot
        plot_pred_hist_by_true_value(df, model, OUTPUT_DIR)

    # Save Metrics
    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df)
    
    results_path = os.path.join('reports', 'forecast_metrics.csv')
    results_df.to_csv(results_path, index=False)
    
    # Save formatted markdown table
    md_path = os.path.join('reports', 'forecast_analysis.md')
    with open(md_path, 'w') as f:
        f.write("# Forecast Analysis Results\n\n")
        # Manual markdown table
        f.write("| " + " | ".join(results_df.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(results_df.columns)) + " |\n")
        for _, row in results_df.iterrows():
            f.write("| " + " | ".join(str(x) for x in row.values) + " |\n")
            
        f.write("\n\n## Plots\n")
        f.write(f"![RMSE Comparison](figures/forecast_analysis/rmse_comparison.png)\n")
        f.write(f"![Preds vs True](figures/forecast_analysis/preds_vs_true_box.png)\n")
        f.write(f"![Residuals](figures/forecast_analysis/residuals.png)\n")
        
        f.write("\n### Predictions by True Value\n")
        for model in MODELS:
            f.write(f"#### {model}\n")
            f.write(f"![{model} Histograms](figures/forecast_analysis/{model}_pred_hist_by_true.png)\n")

    # Save Figures
    fig_box.tight_layout()
    fig_box.savefig(os.path.join(OUTPUT_DIR, 'preds_vs_true_box.png'), dpi=300)
    
    fig_res.tight_layout()
    fig_res.savefig(os.path.join(OUTPUT_DIR, 'residuals.png'), dpi=300)
    
    # Additional Plot: Bar chart of metrics
    if not results_df.empty:
        fig_bar, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Model', y='RMSE', data=results_df, palette='magma', ax=ax)
        ax.set_title('RMSE Comparison by Model')
        ax.set_ylim(0, results_df['RMSE'].max() * 1.1)
        for i, v in enumerate(results_df['RMSE']):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
        fig_bar.tight_layout()
        fig_bar.savefig(os.path.join(OUTPUT_DIR, 'rmse_comparison.png'), dpi=300)

    print(f"\nAnalysis complete. Results saved to {results_path} and {md_path}")
    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
