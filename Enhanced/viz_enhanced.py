import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_ablation_results(dataset_name):
    """Load ablation study results for a dataset."""
    path = f'results/{dataset_name}/ablation_results.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def plot_ablation_comparison(dataset_name, pred_lens, save_dir='results/figures'):
    """
    Plot comparison of all ablation variants across prediction horizons.
    """
    results = load_ablation_results(dataset_name)
    if not results:
        print(f"No results found for {dataset_name}")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data
    models = list(results.keys())
    metrics = ['MSE', 'MAE', 'RMSE']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Ablation Study Results - {dataset_name}', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for model_name in models:
            model_results = results[model_name]
            
            # Extract metric values for each prediction length
            metric_values = []
            for pred_len in pred_lens:
                # Average across seeds
                values = []
                for run in model_results:
                    if pred_len in run['results'] and run['results'][pred_len]:
                        values.append(run['results'][pred_len][metric])
                
                if values:
                    metric_values.append(np.mean(values))
                else:
                    metric_values.append(np.nan)
            
            # Plot line
            line_style = '-o' if 'Full' in model_name else '--s'
            line_width = 2.5 if 'Full' in model_name else 1.5
            ax.plot(pred_lens, metric_values, line_style, 
                   label=model_name.replace('STDformer-GCN ', ''),
                   linewidth=line_width, markersize=6)
        
        ax.set_xlabel('Prediction Horizon (steps)', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} vs Prediction Horizon', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xticks(pred_lens)
        ax.set_xticklabels(pred_lens)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{dataset_name}_ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/{dataset_name}_ablation_comparison.png")


def plot_improvement_heatmap(datasets, pred_lens, save_dir='results/figures'):
    """
    Plot heatmap showing improvement of full model over baseline.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    improvement_data = []
    
    for dataset in datasets:
        results = load_ablation_results(dataset)
        if not results:
            continue
        
        baseline_results = results.get('STDformer (Baseline)', [])
        enhanced_results = results.get('STDformer-GCN (Full)', [])
        
        if not baseline_results or not enhanced_results:
            continue
        
        dataset_improvements = []
        
        for pred_len in pred_lens:
            # Get baseline MSE
            baseline_mse = []
            for run in baseline_results:
                if pred_len in run['results'] and run['results'][pred_len]:
                    baseline_mse.append(run['results'][pred_len]['MSE'])
            
            # Get enhanced MSE
            enhanced_mse = []
            for run in enhanced_results:
                if pred_len in run['results'] and run['results'][pred_len]:
                    enhanced_mse.append(run['results'][pred_len]['MSE'])
            
            if baseline_mse and enhanced_mse:
                avg_baseline = np.mean(baseline_mse)
                avg_enhanced = np.mean(enhanced_mse)
                improvement = ((avg_baseline - avg_enhanced) / avg_baseline) * 100
                dataset_improvements.append(improvement)
            else:
                dataset_improvements.append(0)
        
        improvement_data.append(dataset_improvements)
    
    if not improvement_data:
        print("No data available for heatmap")
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(improvement_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=40)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pred_lens)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(pred_lens)
    ax.set_yticklabels(datasets)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(pred_lens)):
            text = ax.text(j, i, f'{improvement_data[i][j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)
    ax.set_title('STDformer-GCN Improvement over Baseline (MSE Reduction %)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/improvement_heatmap.png")


def plot_component_contribution(dataset_name, pred_len=12, save_dir='results/figures'):
    """
    Plot bar chart showing contribution of each enhancement.
    """
    results = load_ablation_results(dataset_name)
    if not results:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Define model variants and their missing components
    variants = {
        'Baseline': 'STDformer (Baseline)',
        '+ GCN': 'STDformer-GCN (No GCN)',
        '+ Hybrid Seasonal': 'STDformer-GCN (No Hybrid Seasonal)',
        '+ Learnable Trend': 'STDformer-GCN (No Learnable Trend)',
        'Full Model': 'STDformer-GCN (Full)'
    }
    
    mse_values = []
    mae_values = []
    labels = []
    
    for label, model_name in variants.items():
        if model_name not in results:
            continue
        
        model_results = results[model_name]
        mse_list = []
        mae_list = []
        
        for run in model_results:
            if pred_len in run['results'] and run['results'][pred_len]:
                mse_list.append(run['results'][pred_len]['MSE'])
                mae_list.append(run['results'][pred_len]['MAE'])
        
        if mse_list:
            mse_values.append(np.mean(mse_list))
            mae_values.append(np.mean(mae_list))
            labels.append(label)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Component Contribution Analysis - {dataset_name} (pred_len={pred_len})',
                fontsize=14, fontweight='bold')
    
    x = np.arange(len(labels))
    width = 0.6
    
    # MSE plot
    bars1 = ax1.bar(x, mse_values, width, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])
    ax1.set_ylabel('MSE', fontsize=11)
    ax1.set_title('Mean Squared Error', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # MAE plot
    bars2 = ax2.bar(x, mae_values, width, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])
    ax2.set_ylabel('MAE', fontsize=11)
    ax2.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{dataset_name}_component_contribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/{dataset_name}_component_contribution.png")


def plot_training_efficiency(datasets, save_dir='results/figures'):
    """
    Compare training time and final performance across models.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Efficiency Analysis', fontsize=16, fontweight='bold')
    
    for idx, dataset in enumerate(datasets):
        results = load_ablation_results(dataset)
        if not results:
            continue
        
        ax = axes[idx // 2, idx % 2]
        
        models = ['STDformer (Baseline)', 'STDformer-GCN (No GCN)', 'STDformer-GCN (Full)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for model_name, color in zip(models, colors):
            if model_name not in results:
                continue
            
            model_results = results[model_name]
            
            # Extract training time and MSE for pred_len=12
            times = []
            mses = []
            
            for run in model_results:
                if 12 in run['results'] and run['results'][12]:
                    res = run['results'][12]
                    if 'training_time' in res and res['training_time']:
                        times.append(res['training_time'])
                        mses.append(res['MSE'])
            
            if times and mses:
                ax.scatter(np.mean(times), np.mean(mses), s=200, c=color, 
                          label=model_name.replace('STDformer-GCN ', ''), alpha=0.7)
        
        ax.set_xlabel('Training Time (seconds)', fontsize=11)
        ax.set_ylabel('Test MSE', fontsize=11)
        ax.set_title(dataset, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_dir}/training_efficiency.png")


def generate_summary_table(datasets, save_dir='results/figures'):
    """
    Generate LaTeX table summarizing results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    latex_content = []
    latex_content.append("\\begin{table}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{STDformer-GCN Performance Summary}")
    latex_content.append("\\begin{tabular}{lcccccc}")
    latex_content.append("\\hline")
    latex_content.append("Dataset & Model & MSE & MAE & RMSE & R² & Improvement \\\\")
    latex_content.append("\\hline")
    
    for dataset in datasets:
        results = load_ablation_results(dataset)
        if not results:
            continue
        
        # Get baseline and enhanced results for pred_len=12
        baseline = results.get('STDformer (Baseline)', [])
        enhanced = results.get('STDformer-GCN (Full)', [])
        
        if baseline and enhanced:
            # Average metrics across seeds
            baseline_metrics = {'MSE': [], 'MAE': [], 'RMSE': [], 'R2': []}
            enhanced_metrics = {'MSE': [], 'MAE': [], 'RMSE': [], 'R2': []}
            
            for run in baseline:
                if 12 in run['results'] and run['results'][12]:
                    for key in baseline_metrics:
                        baseline_metrics[key].append(run['results'][12][key])
            
            for run in enhanced:
                if 12 in run['results'] and run['results'][12]:
                    for key in enhanced_metrics:
                        enhanced_metrics[key].append(run['results'][12][key])
            
            if baseline_metrics['MSE'] and enhanced_metrics['MSE']:
                # Calculate averages
                baseline_avg = {k: np.mean(v) for k, v in baseline_metrics.items()}
                enhanced_avg = {k: np.mean(v) for k, v in enhanced_metrics.items()}
                
                improvement = ((baseline_avg['MSE'] - enhanced_avg['MSE']) / baseline_avg['MSE']) * 100
                
                # Add baseline row
                latex_content.append(
                    f"{dataset} & Baseline & {baseline_avg['MSE']:.4f} & "
                    f"{baseline_avg['MAE']:.4f} & {baseline_avg['RMSE']:.4f} & "
                    f"{baseline_avg['R2']:.4f} & - \\\\"
                )
                
                # Add enhanced row
                latex_content.append(
                    f" & Enhanced & {enhanced_avg['MSE']:.4f} & "
                    f"{enhanced_avg['MAE']:.4f} & {enhanced_avg['RMSE']:.4f} & "
                    f"{enhanced_avg['R2']:.4f} & {improvement:.1f}\\% \\\\"
                )
                latex_content.append("\\hline")
    
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    
    # Save to file
    with open(f'{save_dir}/results_table.tex', 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"✓ Saved: {save_dir}/results_table.tex")


def main():
    """Generate all visualizations."""
    print("="*70)
    print("Generating Enhanced Visualizations for STDformer-GCN")
    print("="*70)
    
    datasets = ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']
    pred_lens = [12, 24, 48, 96, 192, 336, 720]
    
    # 1. Ablation comparison plots for each dataset
    print("\n1. Generating ablation comparison plots...")
    for dataset in datasets:
        plot_ablation_comparison(dataset, pred_lens)
    
    # 2. Improvement heatmap
    print("\n2. Generating improvement heatmap...")
    plot_improvement_heatmap(datasets, pred_lens)
    
    # 3. Component contribution analysis
    print("\n3. Generating component contribution analysis...")
    for dataset in datasets:
        plot_component_contribution(dataset, pred_len=12)
    
    # 4. Training efficiency comparison
    print("\n4. Generating training efficiency comparison...")
    plot_training_efficiency(datasets)
    
    # 5. Generate LaTeX summary table
    print("\n5. Generating LaTeX summary table...")
    generate_summary_table(datasets)
    
    print("\n" + "="*70)
    print("✅ All visualizations generated successfully!")
    print("📁 Check the 'results/figures/' directory for output files")
    print("="*70)


if __name__ == '__main__':
    main()