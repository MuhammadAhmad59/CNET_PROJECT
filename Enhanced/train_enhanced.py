import os
import random
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Import baseline model and enhanced model
from model import TimeSeriesTransformer
# Assuming the enhanced model is saved as model_enhanced.py
try:
    from model_enhanced import (
        STDformerGCN, 
        STDformerGCN_NoLearnableTrend,
        STDformerGCN_NoHybridSeasonal,
        STDformerGCN_NoGCN
    )
except ImportError:
    print("⚠️  Enhanced model not found. Please save the STDformerGCN code as 'model_enhanced.py'")
    STDformerGCN = None

from data_utils import load_pems_csv, TrafficDataset, zscore_normalize_train, apply_zscore


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(predictions, ground_truth):
    """Compute comprehensive evaluation metrics."""
    p_flat = predictions.reshape(-1)
    g_flat = ground_truth.reshape(-1)
    
    mse = float(mean_squared_error(g_flat, p_flat))
    mae = float(mean_absolute_error(g_flat, p_flat))
    rmse = float(np.sqrt(mse))
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    }


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        pred_y = model(x)
        loss = criterion(pred_y, y)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    
    return total_loss / max(1, len(train_loader.dataset))


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred_y = model(x)
            loss = criterion(pred_y, y)
            total_loss += loss.item() * x.size(0)
    
    return total_loss / max(1, len(val_loader.dataset))


def evaluate_model(model, test_loader, device):
    """Comprehensive evaluation on test set."""
    model.eval()
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            predictions.append(pred)
            ground_truths.append(y.numpy())
    
    if len(predictions) == 0:
        return None
    
    predictions = np.concatenate(predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)
    
    return compute_metrics(predictions, ground_truths)


def train_model(model_class, cfg, data_path, dataset_name, seed, model_name="Model"):
    """
    Train a single model configuration.
    """
    set_seed(seed)
    
    print(f"\n{'='*70}")
    print(f"Training {model_name} on {dataset_name} | Seed: {seed}")
    print('='*70)
    
    # Load and prepare data
    data = load_pems_csv(data_path)
    T = data.shape[0]
    train_cut = int(T * 0.7)
    val_cut = int(T * 0.85)
    
    train_raw = data[:train_cut]
    val_raw = data[train_cut:val_cut]
    test_raw = data[val_cut:]
    
    train_norm, mu, sigma = zscore_normalize_train(train_raw)
    val_norm = apply_zscore(val_raw, mu, sigma)
    test_norm = apply_zscore(test_raw, mu, sigma)
    
    hist_len = cfg['history_len']
    results_by_pred_len = {}
    
    # Train for each prediction length
    for pred_len in cfg.get('pred_lens', [12]):
        print(f"\n  → Prediction length: {pred_len}")
        
        # Create datasets
        train_ds = TrafficDataset(train_norm, hist_len, pred_len)
        val_ds = TrafficDataset(val_norm, hist_len, pred_len)
        test_ds = TrafficDataset(test_norm, hist_len, pred_len)
        
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], 
                                  shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], 
                               shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], 
                                shuffle=False, num_workers=2, pin_memory=True)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() and cfg['device'] == 'cuda' else 'cpu')
        
        if model_class == STDformerGCN or model_class.__name__.startswith('STDformerGCN'):
            model = model_class(
                num_nodes=data.shape[1],
                seq_len=hist_len,
                d_model=cfg['d_model'],
                nhead=cfg['nhead'],
                num_layers=cfg['num_layers'],
                pred_len=pred_len,
                gcn_hidden_dim=cfg.get('gcn_hidden_dim', 64),
                dropout=cfg.get('dropout', 0.2)
            )
        else:
            # Baseline model
            model = model_class(
                num_nodes=data.shape[1],
                seq_len=hist_len,
                d_model=cfg['d_model'],
                nhead=cfg['nhead'],
                num_layers=cfg['num_layers'],
                pred_len=pred_len
            )
        
        model = model.to(device)
        
        # Optimizer with weight decay
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg['learning_rate'],
            weight_decay=cfg.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        criterion = torch.nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        start_time = time.time()
        
        for epoch in range(cfg['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate(model, val_loader, criterion, device)
            
            # Step scheduler
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d}/{cfg['epochs']} | "
                      f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                save_dir = f"results/{dataset_name}/{model_name}"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"{save_dir}/best_pred{pred_len}_seed{seed}.pt"
                )
            else:
                patience_counter += 1
                if patience_counter >= cfg['early_stopping_patience']:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        print(f"    ✓ Training completed in {training_time:.2f}s (best epoch: {best_epoch})")
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(f"{save_dir}/best_pred{pred_len}_seed{seed}.pt"))
        metrics = evaluate_model(model, test_loader, device)
        
        if metrics:
            print(f"\n    📊 Test Results (pred_len={pred_len}):")
            print(f"       MSE:  {metrics['MSE']:.6f}")
            print(f"       MAE:  {metrics['MAE']:.6f}")
            print(f"       RMSE: {metrics['RMSE']:.6f}")
            
            metrics['training_time'] = training_time
            metrics['best_epoch'] = best_epoch
            results_by_pred_len[pred_len] = metrics
        else:
            results_by_pred_len[pred_len] = None
    
    return results_by_pred_len


def run_ablation_study(cfg, data_path, dataset_name, seeds):
    """
    Run comprehensive ablation study comparing all variants.
    """
    print(f"\n{'#'*70}")
    print(f"# ABLATION STUDY: {dataset_name}")
    print('#'*70)
    
    if STDformerGCN is None:
        print("⚠️  Enhanced model not available. Skipping ablation study.")
        return {}
    
    models_to_test = {
        'STDformer-GCN (Full)': STDformerGCN,
        'STDformer-GCN (No Learnable Trend)': STDformerGCN_NoLearnableTrend,
        'STDformer-GCN (No Hybrid Seasonal)': STDformerGCN_NoHybridSeasonal,
        'STDformer-GCN (No GCN)': STDformerGCN_NoGCN,
        'STDformer (Baseline)': TimeSeriesTransformer
    }
    
    all_results = {}
    
    for model_name, model_class in models_to_test.items():
        model_results = []
        
        for seed in seeds:
            results = train_model(model_class, cfg, data_path, dataset_name, seed, model_name)
            model_results.append({'seed': seed, 'results': results})
        
        all_results[model_name] = model_results
    
    return all_results


def print_comparison_table(all_results, pred_lens):
    """Print comparison table across all models."""
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS - AVERAGE ACROSS ALL SEEDS")
    print('='*80)
    
    for pred_len in pred_lens:
        print(f"\n📊 Prediction Horizon: {pred_len} steps")
        print("-" * 80)
        print(f"{'Model':<40} {'MSE':>12} {'MAE':>12} {'RMSE':>12}")
        print("-" * 80)
        
        for model_name, results_list in all_results.items():
            # Average metrics across seeds
            metrics_list = []
            for run in results_list:
                if pred_len in run['results'] and run['results'][pred_len]:
                    metrics_list.append(run['results'][pred_len])
            
            if len(metrics_list) > 0:
                avg_mse = np.mean([m['MSE'] for m in metrics_list])
                avg_mae = np.mean([m['MAE'] for m in metrics_list])
                avg_rmse = np.mean([m['RMSE'] for m in metrics_list])
                
                print(f"{model_name:<40} {avg_mse:>12.6f} {avg_mae:>12.6f} {avg_rmse:>12.6f}")
        
        print("-" * 80)


def main():
    """Main training pipeline with ablation studies."""
    os.makedirs('results', exist_ok=True)
    
    # Load configuration
    configs = json.load(open('experiments/config.json'))
    if not isinstance(configs, list):
        configs = [configs]
    
    # Run experiments for each dataset
    for cfg in configs:
        dataset_name = cfg.get('dataset', 'unknown')
        data_path = cfg.get('data_path')
        
        if not data_path or not os.path.exists(data_path):
            print(f"⚠️  Skipping {dataset_name}: data path not found")
            continue
        
        print(f"\n{'#'*70}")
        print(f"# PROCESSING DATASET: {dataset_name}")
        print(f"# Path: {data_path}")
        print('#'*70)
        
        seeds = cfg.get('random_seeds', [42])
        
        # Run ablation study
        ablation_results = run_ablation_study(cfg, data_path, dataset_name, seeds)
        
        # Print comparison table
        if ablation_results:
            print_comparison_table(ablation_results, cfg.get('pred_lens', [12]))
            
            # Save results
            save_path = f'results/{dataset_name}/ablation_results.json'
            with open(save_path, 'w') as f:
                json.dump(ablation_results, f, indent=2, default=str)
            print(f"\n✅ Results saved to: {save_path}")


if __name__ == '__main__':
    main()