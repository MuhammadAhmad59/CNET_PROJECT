import os, random, torch, numpy as np, json
from torch.utils.data import DataLoader
from model import TimeSeriesTransformer
from data_utils import load_pems_csv, TrafficDataset, zscore_normalize_train, apply_zscore

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_once(cfg, data_path, dataset_name, seed=42):
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Seed: {seed}")
    print('='*60)
    
    data = load_pems_csv(data_path)
    T = data.shape[0]
    train_cut = int(T*0.7); val_cut = int(T*0.85)
    train_raw = data[:train_cut]; val_raw = data[train_cut:val_cut]; test_raw = data[val_cut:]
    train_norm, mu, sigma = zscore_normalize_train(train_raw)
    val_norm = apply_zscore(val_raw, mu, sigma); test_norm = apply_zscore(test_raw, mu, sigma)
    hist = cfg['history_len']
    
    results_by_pred_len = {}
    
    # Train and evaluate for each prediction length
    for pred in cfg.get('pred_lens', [cfg['pred_lens'][0]]):
        print(f"\n  → Prediction length: {pred}")
        
        train_ds = TrafficDataset(train_norm, hist, pred); val_ds = TrafficDataset(val_norm, hist, pred); test_ds = TrafficDataset(test_norm, hist, pred)
        train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True); val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False)
        model = TimeSeriesTransformer(num_nodes=data.shape[1], seq_len=hist, d_model=cfg['d_model'], nhead=cfg['nhead'], num_layers=cfg['num_layers'], pred_len=pred)
        device = torch.device('cuda' if torch.cuda.is_available() and cfg['device']=='cuda' else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        criterion = torch.nn.MSELoss()
        best_val = 1e9; patience = 0
        
        for epoch in range(cfg['epochs']):
            model.train(); train_loss = 0.0
            for x,y in train_loader:
                x = x.to(device); y = y.to(device)
                predy = model(x); loss = criterion(predy, y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= max(1, len(train_loader.dataset))
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for x,y in val_loader:
                    x = x.to(device); y = y.to(device)
                    predy = model(x); val_loss += criterion(predy, y).item() * x.size(0)
            val_loss /= max(1, len(val_loader.dataset))
            print(f"    Epoch {epoch+1}/{cfg['epochs']} Train={train_loss:.6f} Val={val_loss:.6f}")
            
            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(f'results/{dataset_name}', exist_ok=True)
                torch.save(model.state_dict(), f'results/{dataset_name}/best_model_pred{pred}_seed{seed}.pt')
                patience = 0
            else:
                patience += 1
                if patience >= cfg['early_stopping_patience']:
                    print('    Early stopping'); break
        
        # Comprehensive evaluation on test set
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        model.eval()
        preds=[]; gts=[]
        from torch.utils.data import DataLoader as DL
        test_loader = DL(test_ds, batch_size=cfg['batch_size'], shuffle=False)
        with torch.no_grad():
            for x,y in test_loader:
                x = x.to(device)
                out = model(x).cpu().numpy(); preds.append(out); gts.append(y.numpy())
        
        if len(preds)==0:
            results_by_pred_len[pred] = {'MSE': None, 'MAE': None, 'R2': None, 'MAPE': None, 'Accuracy_10pct': None}
            continue
        
        preds = np.concatenate(preds, axis=0); gts = np.concatenate(gts, axis=0)
        p_flat = preds.reshape(-1); g_flat = gts.reshape(-1)
        
        # Basic metrics
        mse = float(mean_squared_error(g_flat, p_flat))
        mae = float(mean_absolute_error(g_flat, p_flat))
        r2 = float(r2_score(g_flat, p_flat))
        
        # MAPE (Mean Absolute Percentage Error) - avoid division by zero
        epsilon = 1e-8
        mape = float(np.mean(np.abs((g_flat - p_flat) / (g_flat + epsilon))) * 100)
        
        # Accuracy within 10% threshold
        within_threshold = np.abs((g_flat - p_flat) / (g_flat + epsilon)) <= 0.10
        accuracy_10pct = float(np.mean(within_threshold) * 100)
        
        print(f"\n    📊 Test Results (pred_len={pred}):")
        print(f"       MSE: {mse:.6f}")
        print(f"       MAE: {mae:.6f}")
        print(f"       R² Score: {r2:.4f}")
        print(f"       MAPE: {mape:.2f}%")
        print(f"       Accuracy (±10%): {accuracy_10pct:.2f}%")
        
        results_by_pred_len[pred] = {
            'MSE': mse, 
            'MAE': mae, 
            'R2': r2, 
            'MAPE': mape, 
            'Accuracy_10pct': accuracy_10pct
        }
    
    return results_by_pred_len

def main():
    os.makedirs('results', exist_ok=True)
    configs = json.load(open('experiments/config.json'))
    
    # Handle if config is a list of configs (one per dataset) or single dict
    if not isinstance(configs, list):
        configs = [configs]
    
    all_results = {}
    
    # Process each dataset configuration
    for cfg in configs:
        dataset_name = cfg.get('dataset', 'unknown')
        data_path = cfg.get('data_path')
        
        if not data_path:
            print(f"⚠️  Skipping config without data_path: {cfg}")
            continue
        
        print(f"\n{'#'*70}")
        print(f"# PROCESSING DATASET: {dataset_name}")
        print(f"# Path: {data_path}")
        print('#'*70)
        
        dataset_results = []
        
        for seed in cfg.get('random_seeds', [42]):
            m = train_once(cfg, data_path, dataset_name, seed)
            dataset_results.append({'seed': seed, 'results': m})
        
        all_results[dataset_name] = {
            'runs': dataset_results,
            'config': cfg
        }
    
    # Aggregate and print summary
    summary = {'datasets': {}}
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - AVERAGE ACROSS ALL SEEDS")
    print('='*70)
    
    for dataset_name, dataset_data in all_results.items():
        print(f"\n🗂️  DATASET: {dataset_name}")
        print("-" * 70)
        
        # Get pred_lens from this dataset's config
        pred_lens = dataset_data['config'].get('pred_lens', [12])
        summary['datasets'][dataset_name] = {}
        
        for pred_len in pred_lens:
            metric_names = ['MSE', 'MAE', 'R2', 'MAPE', 'Accuracy_10pct']
            avg_metrics = {}
            
            for metric in metric_names:
                values = [run['results'][pred_len][metric] for run in dataset_data['runs']
                         if pred_len in run['results'] and run['results'][pred_len][metric] is not None]
                avg_metrics[f'avg_{metric}'] = float(np.mean(values)) if len(values) > 0 else None
            
            summary['datasets'][dataset_name][f'pred_len_{pred_len}'] = avg_metrics
            
            print(f"\n  📊 Prediction Length: {pred_len}")
            for key, val in avg_metrics.items():
                metric_name = key.replace('avg_', '')
                if val is not None:
                    if 'R2' in key:
                        print(f"     {metric_name:20s}: {val:.4f}")
                    elif 'pct' in key or 'MAPE' in key:
                        print(f"     {metric_name:20s}: {val:.2f}%")
                    else:
                        print(f"     {metric_name:20s}: {val:.6f}")
    
    # Save comprehensive results
    json.dump(summary, open('results/summary_all_datasets.json','w'), indent=2)
    print(f"\n✅ Results saved to: results/summary_all_datasets.json")
    print("="*70)

if __name__=='__main__':
    main()