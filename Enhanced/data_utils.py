import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    def __init__(self, data_array, hist_len=96, pred_len=12):
        self.hist_len = hist_len
        self.pred_len = pred_len
        X, Y = [], []
        T, N = data_array.shape
        for i in range(T - hist_len - pred_len + 1):
            x = data_array[i:i+hist_len]
            y = data_array[i+hist_len:i+hist_len+pred_len]
            X.append(x)
            Y.append(y)
        if len(X)==0:
            self.X = np.zeros((0, hist_len, N), dtype=np.float32)
            self.Y = np.zeros((0, pred_len, N), dtype=np.float32)
        else:
            self.X = np.stack(X).astype('float32')
            self.Y = np.stack(Y).astype('float32')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.Y[idx])

def load_pems_csv(path_csv):
    """
    Load PEMS CSV file, automatically handling headers and non-numeric columns
    """
    try:
        # First, try reading with header (most common case)
        df = pd.read_csv(path_csv)
        
        # Check if all columns are numeric
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] == df.shape[1]:
            # All columns are numeric, we're good
            print(f"  ✓ Loaded with header, shape: {numeric_df.shape}")
            return numeric_df.values.astype('float32')
        elif numeric_df.shape[1] > 0:
            # Some columns are numeric, use only those
            print(f"  ⚠️  Dropped {df.shape[1] - numeric_df.shape[1]} non-numeric columns")
            print(f"  ✓ Loaded shape: {numeric_df.shape}")
            return numeric_df.values.astype('float32')
        else:
            # No numeric columns found, try without header
            raise ValueError("No numeric columns with header")
            
    except:
        # Try loading without header
        try:
            df = pd.read_csv(path_csv, header=None)
            
            # Try to convert all to numeric, if it fails, there's a header row
            try:
                data = df.values.astype('float32')
                print(f"  ✓ Loaded without header, shape: {data.shape}")
                return data
            except ValueError:
                # First row is likely a header, skip it
                df = pd.read_csv(path_csv, header=0)
                numeric_df = df.select_dtypes(include=[np.number])
                
                if numeric_df.shape[1] == 0:
                    raise ValueError(f"No numeric data found in {path_csv}")
                
                print(f"  ✓ Skipped header row, loaded shape: {numeric_df.shape}")
                return numeric_df.values.astype('float32')
                
        except Exception as e:
            raise ValueError(f"Failed to load {path_csv}: {str(e)}")

def zscore_normalize_train(data_train):
    mu = data_train.mean(axis=0, keepdims=True)
    sigma = data_train.std(axis=0, keepdims=True) + 1e-6
    data_norm = (data_train - mu) / sigma
    return data_norm, mu, sigma

def apply_zscore(data, mu, sigma):
    return (data - mu) / sigma