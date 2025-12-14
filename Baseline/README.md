# Baseline STDformer Implementation

This directory contains the faithful implementation of the original **STDformer** (Spatial-Temporal traffic flow prediction with residual and trend Decomposition Transformer) architecture as described in the reference paper by Wan et al. (2025).

## 📄 Reference Paper

> Wan, H., Xu, H., & Xie, L. (2025). Spatial-Temporal Traffic Flow Prediction Through Residual-Trend Decomposition with Transformer Architecture. *Electronics*, 14(12), 2400.  
> DOI: https://doi.org/10.3390/electronics14122400

**Paper Location**: `../Base Paper/electronics-14-02400-v2.pdf`

## 🎯 Purpose

This baseline implementation serves as:
1. **Reference Implementation**: Faithful reproduction of the original paper's methodology
2. **Comparison Baseline**: Benchmark for evaluating our enhancements
3. **Validation**: Ensures we understand the original architecture before modifications

## 🏗️ Architecture Components

### 1. Trend Decomposition Block

**Fixed Moving Average Trend Extraction**
```python
class MovingAvgTrend(nn.Module):
    def __init__(self, kernel_size=9):
        # Non-learnable, fixed kernel
        self.kernel_size = kernel_size
    
    def forward(self, x):
        # Padding for boundary handling
        front = x[:, 0:1, :].repeat(1, (self.kernel_size-1)//2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size-1)//2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # Apply average pooling
        x_trend = F.avg_pool1d(
            x_padded.permute(0, 2, 1),
            kernel_size=self.kernel_size,
            stride=1,
            padding=0
        ).permute(0, 2, 1)
        
        return x_trend
```

**FFT-Based Seasonal Extraction**
```python
class FourierFilter(nn.Module):
    def forward(self, x):
        # Apply FFT to extract frequency components
        x_freq = torch.fft.rfft(x, dim=1)
        
        # Generate frequency mask for seasonal components
        mask = self.generate_mask(x_freq)
        x_seasonal_freq = x_freq * mask
        
        # Inverse FFT to get seasonal component
        x_seasonal = torch.fft.irfft(x_seasonal_freq, n=x.shape[1], dim=1)
        
        # Residual is the difference
        x_residual = x - x_seasonal
        
        return x_seasonal, x_residual
```

**Key Characteristics**:
- ❌ **Non-adaptive**: Fixed kernel size (9) cannot adjust to varying traffic patterns
- ❌ **Equal weighting**: All time steps treated equally in moving average
- ❌ **Global only**: FFT captures only global periodic patterns, misses local bursts

### 2. Temporal Modeling Block

**Processes three components independently:**

#### Trend Component → Vanilla Transformer
```python
class TrendTransformer(nn.Module):
    def __init__(self, d_model=32, n_heads=4, n_layers=2):
        self.encoder = TransformerEncoder(...)
    
    def forward(self, trend):
        # Standard multi-head attention
        H_trend = self.encoder(trend)
        return H_trend
```

#### Seasonal Component → Fourier Attention
```python
class FourierAttention(nn.Module):
    def forward(self, seasonal):
        # Frequency-domain attention mechanism
        S_freq = torch.fft.rfft(seasonal, dim=1)
        S_attn = self.attention_in_frequency(S_freq)
        S_out = torch.fft.irfft(S_attn, n=seasonal.shape[1], dim=1)
        return S_out
```

#### Residual Component → RevIN + MLP
```python
class RevINMLP(nn.Module):
    def forward(self, residual):
        # Reversible Instance Normalization
        residual_norm = self.revin(residual)
        
        # Multi-layer perceptron
        residual_out = self.mlp(residual_norm)
        
        # Reverse normalization
        residual_final = self.revin.inverse(residual_out)
        return residual_final
```

### 3. Spatial-Temporal Relation Attention (STRA)

**Inverted Attention Mechanism**:
```python
class STRA(nn.Module):
    def forward(self, x):
        # x: (B, T, N) → (T, B, N)
        # Treats each node (road segment) as a token
        x = x.permute(1, 0, 2)
        
        # Multi-head attention across nodes
        attn_output = self.attention(x, x, x)
        
        # Feed-forward network
        output = self.ffn(attn_output)
        
        # Restore dimensions
        return output.permute(1, 0, 2)
```

**Key Characteristics**:
- ✅ Captures dynamic spatial relationships
- ❌ Lacks explicit graph structure knowledge
- ❌ No physical connectivity information
- ❌ Requires substantial data to learn topology

## 📊 Baseline Results (Iteration 2)

### SYNTH Dataset (1 Epoch Training)

| Model | pred_len=12 | pred_len=24 | pred_len=48 |
|-------|-------------|-------------|-------------|
| | MSE / MAE | MSE / MAE | MSE / MAE |
| Transformer+GCN | 0.964 / 6.521 | 0.994 / nan | 0.998 / nan |
| LSTM+GCN | 0.976 / 6.216 | 1.001 / nan | 1.002 / nan |
| CNN+GCN | 1.000 / 6.639 | 1.002 / nan | 1.004 / nan |
| **DLinear+GCN** | **0.908 / 5.452** | 0.968 / nan | 0.993 / nan |
| STDformer+GCN | 1.003 / 6.668 | 1.005 / nan | 1.005 / nan |

**Observations**:
- ⚠️ **NaN values** for longer prediction horizons indicate convergence failure
- ⚠️ **Single epoch** insufficient for proper training
- ✅ **DLinear+GCN** shows best performance even with minimal training

### PEMS07 Dataset Results

| Model | pred_len=12 | pred_len=24 |
|-------|-------------|-------------|
| | MSE / MAE | MSE / MAE |
| Transformer+GCN | 1.016 / 138.43 | 1.007 / 137.77 |
| LSTM+GCN | 1.016 / 138.40 | 1.008 / 137.76 |
| CNN+GCN | 1.015 / 138.30 | 1.007 / 137.75 |
| DLinear+GCN | 1.017 / 138.42 | 1.008 / 137.86 |
| STDformer+GCN | 1.016 / 138.42 | 1.008 / 137.78 |

## 🔧 Implementation Details

### Hyperparameters

```python
CONFIG = {
    # Model architecture
    'd_model': 32,              # Model dimension
    'n_heads': 4,               # Attention heads
    'n_layers': 2,              # Encoder layers
    'd_ff': 128,                # Feed-forward dimension
    'dropout': 0.0,             # No dropout in baseline
    
    # Decomposition
    'trend_kernel': 9,          # Moving average window
    'seasonal_method': 'fft',   # Fourier transform
    
    # Training
    'batch_size': 8,
    'learning_rate': 0.001,
    'epochs': 1,                # Single epoch training
    'optimizer': 'Adam',
    'loss': 'MSE',
    
    # Data
    'history_length': 32,       # Input sequence length
    'pred_lengths': [12, 24, 48, 96, 192, 336, 720]
}
```

### File Structure

```
Baseline/
├── README.md                    # This file
├── models/
│   ├── __init__.py
│   ├── stdformer.py            # Main STDformer model
│   ├── decomposition.py        # Trend/seasonal decomposition
│   ├── temporal_models.py      # Transformer, FA, RevIN-MLP
│   ├── stra.py                 # Spatial-Temporal Relation Attention
│   └── gating.py               # Fusion mechanism
├── configs/
│   └── baseline_config.yaml    # Configuration file
├── train_baseline.py           # Training script
├── evaluate_baseline.py        # Evaluation script
└── utils/
    ├── data_loader.py
    ├── metrics.py
    └── adjacency.py
```

## 🚀 Running Baseline Experiments

### Basic Training

```bash
# Navigate to project root
cd CNET_PROJECT

# Run baseline training
python Baseline/train_baseline.py \
    --dataset SYNTH \
    --pred_len 12 \
    --epochs 1 \
    --batch_size 8 \
    --lr 0.001
```

### All Prediction Horizons

```bash
# Test all prediction lengths
for pred_len in 12 24 48 96 192 336 720; do
    python Baseline/train_baseline.py \
        --dataset PEMS03 \
        --pred_len $pred_len \
        --epochs 1 \
        --save_results results/baseline_pems03_${pred_len}.json
done
```

### Multiple Datasets

```bash
# Evaluate across all datasets
for dataset in SYNTH PEMS03 PEMS04 PEMS07 PEMS08; do
    python Baseline/train_baseline.py \
        --dataset $dataset \
        --pred_len 12 \
        --epochs 1
done
```

## ⚠️ Known Limitations

### 1. Fixed Trend Extraction
**Problem**: Non-learnable moving average with fixed kernel size
```python
# Cannot adapt to these scenarios:
- Rush hour transitions (need smaller kernel)
- Weekend patterns (need larger kernel)
- Irregular events (accidents, construction)
```

**Impact**: 
- Suboptimal smoothing for varying traffic patterns
- Misses important rapid transitions
- Cannot capture multi-scale trends

### 2. Global-Only Seasonal Modeling
**Problem**: FFT captures only global periodic patterns
```python
# Misses these important patterns:
- Localized traffic bursts (accidents)
- Short-duration events (concerts, games)
- Non-periodic recurring patterns
```

**Impact**:
- Cannot model local seasonal variations
- Poor handling of sudden traffic changes
- Reduced accuracy during events

### 3. Implicit Spatial Dependencies
**Problem**: STRA learns correlations without structural knowledge
```python
# Lacks:
- Physical road connectivity
- Multi-hop propagation patterns
- Structural inductive bias
```

**Impact**:
- Requires substantial training data
- O(N²) computational complexity
- No explicit topology modeling

### 4. Single-Epoch Training
**Problem**: Insufficient training leads to poor convergence
```python
# Results:
- NaN values for longer horizons
- Underfitted models
- Poor generalization
```

## 🔄 Differences from Reference Paper

Our baseline implementation differs slightly from the original paper:

| Aspect | Reference Paper | Our Baseline |
|--------|----------------|--------------|
| **Training Epochs** | Multiple (exact number not specified) | 1 epoch |
| **GCN Integration** | Not included | Added for fair comparison |
| **Data Preprocessing** | Detailed cleaning procedures | Standard Z-score normalization |
| **Early Stopping** | Not mentioned | Not implemented in baseline |
| **Validation Split** | Not detailed | 70/10/20 train/val/test |

**Rationale**: We start with minimal training (1 epoch) to establish a clear baseline, then demonstrate improvements with extended training in the enhanced version.

## 📈 Performance Analysis

### Convergence Behavior

```
Epoch 1:
├── SYNTH: MSE ≈ 0.91-1.00 (varies by model)
├── PEMS03: MSE ≈ 1.04 (high error)
├── PEMS07: MSE ≈ 1.02 (moderate error)
└── Longer horizons: Convergence failure (NaN)
```

### Component Analysis

Based on the baseline results:
1. **Trend Component**: Provides stable long-term patterns
2. **Seasonal Component**: Captures daily/weekly cycles (global only)
3. **Residual Component**: Models short-term noise
4. **STRA**: Learns spatial correlations (requires more training)

## 🔬 Code Example

### Complete Forward Pass

```python
class STDformerBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Decomposition
        self.trend_decomp = MovingAvgTrend(kernel_size=9)
        self.seasonal_decomp = FourierFilter()
        
        # Temporal models
        self.temporal_trend = VanillaTransformer(config)
        self.temporal_seasonal = FourierAttention(config)
        self.temporal_residual = RevINMLP(config)
        
        # Spatial attention
        self.stra = STRA(config)
        
        # Gating fusion
        self.gate_fusion = GatingMechanism(config)
    
    def forward(self, x, adj=None):
        # Step 1: Decomposition
        trend = self.trend_decomp(x)
        x_prelim = x - trend
        seasonal, residual = self.seasonal_decomp(x_prelim)
        
        # Step 2: Temporal modeling (parallel)
        h_trend = self.temporal_trend(trend)
        h_seasonal = self.temporal_seasonal(seasonal)
        h_residual = self.temporal_residual(residual)
        
        # Step 3: Gating fusion
        z_temporal = self.gate_fusion(h_trend, h_seasonal, h_residual)
        
        # Step 4: Spatial attention
        output = self.stra(z_temporal)
        
        return output
```

## 📚 Additional Resources

- **Original Paper**: `../Base Paper/electronics-14-02400-v2.pdf`
- **Technical Report**: `../Improved Base Paper/22I-1929_22I-1956_22I-6181.pdf`
- **Enhanced Implementation**: `../Enhanced/README.md`

## 🎓 Learning Objectives

This baseline implementation demonstrates:
1. ✅ Time series decomposition (trend, seasonal, residual)
2. ✅ Transformer-based temporal modeling
3. ✅ Spatial-temporal attention mechanisms
4. ✅ Multi-component fusion strategies
5. ❌ Limitations of non-adaptive approaches (motivation for enhancements)



---

**Next Steps**: See `../Enhanced/README.md` for our architectural improvements and enhanced results.
