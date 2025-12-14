# Traffic Flow Prediction Using STDformer with GCN Enhancement

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Team Members**:  
> - Muhammad Ahmad (22I-1929)  
> - Shahzaib Afzal (22I-1956)  
> - Uzair Siddique (22I-6181)

## 📋 Project Overview

This repository contains a comprehensive implementation and evaluation of **STDformer-GCN**, an enhanced traffic flow prediction model that integrates Graph Convolutional Networks with the baseline STDformer architecture. Our work builds upon the research paper by Wan et al. (2025) and introduces three major architectural improvements.

### Key Achievements

- ✅ **96.5% MSE reduction** on synthetic data (LSTM+GCN model)
- ✅ **98.6% MSE reduction** on synthetic data (DLinear+GCN model)
- ✅ **7.2% MSE improvement** on PEMS07 dataset (pred_len=24)
- ✅ **Comprehensive evaluation** across 5 datasets and 7 prediction horizons
- ✅ **Detailed ablation study** demonstrating component contributions

### Major Contributions

1. **Learnable Multi-Scale Trend Extraction**: Replaces fixed moving averages with adaptive 1D CNNs (kernel sizes: 3, 5, 7)
2. **Hybrid Seasonal Decomposition**: Combines FFT with dilated temporal convolutions for multi-resolution pattern capture
3. **GCN Spatial Module**: Explicitly models road network topology with 2-layer GCN stack
4. **Extended Training Framework**: 10 epochs with early stopping, enabling proper model convergence

## 🗂️ Repository Structure

```
CNET_PROJECT/
├── Base Paper/              # Original STDformer paper (Wan et al., 2025)
│   └── electronics-14-02400-v2.pdf
│
├── Improved Base Paper/     # Our technical report and analysis
│   └── 22I-1929_22I-1956_22I-6181.pdf
│
├── Baseline/               # Original STDformer implementation
│   ├── models/            # Baseline model components
│   ├── train_baseline.py  # Training script (1 epoch)
│   └── README.md          # Baseline documentation
│
├── Enhanced/              # STDformer-GCN enhancements
│   ├── models/           # Enhanced components
│   │   ├── learnable_trend.py
│   │   ├── hybrid_seasonal.py
│   │   └── gcn_spatial.py
│   ├── train_enhanced.py # Training script (10 epochs)
│   └── README.md         # Enhancement documentation
│
├── data/                 # Datasets and preprocessing
│   ├── SYNTH/           # Synthetic traffic data
│   ├── PEMS03/          # Real-world traffic flow
│   ├── PEMS04/          # Real-world traffic speed
│   ├── PEMS07/          # Real-world traffic flow
│   ├── PEMS08/          # Real-world traffic speed
│   └── README.md        # Data documentation
│
├── experiments/         # Training and evaluation scripts
│   ├── run_baseline.py
│   ├── run_enhanced.py
│   └── run_ablation.py
│
├── results/            # Experimental results
│   ├── tables/        # Performance metrics (CSV/JSON)
│   ├── figures/       # Visualizations and plots
│   └── checkpoints/   # Saved model weights
│
├── requirements.txt   # Python dependencies
├── LICENSE           # MIT License
└── README.md        # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# CUDA-capable GPU (recommended)
nvidia-smi
```

### Installation

```bash
# Clone the repository
git clone https://github.com/MuhammadAhmad59/CNET_PROJECT.git
cd CNET_PROJECT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

```bash
# Datasets are already included in the repository
# Verify dataset structure
ls data/SYNTH data/PEMS03 data/PEMS04 data/PEMS07 data/PEMS08
```

### Running Experiments

#### Baseline Model (1 Epoch)
```bash
python experiments/run_baseline.py \
    --dataset SYNTH \
    --pred_len 12 \
    --epochs 1 \
    --batch_size 8
```

#### Enhanced Model (10 Epochs)
```bash
python experiments/run_enhanced.py \
    --dataset SYNTH \
    --pred_len 12 \
    --epochs 10 \
    --batch_size 8 \
    --gcn_hidden 64 \
    --dropout 0.2
```

#### Ablation Studies
```bash
# Test without learnable trend
python experiments/run_ablation.py --variant no_learnable_trend

# Test without hybrid seasonal
python experiments/run_ablation.py --variant no_hybrid_seasonal

# Test without GCN
python experiments/run_ablation.py --variant no_gcn

# Full model
python experiments/run_ablation.py --variant full
```

## 📊 Key Results

### Performance Improvements (SYNTH Dataset, pred_len=12)

| Model | Baseline MSE | Enhanced MSE | Improvement |
|-------|--------------|--------------|-------------|
| Transformer+GCN | 0.964 | 0.713 | 26.0% |
| **LSTM+GCN** | 0.976 | **0.034** | **96.5%** |
| CNN+GCN | 1.000 | 0.701 | 29.9% |
| **DLinear+GCN** | 0.908 | **0.013** | **98.6%** |
| STDformer+GCN | 1.003 | 0.614 | 38.8% |

### Real-World Performance (PEMS07, pred_len=24)

| Model | Baseline MSE | Enhanced MSE | Improvement |
|-------|--------------|--------------|-------------|
| Transformer+GCN | 1.007 | 0.963 | 4.4% |
| **LSTM+GCN** | 1.008 | **0.935** | **7.2%** |
| STDformer+GCN | 1.008 | 0.940 | 6.7% |

### Ablation Study (Average Across All Datasets)

| Configuration | Avg MSE | Avg MAE | Avg RMSE |
|--------------|---------|---------|----------|
| STDformer-GCN (Full) | 1.169 | 0.900 | 1.080 |
| **No Learnable Trend** | **1.145** | **0.892** | **1.068** |
| No Hybrid Seasonal | 1.171 | 0.903 | 1.082 |
| No GCN | 1.186 | 0.909 | 1.088 |
| Baseline STDformer | 1.137 | 0.888 | 1.061 |

**Key Finding**: The "No Learnable Trend" variant achieves the best average performance, suggesting that fixed moving averages may be more stable for the current training setup.

## 🏗️ Architecture Overview

### Enhanced STDformer-GCN Pipeline

```
Input Sequence (B×T×N)
        ↓
┌─────────────────────────────┐
│ Learnable Trend Extraction  │
│ - Multi-scale CNNs (3,5,7)  │
│ - Adaptive fusion            │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ Hybrid Seasonal Decomp      │
│ - FFT (global patterns)     │
│ - Dilated TCN (local bursts)│
└─────────────────────────────┘
        ↓
  [Trend | Seasonal | Residual]
        ↓
┌─────────────────────────────┐
│ Temporal Modeling (Parallel)│
│ - Transformer (trend)       │
│ - Fourier Attn (seasonal)   │
│ - RevIN-MLP (residual)      │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ Gating Fusion               │
│ - Learnable gates           │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ GCN Spatial Module          │
│ - 2-layer GCN stack         │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│ STRA (Spatial-Temporal      │
│       Relation Attention)   │
└─────────────────────────────┘
        ↓
    Output Predictions (B×F×N)
```

## 📈 Datasets

| Dataset | Nodes | Length | Frequency | Type |
|---------|-------|--------|-----------|------|
| SYNTH | 10 | 499 | 5 min | Synthetic |
| PEMS03 | 358 | 547 | 5 min | Traffic Flow |
| PEMS04 | 307 | 340 | 5 min | Traffic Speed |
| PEMS07 | 883 | 866 | 5 min | Traffic Flow |
| PEMS08 | 170 | 295 | 5 min | Traffic Speed |

All datasets collected at 5-minute intervals with Z-score normalization. Train/validation/test split: 70%/10%/20%.

## 🔬 Technical Details

### Hyperparameters

| Parameter | Baseline | Enhanced |
|-----------|----------|----------|
| Model Dimension (d_model) | 32 | 32 |
| GCN Hidden Dimension | N/A | 64 |
| Attention Heads | 4 | 4 |
| Encoder Layers | 2 | 2 |
| Dropout | 0.0 | 0.2 |
| Batch Size | 8 | 8 |
| Learning Rate | 0.001 | 0.001 |
| Epochs | 1 | 10 |
| Early Stopping Patience | N/A | 5 |

### Training Environment

- **Hardware**: NVIDIA GeForce RTX 4090 24GB
- **Operating System**: Windows 11
- **Framework**: PyTorch 1.12+
- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Loss Function**: Mean Squared Error (MSE)

## 📚 Documentation

- **[Base Paper](Base%20Paper/electronics-14-02400-v2.pdf)**: Original STDformer research
- **[Technical Report](Improved%20Base%20Paper/22I-1929_22I-1956_22I-6181.pdf)**: Complete implementation analysis
- **[Baseline README](Baseline/README.md)**: Original implementation details
- **[Enhanced README](Enhanced/README.md)**: Enhancement documentation
- **[Data README](data/README.md)**: Dataset information

## 🤝 Team Contributions

| Member | Responsibilities |
|--------|-----------------|
| **Muhammad Ahmad** (22I-1929) | GCN Integration, Adjacency Matrix Construction, Ablation Studies |
| **Shahzaib Afzal** (22I-1956) | Learnable Trend Extraction, Hybrid Seasonal Decomposition, Training Framework |
| **Uzair Siddique** (22I-6181) | Experimental Evaluation, Performance Analysis, Data Pipeline, Visualization |

All team members contributed to architecture design, code review, documentation, and problem-solving.

## 📖 Citations

### Reference Paper

```bibtex
@article{wan2025stdformer,
  title={Spatial-Temporal Traffic Flow Prediction Through Residual-Trend Decomposition with Transformer Architecture},
  author={Wan, Hongyang and Xu, Haijiao and Xie, Liang},
  journal={Electronics},
  volume={14},
  number={12},
  pages={2400},
  year={2025},
  publisher={MDPI}
}
```

### Our Work

```bibtex
@techreport{ahmad2025stdformergcn,
  title={Traffic Flow Prediction Using STDformer with GCN Enhancement: Implementation and Performance Analysis},
  author={Ahmad, Muhammad and Afzal, Shahzaib and Siddique, Uzair},
  year={2025},
  institution={CS3001 - Computer Networks, Fall 2025},
  type={Course Project Report}
}
```

## ⚠️ Known Limitations

1. **Long-Horizon Predictions**: Both baseline and enhanced models fail on 720-step predictions
2. **Small Datasets**: Degraded performance on PEMS04 and PEMS08 (limited training samples)
3. **Dataset-Specific Behavior**: PEMS03 shows mixed results requiring further investigation
4. **Computational Cost**: 33% memory increase, 10-15% longer training time per epoch

## 🔮 Future Work

- [ ] Hierarchical multi-stage forecasting for long horizons (720+ steps)
- [ ] Transfer learning from large datasets to small datasets
- [ ] Dataset-specific hyperparameter tuning and adaptive learning rates
- [ ] Physical road network integration with actual connectivity data
- [ ] Real-time adaptation and online learning mechanisms
- [ ] Uncertainty quantification with confidence intervals
- [ ] Model compression for deployment efficiency

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original STDformer authors (Wan et al., 2025) for the foundational architecture
- PEMS database for providing publicly accessible traffic data
- PyTorch and open-source community for development tools
- Course instructor and teaching assistants for guidance and support

## 📞 Contact

For questions, collaborations, or feedback:

- Muhammad Ahmad: [ahmad22i1929@gmail.com]
- Shahzaib Afzal: [shahzaib22i1956@gmail.com]
- Uzair Siddique: [uzair22i6181@gmail.com]

---

**Academic Context**: This work was developed as part of CS3001 - Computer Networks (Fall 2025) and represents a research-based implementation project combining computer networks principles with advanced machine learning techniques for traffic flow prediction.
