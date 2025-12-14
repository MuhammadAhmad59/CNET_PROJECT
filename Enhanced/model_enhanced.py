"""
STDformer-GCN: Enhanced Traffic Flow Prediction Model

This file implements the complete STDformer-GCN architecture with three major enhancements:
1. Learnable Multi-Scale Trend Extraction
2. Hybrid Seasonal Decomposition (FFT + Dilated TCN)
3. GCN-Enhanced Spatial Modeling

Authors: Muhammad Ahmad (22I-1929), Shahzaib Afzal (22I-1956)
Course: CS3001 - Computer Networks
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# ENHANCEMENT A: Learnable Multi-Scale Trend Extractor
# ============================================================================
class LearnableTrendExtractor(nn.Module):
    """
    Multi-scale 1D CNN trend extractor with adaptive fusion.
    Replaces fixed moving average with learnable trend extraction.
    
    Args:
        num_nodes: Number of traffic sensors/nodes
        kernel_sizes: List of kernel sizes for multi-scale extraction (default: [3, 5, 7])
    """
    def __init__(self, num_nodes, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        
        # Parallel convolutional branches for multi-scale trends
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_nodes, num_nodes, kernel_size=k, 
                         padding=k//2, groups=num_nodes),
                nn.BatchNorm1d(num_nodes),
                nn.ReLU()
            ) for k in kernel_sizes
        ])
        
        # Learnable fusion weights with attention
        self.fusion_attn = nn.Sequential(
            nn.Linear(len(kernel_sizes), len(kernel_sizes) * 2),
            nn.ReLU(),
            nn.Linear(len(kernel_sizes) * 2, len(kernel_sizes)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: [B, S, N]
        x_t = x.permute(0, 2, 1)  # [B, N, S]
        
        # Extract multi-scale trends
        trends = []
        for conv in self.convs:
            trend = conv(x_t)
            trends.append(trend)
        
        # Stack trends: [B, N, S, num_scales]
        trends_stack = torch.stack(trends, dim=-1)
        
        # Compute adaptive fusion weights
        # Use global average pooling across spatial and temporal dims
        gap = trends_stack.mean(dim=[1, 2])  # [B, num_scales]
        fusion_weights = self.fusion_attn(gap)  # [B, num_scales]
        
        # Weighted combination
        fusion_weights = fusion_weights.view(-1, 1, 1, len(self.kernel_sizes))
        fused_trend = (trends_stack * fusion_weights).sum(dim=-1)  # [B, N, S]
        
        return fused_trend.permute(0, 2, 1)  # [B, S, N]


# ============================================================================
# ENHANCEMENT B: Hybrid Seasonal Decomposition
# ============================================================================
class DilatedTemporalConvNet(nn.Module):
    """
    Dilated TCN for capturing local seasonal patterns at multiple scales.
    
    Args:
        num_nodes: Number of traffic sensors
        seq_len: Sequence length
        dilation_rates: List of dilation rates (default: [1, 2, 4, 8])
    """
    def __init__(self, num_nodes, seq_len, dilation_rates=[1, 2, 4, 8]):
        super().__init__()
        self.dilation_rates = dilation_rates
        
        # Dilated causal convolutions
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_nodes, num_nodes, kernel_size=3,
                         padding=d, dilation=d, groups=num_nodes),
                nn.BatchNorm1d(num_nodes),
                nn.ReLU()
            ) for d in dilation_rates
        ])
        
        # Residual connections between layers
        self.residual_proj = nn.Conv1d(num_nodes, num_nodes, kernel_size=1)
        
    def forward(self, x):
        # x: [B, S, N]
        x_t = x.permute(0, 2, 1)  # [B, N, S]
        
        # Apply dilated convolutions with residual connections
        out = x_t
        for conv in self.dilated_convs:
            residual = out
            out = conv(out)
            out = out + residual  # Residual connection
        
        return out.permute(0, 2, 1)  # [B, S, N]


class HybridSeasonalDecomposition(nn.Module):
    """
    Combines FFT-based global periodicity with dilated TCN for local patterns.
    
    Args:
        num_nodes: Number of traffic sensors
        seq_len: Sequence length
    """
    def __init__(self, num_nodes, seq_len):
        super().__init__()
        self.seq_len = seq_len
        
        # FFT-based global seasonal extractor
        self.log_mask = nn.Parameter(torch.zeros(seq_len//2 + 1))
        
        # Dilated TCN for local seasonal patterns
        self.tcn = DilatedTemporalConvNet(num_nodes, seq_len)
        
        # Learnable balance parameter between global and local
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # x: [B, S, N]
        
        # Global seasonal extraction via FFT
        Xf = torch.fft.rfft(x, dim=1)
        mask = torch.sigmoid(self.log_mask).unsqueeze(0).unsqueeze(2)
        Xf_filtered = Xf * mask
        seasonal_global = torch.fft.irfft(Xf_filtered, n=self.seq_len, dim=1)
        
        # Local seasonal extraction via dilated TCN
        seasonal_local = self.tcn(x)
        
        # Adaptive fusion
        alpha = torch.sigmoid(self.alpha)
        seasonal_hybrid = alpha * seasonal_global + (1 - alpha) * seasonal_local
        
        return seasonal_hybrid


# ============================================================================
# ENHANCEMENT C: Graph Convolutional Network for Spatial Modeling
# ============================================================================
class GraphConvLayer(nn.Module):
    """
    Single GCN layer with symmetric normalization.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: [B, N, F], adj: [N, N]
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        support = self.linear(x)  # [B, N, out_features]
        output = torch.matmul(adj, support)  # [B, N, out_features]
        return output


class GCNSpatialModule(nn.Module):
    """
    Two-layer GCN for capturing network topology and spatial dependencies.
    
    Args:
        num_nodes: Number of traffic sensors
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate (default: 0.2)
    """
    def __init__(self, num_nodes, hidden_dim, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        
        # GCN layers
        self.gcn1 = GraphConvLayer(num_nodes, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, num_nodes)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Learnable adjacency matrix (if physical connectivity not available)
        self.adaptive_adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
    def get_normalized_adj(self):
        """Compute symmetric normalized adjacency matrix."""
        # Make adjacency symmetric and add self-loops
        adj = torch.sigmoid(self.adaptive_adj)
        adj = (adj + adj.T) / 2
        adj = adj + torch.eye(self.num_nodes, device=adj.device)
        
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        rowsum = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        normalized_adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), 
                                      d_mat_inv_sqrt)
        return normalized_adj
    
    def forward(self, x):
        # x: [B, S, N] - fused temporal features
        # Average over time to get node features
        node_features = x.mean(dim=1)  # [B, N]
        # Expand to match GCN input requirement: [B, N] -> [B, N, num_nodes]
        # Each node gets the same feature vector repeated num_nodes times
        node_features_expanded = node_features.unsqueeze(-1).repeat(1, 1, self.num_nodes)  # [B, N, num_nodes]
        
        adj = self.get_normalized_adj()  # [N, N]
        
        # First GCN layer: [B, N, num_nodes] -> [B, N, hidden_dim]
        h = self.gcn1(node_features_expanded, adj)  # [B, N, hidden_dim]
        h = self.activation(h)
        h = self.dropout(h)
        
        # Second GCN layer: [B, N, hidden_dim] -> [B, N, num_nodes]
        h = self.gcn2(h, adj)  # [B, N, num_nodes]
        
        # Extract node representations: take diagonal elements h[b, n, n] for each batch
        # Use torch.diagonal to extract [B, N] from [B, N, num_nodes]
        # We want h[b, n, n] for all b and n, which is the diagonal along dims 1 and 2
        spatial_features = torch.diagonal(h, dim1=1, dim2=2)  # [B, N]
        
        # Broadcast to temporal dimension
        spatial_features = spatial_features.unsqueeze(1).repeat(1, x.size(1), 1)  # [B, S, N]
        
        return spatial_features


# ============================================================================
# Core Components (from baseline)
# ============================================================================
class RevIN(nn.Module):
    """Reversible Instance Normalization for residual processing."""
    def __init__(self, num_nodes, affine=True, eps=1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.scale = nn.Parameter(torch.ones(1, 1, num_nodes))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_nodes))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            mu = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + self.eps
            x_norm = (x - mu) / std
            if self.affine:
                x_norm = x_norm * self.scale + self.bias
            return x_norm, mu, std
        else:
            return x


class STRA(nn.Module):
    """Spatial-Temporal Relation Attention (from baseline)."""
    def __init__(self, num_nodes, d_model, nhead):
        super().__init__()
        self.node_proj = nn.Linear(num_nodes, d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.out = nn.Linear(d_model, num_nodes)

    def forward(self, x):
        node_tokens = x.mean(dim=1)
        h = self.node_proj(node_tokens)
        q = h.unsqueeze(0)
        k = h.unsqueeze(0)
        v = h.unsqueeze(0)
        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.squeeze(0)
        out = self.out(attn_out).unsqueeze(1).repeat(1, x.size(1), 1)
        return out


class FourierAttentionBlock(nn.Module):
    """
    Fourier Attention mechanism for seasonal component processing.
    Operates in frequency domain to capture periodic patterns.
    """
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Frequency domain projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        self.num_layers = num_layers
        
    def forward(self, x):
        # x: [S, B, D] (sequence_len, batch, d_model)
        
        for _ in range(self.num_layers):
            residual = x
            
            # Convert to frequency domain
            x_freq = torch.fft.rfft(x, dim=0)
            
            # Apply attention in frequency domain
            q = self.q_proj(x_freq.real)
            k = self.k_proj(x_freq.real)
            v = self.v_proj(x_freq.real)
            
            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            attn_out = torch.matmul(attn_weights, v)
            attn_out = self.out_proj(attn_out)
            
            # Convert back to time domain
            attn_out_complex = torch.complex(attn_out, torch.zeros_like(attn_out))
            x = torch.fft.irfft(attn_out_complex, n=x.size(0), dim=0)
            
            # Add & Norm
            x = self.norm1(x + residual)
            
            # Feed-forward
            residual = x
            x = self.ffn(x)
            x = self.norm2(x + residual)
        
        return x


# ============================================================================
# MAIN MODEL: STDformer-GCN
# ============================================================================
class STDformerGCN(nn.Module):
    """
    Enhanced STDformer with:
    - Learnable multi-scale trend extraction
    - Hybrid seasonal decomposition (FFT + Dilated TCN)
    - GCN-based spatial modeling integrated with STRA
    
    Args:
        num_nodes: Number of traffic sensors
        seq_len: Input sequence length (default: 96)
        d_model: Model dimension (default: 32)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
        pred_len: Prediction horizon (default: 12)
        gcn_hidden_dim: GCN hidden dimension (default: 64)
        dropout: Dropout rate (default: 0.2)
    """
    def __init__(self, num_nodes, seq_len=96, d_model=32, nhead=4, 
                 num_layers=2, pred_len=12, gcn_hidden_dim=64, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        
        # ============= Enhanced Decomposition =============
        # Enhancement A: Learnable trend extraction
        self.trend_extractor = LearnableTrendExtractor(num_nodes)
        
        # Enhancement B: Hybrid seasonal decomposition
        self.seasonal_extractor = HybridSeasonalDecomposition(num_nodes, seq_len)
        
        # Residual processing (RevIN + MLP)
        self.revin = RevIN(num_nodes)
        self.residual_mlp = nn.Sequential(
            nn.Linear(num_nodes, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # ============= Temporal Modeling =============
        self.input_proj = nn.Linear(num_nodes, d_model)
        
        # Trend component: Transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=4*d_model,
            dropout=dropout
        )
        self.trend_transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # Seasonal component: Fourier Attention
        self.seasonal_fourier_attention = FourierAttentionBlock(d_model, nhead, num_layers)
        
        # ============= Gating Mechanism =============
        self.gate_proj = nn.Linear(d_model, 3)
        
        # ============= Enhancement C: GCN Spatial Module =============
        self.gcn = GCNSpatialModule(num_nodes, gcn_hidden_dim, dropout)
        
        # ============= STRA (baseline spatial attention) =============
        self.stra = STRA(num_nodes, d_model, nhead)
        
        # ============= Prediction Head =============
        self.pred_head = nn.Linear(d_model, num_nodes)
        
        # Residual connection weight for GCN + STRA fusion
        self.spatial_fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # x: [B, S, N]
        
        # ============= STEP 1: Enhanced Decomposition =============
        # Extract trend component (learnable multi-scale)
        X_trend = self.trend_extractor(x)
        
        # Get preliminary component
        X_prelim = x - X_trend
        
        # Extract seasonal component (hybrid: FFT + TCN)
        X_seasonal = self.seasonal_extractor(X_prelim)
        
        # Extract residual component
        X_residual = X_prelim - X_seasonal
        
        # ============= STEP 2: Temporal Modeling =============
        # Process trend component with Transformer
        trend_proj = self.input_proj(X_trend).permute(1, 0, 2)  # [S, B, D]
        trend_encoded = self.trend_transformer(trend_proj)
        trend_feat = trend_encoded[-1]  # [B, D]
        
        # Process seasonal component with Fourier Attention
        seasonal_proj = self.input_proj(X_seasonal).permute(1, 0, 2)  # [S, B, D]
        seasonal_encoded = self.seasonal_fourier_attention(seasonal_proj)
        seasonal_feat = seasonal_encoded[-1]  # [B, D]
        
        # Process residual component with RevIN + MLP
        residual_norm, mu, sigma = self.revin(X_residual, mode='norm')
        residual_mean = residual_norm.mean(dim=1)  # [B, N]
        residual_feat = self.residual_mlp(residual_mean)  # [B, D]
        
        # ============= STEP 3: Gating Mechanism Fusion =============
        gates = torch.sigmoid(self.gate_proj(trend_feat + seasonal_feat + residual_feat))
        gate_trend, gate_seasonal, gate_residual = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        
        fused_temporal = (gate_trend * trend_feat + 
                         gate_seasonal * seasonal_feat + 
                         gate_residual * residual_feat)  # [B, D]
        
        # Expand to sequence
        fused_temporal_seq = fused_temporal.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, S, D]
        
        # Project back to node dimension for spatial processing
        fused_nodes = self.pred_head(fused_temporal_seq)  # [B, S, N]
        
        # ============= STEP 4: GCN-Enhanced Spatial Modeling =============
        # Apply GCN to capture graph structure
        gcn_spatial = self.gcn(fused_nodes)  # [B, S, N]
        
        # Apply STRA for dynamic attention-based spatial modeling
        stra_spatial = self.stra(fused_nodes)  # [B, S, N]
        
        # Hybrid spatial fusion with learnable weight
        alpha = torch.sigmoid(self.spatial_fusion_weight)
        spatial_features = alpha * gcn_spatial + (1 - alpha) * stra_spatial
        
        # Residual connection with original fused features
        spatial_features = spatial_features + fused_nodes
        
        # ============= STEP 5: Multi-Step Prediction =============
        # Project to d_model for prediction
        spatial_proj = self.input_proj(spatial_features)  # [B, S, D]
        
        # Use final time step for autoregressive prediction
        last_feat = spatial_proj[:, -1, :]  # [B, D]
        
        predictions = []
        current_feat = last_feat
        
        for _ in range(self.pred_len):
            pred_step = self.pred_head(current_feat)  # [B, N]
            predictions.append(pred_step.unsqueeze(1))
            # Update feature for next step (simple approach)
            current_feat = current_feat + self.residual_mlp(pred_step)
        
        predictions = torch.cat(predictions, dim=1)  # [B, pred_len, N]
        
        return predictions


# ============================================================================
# Ablation Variants for Systematic Evaluation
# ============================================================================
class STDformerGCN_NoLearnableTrend(STDformerGCN):
    """Ablation: Remove learnable trend, use fixed moving average."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Import and use fixed moving average
        from model import MovingAvgTrend
        self.trend_extractor = MovingAvgTrend(kernel_size=9)


class STDformerGCN_NoHybridSeasonal(STDformerGCN):
    """Ablation: Remove hybrid seasonal, use only FFT."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simple FFT-only seasonal extractor
        class SimpleFourierFilter(nn.Module):
            def __init__(self, seq_len):
                super().__init__()
                self.seq_len = seq_len
                self.log_mask = nn.Parameter(torch.zeros(seq_len//2+1))
            def forward(self, x):
                Xf = torch.fft.rfft(x, dim=1)
                mask = torch.sigmoid(self.log_mask).unsqueeze(0).unsqueeze(2)
                Xf_filtered = Xf * mask
                return torch.fft.irfft(Xf_filtered, n=self.seq_len, dim=1)
        
        self.seasonal_extractor = SimpleFourierFilter(kwargs.get('seq_len', 96))


class STDformerGCN_NoGCN(STDformerGCN):
    """Ablation: Remove GCN, use only STRA."""
    def forward(self, x):
        # Same as parent but skip GCN
        X_trend = self.trend_extractor(x)
        X_prelim = x - X_trend
        X_seasonal = self.seasonal_extractor(X_prelim)
        X_residual = X_prelim - X_seasonal
        
        trend_proj = self.input_proj(X_trend).permute(1, 0, 2)
        trend_encoded = self.trend_transformer(trend_proj)
        trend_feat = trend_encoded[-1]
        
        seasonal_proj = self.input_proj(X_seasonal).permute(1, 0, 2)
        seasonal_encoded = self.seasonal_fourier_attention(seasonal_proj)
        seasonal_feat = seasonal_encoded[-1]
        
        residual_norm, mu, sigma = self.revin(X_residual, mode='norm')
        residual_mean = residual_norm.mean(dim=1)
        residual_feat = self.residual_mlp(residual_mean)
        
        gates = torch.sigmoid(self.gate_proj(trend_feat + seasonal_feat + residual_feat))
        gate_trend, gate_seasonal, gate_residual = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        
        fused_temporal = (gate_trend * trend_feat + 
                         gate_seasonal * seasonal_feat + 
                         gate_residual * residual_feat)  # [B, D]
        
        # Expand to sequence
        fused_temporal_seq = fused_temporal.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, S, D]
        
        # Project back to node dimension for spatial processing
        fused_nodes = self.pred_head(fused_temporal_seq)  # [B, S, N]
        
        # ============= STEP 4: Spatial Modeling (STRA only, no GCN) =============
        # Apply STRA for dynamic attention-based spatial modeling
        stra_spatial = self.stra(fused_nodes)  # [B, S, N]
        
        # Residual connection with original fused features
        spatial_features = stra_spatial + fused_nodes
        
        # ============= STEP 5: Multi-Step Prediction =============
        # Project to d_model for prediction
        spatial_proj = self.input_proj(spatial_features)  # [B, S, D]
        
        # Use final time step for autoregressive prediction
        last_feat = spatial_proj[:, -1, :]  # [B, D]
        
        predictions = []
        current_feat = last_feat
        
        for _ in range(self.pred_len):
            pred_step = self.pred_head(current_feat)  # [B, N]
            predictions.append(pred_step.unsqueeze(1))
            # Update feature for next step (simple approach)
            current_feat = current_feat + self.residual_mlp(pred_step)
        
        predictions = torch.cat(predictions, dim=1)  # [B, pred_len, N]
        
        return predictions