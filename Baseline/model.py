import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAvgTrend(nn.Module):
    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        b, s, n = x.shape
        x_t = x.permute(0,2,1)
        pad = self.kernel_size//2
        x_pad = F.pad(x_t, (pad,pad), mode='replicate')
        trend = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        trend = trend[:,:,:s]
        trend = trend.permute(0,2,1)
        return trend

class RevIN(nn.Module):
    def __init__(self, num_nodes, affine=True, eps=1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        if affine:
            self.scale = nn.Parameter(torch.ones(1,1,num_nodes))
            self.bias = nn.Parameter(torch.zeros(1,1,num_nodes))

    def forward(self, x, mode='norm'):
        if mode=='norm':
            mu = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + self.eps
            x_norm = (x - mu) / std
            if self.affine:
                x_norm = x_norm * self.scale + self.bias
            return x_norm, mu, std
        else:
            return x

class FourierFilter(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.log_mask = nn.Parameter(torch.zeros(seq_len//2+1))

    def forward(self, x):
        Xf = torch.fft.rfft(x, dim=1)
        mask = torch.sigmoid(self.log_mask).unsqueeze(0).unsqueeze(2)
        Xf_filtered = Xf * mask
        xr = torch.fft.irfft(Xf_filtered, n=self.seq_len, dim=1)
        return xr

class STRA(nn.Module):
    def __init__(self, num_nodes, d_model, nhead):
        super().__init__()
        self.node_proj = nn.Linear(num_nodes, d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.out = nn.Linear(d_model, num_nodes)

    def forward(self, x):
        node_tokens = x.mean(dim=1)
        h = self.node_proj(node_tokens)
        q = h.unsqueeze(0); k = h.unsqueeze(0); v = h.unsqueeze(0)
        attn_out, _ = self.attn(q,k,v)
        attn_out = attn_out.squeeze(0)
        out = self.out(attn_out).unsqueeze(1).repeat(1, x.size(1), 1)
        return out

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_nodes, seq_len=32, d_model=32, nhead=4, num_layers=2, pred_len=12):
        super().__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.input_proj = nn.Linear(num_nodes, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.trend_transformer = self.encoder
        self.seasonal_transformer = self.encoder
        self.residual_mlp = nn.Sequential(nn.Linear(num_nodes, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.pred_head = nn.Linear(d_model, num_nodes)
        self.fourier = FourierFilter(seq_len)
        self.trend_extractor = MovingAvgTrend(kernel_size=9)
        self.revin = RevIN(num_nodes)
        self.stra = STRA(num_nodes, d_model, nhead)
        self.gate_proj = nn.Linear(d_model, 3)
        self.pred_len = pred_len

    def forward(self, x):
        Xt = self.trend_extractor(x)
        Xw = x - Xt
        Xs = self.fourier(Xw)
        Xr = Xw - Xs
        t = self.input_proj(Xt).permute(1,0,2)
        t_enc = self.trend_transformer(t)
        t_feat = t_enc[-1]
        s = self.input_proj(Xs).permute(1,0,2)
        s_enc = self.seasonal_transformer(s)
        s_feat = s_enc[-1]
        r_norm, mu, sigma = self.revin(Xr, mode='norm')
        r_mean = r_norm.mean(dim=1)
        r_feat = self.residual_mlp(r_mean)
        gates = torch.sigmoid(self.gate_proj(t_feat + s_feat + r_feat))
        gT, gS, gR = gates[:,0:1], gates[:,1:2], gates[:,2:3]
        fused = gT * t_feat + gS * s_feat + gR * r_feat
        preds = []
        last = fused
        for _ in range(self.pred_len):
            out = self.pred_head(last)
            preds.append(out.unsqueeze(1))
        preds = torch.cat(preds, dim=1)
        return preds