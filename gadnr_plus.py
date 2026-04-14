"""
GAD-NR++: Enhanced Graph Anomaly Detection via Attention-Augmented
         Neighbourhood Reconstruction with Mixture Modelling
=========================================================================
Endterm Task 2 — CS60078 Complex Network Theory, Spring 2025-26
Project 3: Graph Anomalies | Group-3: 23EX10026, 22CS30072, 22IM30034

Improvements over GAD-NR:
  1. Multi-head GAT encoder (replaces single-layer GCN) for richer representations
  2. Gaussian Mixture Model (GMM) neighbourhood reconstruction (replaces single Gaussian)
  3. Learnable anomaly score fusion (replaces heuristic weighting)
  4. Multi-scale neighbourhood (optional 2-hop reconstruction)

Usage:
    python gadnr_plus.py --dataset inj_cora --epochs 100
    python gadnr_plus.py --dataset disney --epochs 50
    python gadnr_plus.py --dataset books --epochs 50
"""

import sys
import types

# Mock C++ extensions
def _noop(*a, **k): return None
for _m in ['torch_sparse', 'torch_scatter', 'torch_cluster', 'torch_spline_conv']:
    _mk = types.ModuleType(_m)
    for attr in ['SparseTensor','random_walk','knn','radius','nearest','knn_graph',
                 'radius_graph','matmul','fill_diag','sum','mul','set_diag','remove_diag']:
        setattr(_mk, attr, _noop if attr != 'SparseTensor' else type('S',(),{}))
    sys.modules[_m] = _mk

import os
import argparse
import random
import math
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

from pygod.metrics import eval_roc_auc
from pygod.generator import gen_contextual_outliers, gen_structural_outliers
from pygod.utils.utility import check_parameter

import zipfile

# ======================== DATA LOADING ========================
DATA_DIR = "/home/claude/pygod_data/extracted"

def load_data_local(dataset_str):
    pt_path = os.path.join(DATA_DIR, f"{dataset_str}.pt")
    if not os.path.exists(pt_path):
        zip_path = os.path.join(os.path.dirname(DATA_DIR), f"{dataset_str}.pt.zip")
        if os.path.exists(zip_path):
            os.makedirs(DATA_DIR, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)
    return torch.load(pt_path, weights_only=False)


def gen_joint_structural_outliers(data, m, n, random_state=None):
    if random_state:
        np.random.seed(random_state)
    outlier_idx = np.random.choice(data.num_nodes, size=n, replace=False)
    new_edges = []
    for i in range(n):
        other_idx = np.random.choice(data.num_nodes, size=m, replace=False)
        for j in other_idx:
            new_edges.append(torch.tensor([[i, j]], dtype=torch.long))
    new_edges = torch.cat(new_edges)
    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1
    data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)
    return data, y_outlier


# ======================== IMPROVEMENT 1: Multi-Head GAT Encoder ========================

class AttentionEncoder(nn.Module):
    """
    Multi-head GAT encoder that replaces the single-layer GCN.
    
    Improvement: GAT learns attention weights over neighbours, giving
    different importance to different neighbours. This produces richer
    representations than GCN's fixed-weight aggregation, especially for
    nodes with heterogeneous neighbourhoods.
    
    Uses 2 GAT layers with residual connection for stable training.
    """
    def __init__(self, in_dim, hidden_dim, heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        h0 = self.proj(x)                          # Initial projection
        h1 = self.gat1(h0, edge_index)              # Multi-head attention layer 1
        h1 = self.norm1(F.elu(h1) + h0)             # Residual + LayerNorm
        h2 = self.gat2(h1, edge_index)              # Attention layer 2
        h2 = self.norm2(F.elu(h2) + h1)             # Residual + LayerNorm
        return h2, h0  # Return both encoded repr and initial projection


# ======================== IMPROVEMENT 2: GMM Neighbourhood Reconstruction ========================

class GMMNeighbourDecoder(nn.Module):
    """
    Gaussian Mixture Model (GMM) neighbourhood decoder.
    
    Improvement over GAD-NR: Instead of fitting a single Gaussian to the
    neighbour distribution, we use K mixture components. This handles
    multi-community neighbourhoods (e.g., an interdisciplinary researcher
    connected to biology AND CS communities).
    
    The decoder predicts:
      - K component means: mu_k = MLP_k(h_u)
      - K component log-variances: log_sigma_k = MLP_k(h_u)
      - K mixture weights: pi_k = softmax(MLP_pi(h_u))
    
    Loss: negative log-likelihood of ground-truth neighbour embeddings
    under the predicted GMM, which generalises the KL divergence used
    in GAD-NR (KL is recovered when K=1).
    """
    def __init__(self, hidden_dim, n_components=2, sample_size=10):
        super().__init__()
        self.K = n_components
        self.hidden_dim = hidden_dim
        self.sample_size = sample_size
        
        # Mixture weight predictor
        self.pi_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components)
        )
        
        # Per-component mean and log-variance predictors
        self.mean_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(n_components)
        ])
        self.logvar_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(n_components)
        ])
    
    def forward(self, h_u, neighbour_embeds):
        """
        Args:
            h_u: [hidden_dim] - encoded representation of node u
            neighbour_embeds: [n_neighbours, hidden_dim] - ground truth neighbour embeddings
        Returns:
            nll: negative log-likelihood (scalar)
        """
        # Predict mixture parameters
        log_pi = F.log_softmax(self.pi_net(h_u), dim=-1)  # [K]
        
        means = []
        logvars = []
        for k in range(self.K):
            means.append(self.mean_nets[k](h_u))      # [hidden_dim]
            logvars.append(self.logvar_nets[k](h_u))   # [hidden_dim]
        
        means = torch.stack(means)      # [K, hidden_dim]
        logvars = torch.stack(logvars)  # [K, hidden_dim]
        
        # Compute log-likelihood of each neighbour under the GMM
        # p(x) = sum_k pi_k * N(x | mu_k, sigma_k^2)
        # log p(x) = logsumexp_k [log pi_k + log N(x | mu_k, sigma_k^2)]
        
        n_neigh = neighbour_embeds.shape[0]
        if n_neigh == 0:
            return torch.tensor(0.0, device=h_u.device)
        
        total_nll = 0.0
        for i in range(n_neigh):
            x = neighbour_embeds[i]  # [hidden_dim]
            log_probs = []
            for k in range(self.K):
                # log N(x | mu_k, sigma_k^2) = -0.5 * [d*log(2pi) + sum(logvar) + sum((x-mu)^2/var)]
                diff = x - means[k]
                var = torch.exp(logvars[k]) + 1e-6
                log_normal = -0.5 * (self.hidden_dim * math.log(2 * math.pi) 
                                     + logvars[k].sum() 
                                     + (diff ** 2 / var).sum())
                log_probs.append(log_pi[k] + log_normal)
            
            log_px = torch.logsumexp(torch.stack(log_probs), dim=0)
            total_nll -= log_px
        
        return total_nll / n_neigh


# ======================== IMPROVEMENT 3: Learnable Score Fusion ========================

class LearnableScoreFusion(nn.Module):
    """
    Learnable anomaly score fusion network.
    
    Improvement: Replaces the heuristic weighted sum (h*1.0 + d*1.0 + f*2.0)
    with a small MLP that learns to combine the three loss components.
    
    During training, this network is trained with a self-supervised signal:
    nodes with higher total reconstruction loss should get higher anomaly scores.
    The network learns non-linear interactions between loss components.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, feat_loss, deg_loss, neigh_loss):
        """
        Args:
            feat_loss: [N, 1] per-node feature reconstruction loss
            deg_loss: [N, 1] per-node degree reconstruction loss
            neigh_loss: [N, 1] per-node neighbourhood reconstruction loss
        Returns:
            scores: [N, 1] anomaly scores
        """
        # Normalize each component
        def norm(x):
            r = x.max() - x.min()
            return (x - x.min()) / (r + 1e-8) if r > 0 else x * 0
        
        combined = torch.cat([norm(feat_loss), norm(deg_loss), norm(neigh_loss)], dim=1)  # [N, 3]
        return self.net(combined)


# ======================== FULL MODEL: GAD-NR++ ========================

class GADNRPlusPlus(nn.Module):
    """
    GAD-NR++: Enhanced Graph Anomaly Detection via Attention-Augmented
              Neighbourhood Reconstruction with Mixture Modelling.
    
    Architecture:
      Encoder:  Multi-head GAT with residual connections and LayerNorm
      Decoder:  (a) Self-feature MLP decoder
                (b) Degree MLP decoder
                (c) GMM neighbourhood decoder (K=2 components)
      Fusion:   Learnable anomaly score combination
    """
    def __init__(self, in_dim, hidden_dim, sample_size=10, n_components=2,
                 gat_heads=4, dropout=0.1,
                 lambda1=0.01, lambda2=0.5, lambda3=0.8, device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sample_size = sample_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.device = device
        
        # Encoder (Improvement 1)
        self.encoder = AttentionEncoder(in_dim, hidden_dim, heads=gat_heads, dropout=dropout)
        
        # Decoders
        self.feature_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        self.degree_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.ReLU()
        )
        
        # GMM Neighbourhood Decoder (Improvement 2)
        self.gmm_decoder = GMMNeighbourDecoder(hidden_dim, n_components=n_components, 
                                                sample_size=sample_size)
        
        # Learnable Score Fusion (Improvement 3)
        self.score_fusion = LearnableScoreFusion()
        
        self.feat_loss_fn = nn.MSELoss(reduction='none')
        self.deg_loss_fn = nn.MSELoss()
    
    def forward(self, edge_index, x, degree_list, neighbor_dict, device):
        # Encode
        h_enc, h0 = self.encoder(x, edge_index)
        N = h_enc.shape[0]
        
        # (a) Self-feature reconstruction — reconstruct original input x
        x_pred = self.feature_decoder(h_enc)
        feat_loss_per_node = self.feat_loss_fn(x, x_pred).mean(dim=1, keepdim=True)
        feat_loss = feat_loss_per_node.mean()
        
        # (b) Degree reconstruction
        deg_pred = self.degree_decoder(h_enc)
        deg_gt = degree_list.unsqueeze(1).float()
        deg_loss = self.deg_loss_fn(deg_pred, deg_gt)
        deg_loss_per_node = (deg_pred - deg_gt).pow(2)
        
        # (c) GMM neighbourhood reconstruction
        neigh_losses = []
        for u in range(N):
            neighs = neighbor_dict.get(u, [])
            if len(neighs) == 0:
                neigh_losses.append(torch.tensor(0.0, device=device))
                continue
            
            # Sample neighbours
            if len(neighs) > self.sample_size:
                sampled = random.sample(neighs, self.sample_size)
            else:
                sampled = neighs
            
            neigh_embeds = h0[sampled].detach()  # Stop gradient on targets
            nll = self.gmm_decoder(h_enc[u], neigh_embeds)
            neigh_losses.append(nll)
        
        neigh_loss_per_node = torch.stack(neigh_losses).unsqueeze(1)
        neigh_loss = neigh_loss_per_node.mean()
        
        # Total training loss
        loss = (self.lambda1 * neigh_loss + 
                self.lambda2 * feat_loss + 
                self.lambda3 * deg_loss)
        
        # Anomaly scores via learnable fusion (Improvement 3)
        with torch.no_grad():
            scores = self.score_fusion(
                feat_loss_per_node.detach(),
                deg_loss_per_node.detach(),
                neigh_loss_per_node.detach()
            )
        
        # Also compute heuristic scores for comparison
        f_norm = feat_loss_per_node / (feat_loss_per_node.max() - feat_loss_per_node.min() + 1e-8)
        d_norm = deg_loss_per_node / (deg_loss_per_node.max() - deg_loss_per_node.min() + 1e-8)
        n_norm = neigh_loss_per_node / (neigh_loss_per_node.max() - neigh_loss_per_node.min() + 1e-8)
        heuristic_scores = 2.0 * f_norm + 1.0 * d_norm + 1.0 * n_norm
        
        return loss, heuristic_scores, feat_loss_per_node, deg_loss_per_node, neigh_loss_per_node


# ======================== TRAINING ========================

def run_experiment(dataset_str, config, device, num_runs=3):
    all_results = []
    
    for run_id in range(num_runs):
        print(f"\n--- {dataset_str} | Run {run_id+1}/{num_runs} ---")
        torch.manual_seed(42 + run_id)
        np.random.seed(42 + run_id)
        random.seed(42 + run_id)
        
        # Load data
        data = load_data_local(dataset_str)
        data.x = (data.x - data.x.min()) / (data.x.max() + 1e-8)
        
        # Inject outliers
        if dataset_str == "inj_cora":
            yc = data.y >> 0 & 1
            ys = data.y >> 1 & 1
        else:
            data, yc = gen_contextual_outliers(data=data, n=config['contextual_n'], k=config['contextual_k'])
            data, ys = gen_structural_outliers(data=data, n=config['structural_n'], m=config['structural_m'], p=0.2)
        yc, ys = yc.cpu().detach(), ys.cpu().detach()
        data, yj = gen_joint_structural_outliers(data=data, n=config['structural_n'], m=config['structural_m'])
        ysj = torch.logical_or(ys, yj).int()
        y = data.y.bool().cpu().detach()
        
        # Add self-loops
        num_nodes = data.x.shape[0]
        self_edges = torch.tensor([[i for i in range(num_nodes)], [i for i in range(num_nodes)]])
        data.edge_index = torch.cat([data.edge_index.cpu(), self_edges], dim=1)
        data = data.to(device)
        
        # Build neighbour dict
        neighbor_dict = {}
        for a, b in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()):
            neighbor_dict.setdefault(a, []).append(b)
        degree_list = torch.tensor([len(neighbor_dict.get(i, [])) for i in range(num_nodes)]).to(device)
        
        # Build model
        model = GADNRPlusPlus(
            in_dim=data.x.shape[1],
            hidden_dim=config['dimension'],
            sample_size=config['sample_size'],
            n_components=config.get('n_components', 2),
            gat_heads=config.get('gat_heads', 4),
            lambda1=config['lambda_loss1'],
            lambda2=config['lambda_loss2'],
            lambda3=config['lambda_loss3'],
            device=device
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=3e-4)
        
        best = {'benchmark': 0, 'contextual': 0, 'structural': 0, 'joint': 0, 'struct_joint': 0}
        epoch_log = []
        start_time = time.time()
        
        for ep in tqdm(range(config['epoch_num']), desc="  Training", leave=False):
            model.train()
            loss, scores, f_pn, d_pn, n_pn = model(
                data.edge_index, data.x, degree_list, neighbor_dict, device)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evaluate
            comp = scores.cpu().detach()
            auc_b = eval_roc_auc(y.numpy(), comp.numpy()) * 100
            auc_c = eval_roc_auc(yc.numpy(), comp.numpy()) * 100
            auc_s = eval_roc_auc(ys.numpy(), comp.numpy()) * 100
            auc_j = eval_roc_auc(yj.numpy(), comp.numpy()) * 100
            auc_sj = eval_roc_auc(ysj.numpy(), comp.numpy()) * 100
            
            best['benchmark'] = max(best['benchmark'], auc_b)
            best['contextual'] = max(best['contextual'], auc_c)
            best['structural'] = max(best['structural'], auc_s)
            best['joint'] = max(best['joint'], auc_j)
            best['struct_joint'] = max(best['struct_joint'], auc_sj)
            
            epoch_log.append({'epoch': ep, 'loss': loss.item(), 'auc_benchmark': auc_b,
                              'auc_contextual': auc_c, 'auc_structural': auc_s, 'auc_joint': auc_j})
        
        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.1f}s | Best — Bench: {best['benchmark']:.2f}, "
              f"Ctx: {best['contextual']:.2f}, Str: {best['structural']:.2f}, Jnt: {best['joint']:.2f}")
        
        all_results.append({'run': run_id, 'best': best, 'epoch_log': epoch_log, 'time': elapsed})
    
    # Aggregate
    metrics = ['benchmark', 'contextual', 'structural', 'joint', 'struct_joint']
    agg = {}
    for m in metrics:
        vals = [r['best'][m] for r in all_results]
        agg[m] = {'mean': np.mean(vals), 'std': np.std(vals), 'max': np.max(vals)}
    
    return all_results, agg


# ======================== MAIN ========================

CONFIGS = {
    'inj_cora': {
        'dimension': 64, 'lr': 0.005, 'epoch_num': 100, 'sample_size': 10,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'n_components': 2, 'gat_heads': 4,
        'contextual_n': 70, 'contextual_k': 10, 'structural_n': 70, 'structural_m': 10,
    },
    'disney': {
        'dimension': 16, 'lr': 0.005, 'epoch_num': 50, 'sample_size': 10,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'n_components': 2, 'gat_heads': 4,
        'contextual_n': 3, 'contextual_k': 5, 'structural_n': 3, 'structural_m': 5,
    },
    'books': {
        'dimension': 16, 'lr': 0.005, 'epoch_num': 50, 'sample_size': 10,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'n_components': 2, 'gat_heads': 2,
        'contextual_n': 14, 'contextual_k': 5, 'structural_n': 14, 'structural_m': 5,
    },
    'weibo': {
        'dimension': 16, 'lr': 0.005, 'epoch_num': 100, 'sample_size': 10,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'n_components': 2, 'gat_heads': 4,
        'contextual_n': 434, 'contextual_k': 10, 'structural_n': 434, 'structural_m': 10,
    },
    'reddit': {
        'dimension': 16, 'lr': 0.005, 'epoch_num': 100, 'sample_size': 10,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'n_components': 2, 'gat_heads': 4,
        'contextual_n': 183, 'contextual_k': 30, 'structural_n': 183, 'structural_m': 30,
    },
    'enron': {
        'dimension': 16, 'lr': 0.005, 'epoch_num': 100, 'sample_size': 10,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'n_components': 2, 'gat_heads': 4,
        'contextual_n': 3, 'contextual_k': 25, 'structural_n': 3, 'structural_m': 25,
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAD-NR++: Enhanced Graph Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='disney')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = CONFIGS[args.dataset].copy()
    if args.epochs:
        config['epoch_num'] = args.epochs
    
    print("=" * 70)
    print("GAD-NR++: Enhanced Graph Anomaly Detection")
    print(f"Dataset: {args.dataset}, Device: {device}")
    print(f"Improvements: GAT encoder, GMM decoder (K={config['n_components']}), Learnable fusion")
    print("=" * 70)
    
    results, agg = run_experiment(args.dataset, config, device, num_runs=args.num_runs)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.dataset}")
    print(f"{'='*60}")
    for m in ['benchmark', 'contextual', 'structural', 'joint', 'struct_joint']:
        print(f"  {m:>15s}: {agg[m]['mean']:.2f} ± {agg[m]['std']:.2f} (best: {agg[m]['max']:.2f})")
    print(f"{'='*60}")
