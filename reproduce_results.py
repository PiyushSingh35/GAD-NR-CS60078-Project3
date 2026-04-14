"""
GAD-NR Reproduction Experiments — Task T3 & T4
================================================
Reproduces the paper results from Tables 2, 3, and 4 of:
  "GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction" (WSDM 2024)

This script runs GAD-NR on all 6 datasets with the paper's fixed hyperparameters,
collects AUC scores across 3 runs, and produces:
  - Benchmark anomaly detection results (Table 2 reproduction)
  - Contextual anomaly detection results (Table 3 left)
  - Structural + Joint-type anomaly detection results (Table 3 right)
  - Ablation study (removing each loss component)
  - Per-epoch loss curves (for qualitative analysis)
  - Saves all results to CSV and generates plots

Usage:
    python reproduce_results.py --dataset inj_cora
    python reproduce_results.py --dataset all
    python reproduce_results.py --dataset inj_cora --ablation
"""

import sys
import types

# Mock C++ extension modules
def _noop(*a, **k):
    return None

for _mod_name in ['torch_sparse', 'torch_scatter', 'torch_cluster', 'torch_spline_conv']:
    _mock = types.ModuleType(_mod_name)
    _mock.SparseTensor = type('SparseTensor', (), {})
    _mock.random_walk = _noop
    _mock.knn = _noop
    _mock.radius = _noop
    _mock.nearest = _noop
    _mock.knn_graph = _noop
    _mock.radius_graph = _noop
    _mock.matmul = _noop
    _mock.fill_diag = _noop
    _mock.sum = _noop
    _mock.mul = _noop
    _mock.set_diag = _noop
    _mock.remove_diag = _noop
    sys.modules[_mod_name] = _mock

import os
import json
import csv
import zipfile
import argparse
import random
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg import sqrtm
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv

from pygod.metrics import eval_roc_auc
from pygod.generator import gen_contextual_outliers, gen_structural_outliers
from pygod.utils.utility import check_parameter


# ======================== CONFIG ========================
DATA_DIR = "/home/claude/pygod_data/extracted"
OUTPUT_DIR = "/home/claude/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paper's fixed hyperparameters (Section 5.2)
PAPER_CONFIGS = {
    'inj_cora': {
        'dimension': 128, 'lr': 0.01, 'epoch_num': 100,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'sample_size': 10, 'loss_step': 30, 'encoder': 'GCN',
        'contextual_n': 70, 'contextual_k': 10,
        'structural_n': 70, 'structural_m': 10,
    },
    'weibo': {
        'dimension': 16, 'lr': 0.01, 'epoch_num': 100,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'sample_size': 10, 'loss_step': 30, 'encoder': 'GCN',
        'contextual_n': 434, 'contextual_k': 10,
        'structural_n': 434, 'structural_m': 10,
    },
    'reddit': {
        'dimension': 16, 'lr': 0.01, 'epoch_num': 100,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'sample_size': 10, 'loss_step': 30, 'encoder': 'GCN',
        'contextual_n': 183, 'contextual_k': 30,
        'structural_n': 183, 'structural_m': 30,
    },
    'disney': {
        'dimension': 16, 'lr': 0.01, 'epoch_num': 100,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'sample_size': 10, 'loss_step': 30, 'encoder': 'GCN',
        'contextual_n': 3, 'contextual_k': 5,
        'structural_n': 3, 'structural_m': 5,
    },
    'books': {
        'dimension': 16, 'lr': 0.01, 'epoch_num': 100,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'sample_size': 10, 'loss_step': 30, 'encoder': 'GCN',
        'contextual_n': 14, 'contextual_k': 5,
        'structural_n': 14, 'structural_m': 5,
    },
    'enron': {
        'dimension': 16, 'lr': 0.01, 'epoch_num': 100,
        'lambda_loss1': 1e-2, 'lambda_loss2': 0.5, 'lambda_loss3': 0.8,
        'sample_size': 10, 'loss_step': 30, 'encoder': 'GCN',
        'contextual_n': 3, 'contextual_k': 25,
        'structural_n': 3, 'structural_m': 25,
    },
}

# Paper's reported results (Table 2) for comparison
PAPER_TABLE2 = {
    'inj_cora': {'avg': 87.55, 'std': 2.56, 'best': 88.40},
    'weibo':    {'avg': 87.71, 'std': 5.39, 'best': 92.09},
    'reddit':   {'avg': 57.99, 'std': 1.67, 'best': 59.90},
    'disney':   {'avg': 76.76, 'std': 2.75, 'best': 80.03},
    'books':    {'avg': 65.71, 'std': 4.98, 'best': 69.79},
    'enron':    {'avg': 80.87, 'std': 2.95, 'best': 82.92},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== DATA LOADING ========================
def load_data_local(dataset_str):
    pt_path = os.path.join(DATA_DIR, f"{dataset_str}.pt")
    if not os.path.exists(pt_path):
        zip_path = os.path.join(os.path.dirname(DATA_DIR), f"{dataset_str}.pt.zip")
        if os.path.exists(zip_path):
            os.makedirs(DATA_DIR, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)
        else:
            raise FileNotFoundError(f"Dataset not found: {pt_path}")
    return torch.load(pt_path, weights_only=False)


# ======================== UTILS ========================
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


def KL_neighbor_loss(predictions, targets, mask_len):
    x1 = predictions.squeeze().cpu().detach()
    x2 = targets.squeeze().cpu().detach()
    mean_x1, mean_x2 = x1.mean(0), x2.mean(0)
    nn_val, h_dim = x1.shape
    cov_x1 = (x1 - mean_x1).T.matmul(x1 - mean_x1) / max((nn_val - 1), 1)
    cov_x2 = (x2 - mean_x2).T.matmul(x2 - mean_x2) / max((nn_val - 1), 1)
    eye = torch.eye(h_dim)
    cov_x1, cov_x2 = cov_x1 + eye, cov_x2 + eye
    det_ratio = torch.det(cov_x1) / torch.det(cov_x2)
    if det_ratio <= 0:
        det_ratio = torch.abs(det_ratio) + 1e-10
    KL = 0.5 * (math.log(det_ratio) - h_dim
                 + torch.trace(torch.inverse(cov_x2).matmul(cov_x1))
                 + (mean_x2 - mean_x1).reshape(1, -1).matmul(torch.inverse(cov_x2)).matmul(mean_x2 - mean_x1))
    return KL.to(device)


# ======================== MODEL LAYERS ========================
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_layers = num_layers
        self.linear_or_not = (num_layers == 1)
        if self.linear_or_not:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        h = x
        for i in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[i](self.linears[i](h)))
        return self.linears[self.num_layers - 1](h)


class MLP_generator(nn.Module):
    def __init__(self, input_dim, output_dim, sample_size):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.linear3 = nn.Linear(output_dim, output_dim)
        self.linear4 = nn.Linear(output_dim, output_dim)

    def forward(self, embedding, device):
        x = F.relu(self.linear(embedding))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)


class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super().__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, embedding):
        return self.linear2(F.relu(self.linear1(embedding)))


# ======================== GAD-NR MODEL ========================
class GNNStructEncoder(nn.Module):
    def __init__(self, in_dim0, in_dim, hidden_dim, layer_num, sample_size, device,
                 neighbor_num_list, GNN_name="GCN",
                 lambda_loss1=0.01, lambda_loss2=0.001, lambda_loss3=0.0001):
        super().__init__()
        self.mlp0 = nn.Linear(in_dim0, hidden_dim)
        self.out_dim = hidden_dim
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3

        if GNN_name == "GIN":
            lin1 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv1 = GINConv(lin1)
        elif GNN_name == "GCN":
            self.graphconv1 = GCNConv(hidden_dim, hidden_dim)
        elif GNN_name == "GAT":
            self.graphconv1 = GATConv(hidden_dim, hidden_dim)
        else:
            self.graphconv1 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')

        self.m = torch.distributions.Normal(
            torch.zeros(sample_size, hidden_dim), torch.ones(sample_size, hidden_dim))
        self.mlp_mean = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_sigma = nn.Linear(hidden_dim, hidden_dim)
        self.layer1_generator = MLP_generator(hidden_dim, hidden_dim, sample_size)

        self.degree_decoder = FNN(hidden_dim, hidden_dim, 1, 4)
        self.feature_decoder = FNN(hidden_dim, hidden_dim, in_dim, 3)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()
        self.in_dim = in_dim
        self.sample_size = sample_size

    def forward_encoder(self, x, edge_index):
        h0 = self.mlp0(x)
        l1 = self.graphconv1(h0, edge_index)
        return l1, h0

    def sample_neighbors(self, indexes, neighbor_dict, gt_embeddings):
        sampled_list, mark_list = [], []
        for idx in indexes:
            neighs = neighbor_dict[idx]
            if len(neighs) < self.sample_size:
                sample_idx = neighs
                mask_len = len(neighs)
            else:
                sample_idx = random.sample(neighs, self.sample_size)
                mask_len = self.sample_size
            embs = [gt_embeddings[i].tolist() for i in sample_idx]
            while len(embs) < self.sample_size:
                embs.append(torch.zeros(self.out_dim).tolist())
            sampled_list.append(embs)
            mark_list.append(mask_len)
        return sampled_list, mark_list

    def reconstruction_neighbors(self, FNN_gen, indexes, neighbor_dict, from_layer, to_layer, device):
        total_loss = 0
        per_node = []
        sampled_list, mark_list = self.sample_neighbors(indexes, neighbor_dict, to_layer)
        for i, nembs in enumerate(sampled_list):
            idx = indexes[i]
            mean = self.mlp_mean(from_layer[idx].repeat(self.sample_size, 1))
            sigma = self.mlp_sigma(from_layer[idx].repeat(self.sample_size, 1))
            std_z = self.m.sample().to(device)
            var = mean + sigma.exp() * std_z
            gen = FNN_gen(var, device)
            gen = gen.unsqueeze(0).to(device)
            tgt = torch.FloatTensor(nembs).unsqueeze(0).to(device)
            kl = KL_neighbor_loss(gen, tgt, mark_list[i])
            total_loss += kl
            per_node.append(kl)
        return total_loss, torch.stack(per_node)

    def forward(self, edge_index, x, degree_list, neighbor_dict, device,
                disable_feat=False, disable_deg=False, disable_neigh=False):
        l1, h0 = self.forward_encoder(x, edge_index)
        tot = l1.shape[0]

        # Degree reconstruction
        deg_logits = F.relu(self.degree_decoder(l1))
        deg_gt = degree_list.unsqueeze(1).float()
        deg_loss = self.degree_loss_func(deg_logits, deg_gt)
        deg_loss_per_node = (deg_logits - deg_gt).pow(2)

        loss_runs, loss_runs_per_node, feat_runs = [], [], []
        for _ in range(3):
            h0_prime = self.feature_decoder(l1)
            feat_pn = (h0 - h0_prime).pow(2).mean(1)
            feat_runs.append(feat_pn)
            indexes = list(range(tot))
            nl, nlpn = self.reconstruction_neighbors(
                self.layer1_generator, indexes, neighbor_dict, l1, h0, device)
            loss_runs.append(nl)
            loss_runs_per_node.append(nlpn)

        h_loss = torch.mean(torch.stack(loss_runs))
        h_loss_pn = torch.mean(torch.stack(loss_runs_per_node), dim=0).reshape(tot, 1)
        feat_loss = torch.mean(torch.stack(feat_runs))
        feat_loss_pn = torch.mean(torch.stack(feat_runs), dim=0).reshape(tot, 1)

        deg_loss_pn = deg_loss_per_node.reshape(tot, 1)

        # Apply ablation masks
        lam1 = 0.0 if disable_neigh else self.lambda_loss1
        lam2 = 0.0 if disable_feat else self.lambda_loss2
        lam3 = 0.0 if disable_deg else self.lambda_loss3

        loss = lam1 * h_loss + lam3 * deg_loss + lam2 * feat_loss
        loss_pn = lam1 * h_loss_pn + lam3 * deg_loss_pn + lam2 * feat_loss_pn

        return loss, loss_pn, h_loss_pn, deg_loss_pn, feat_loss_pn


# ======================== TRAINING ========================
def run_single_experiment(dataset_str, config, num_runs=3,
                          disable_feat=False, disable_deg=False, disable_neigh=False):
    """Run GAD-NR on a dataset for multiple runs and return aggregated results."""
    all_results = []

    for run_id in range(num_runs):
        print(f"\n--- {dataset_str} | Run {run_id+1}/{num_runs} ---")
        seed = 42 + run_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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

        yc = yc.cpu().detach()
        ys = ys.cpu().detach()
        data, yj = gen_joint_structural_outliers(data=data, n=config['structural_n'], m=config['structural_m'])
        ysj = torch.logical_or(ys, yj).int()
        y = data.y.bool().cpu().detach()

        # Add self-loops
        edge_index = data.edge_index.cpu()
        num_nodes = data.x.shape[0]
        self_edges = torch.tensor([[i for i in range(num_nodes)], [i for i in range(num_nodes)]])
        data.edge_index = torch.cat([edge_index, self_edges], dim=1)
        data = data.to(device)

        # Build neighbor dict
        in_n, out_n = data.edge_index[0], data.edge_index[1]
        neighbor_dict = {}
        for a, b in zip(in_n.tolist(), out_n.tolist()):
            neighbor_dict.setdefault(a, []).append(b)
        neighbor_num = torch.tensor([len(neighbor_dict.get(i, [])) for i in range(num_nodes)]).to(device)

        # Build model
        in_dim = data.x.shape[1]
        model = GNNStructEncoder(
            in_dim, config['dimension'], config['dimension'], 2,
            config['sample_size'], device=device, neighbor_num_list=neighbor_num,
            GNN_name=config['encoder'],
            lambda_loss1=config['lambda_loss1'],
            lambda_loss2=config['lambda_loss2'],
            lambda_loss3=config['lambda_loss3']).to(device)

        deg_params = list(map(id, model.degree_decoder.parameters()))
        base_params = filter(lambda p: id(p) not in deg_params, model.parameters())
        opt = torch.optim.Adam(
            [{'params': base_params},
             {'params': model.degree_decoder.parameters(), 'lr': 1e-2}],
            lr=config['lr'], weight_decay=0.0003)

        best = {'benchmark': 0, 'contextual': 0, 'structural': 0, 'joint': 0, 'struct_joint': 0}
        epoch_log = []
        start_time = time.time()

        for ep in tqdm(range(config['epoch_num']), desc=f"  Training", leave=False):
            if ep % config['loss_step'] == 0:
                model.lambda_loss2 += 0.5
                model.lambda_loss3 /= 2

            loss, loss_pn, h_pn, d_pn, f_pn = model(
                data.edge_index, data.x, neighbor_num, neighbor_dict, device,
                disable_feat=disable_feat, disable_deg=disable_deg, disable_neigh=disable_neigh)

            loss_pn = loss_pn.cpu().detach()
            h_pn = h_pn.cpu().detach()
            d_pn = d_pn.cpu().detach()
            f_pn = f_pn.cpu().detach()

            # Adaptive loss weighting
            h_norm = h_pn / (h_pn.max() - h_pn.min() + 1e-8)
            d_norm = d_pn / (d_pn.max() - d_pn.min() + 1e-8)
            f_norm = f_pn / (f_pn.max() - f_pn.min() + 1e-8)
            comp = 1.0 * h_norm + 1.0 * d_norm + 2.0 * f_norm

            auc_bench = eval_roc_auc(y.numpy(), comp.numpy()) * 100
            auc_ctx = eval_roc_auc(yc.numpy(), comp.numpy()) * 100
            auc_str = eval_roc_auc(ys.numpy(), comp.numpy()) * 100
            auc_jnt = eval_roc_auc(yj.numpy(), comp.numpy()) * 100
            auc_sj = eval_roc_auc(ysj.numpy(), comp.numpy()) * 100

            best['benchmark'] = max(best['benchmark'], auc_bench)
            best['contextual'] = max(best['contextual'], auc_ctx)
            best['structural'] = max(best['structural'], auc_str)
            best['joint'] = max(best['joint'], auc_jnt)
            best['struct_joint'] = max(best['struct_joint'], auc_sj)

            epoch_log.append({
                'epoch': ep, 'loss': loss.item(),
                'auc_benchmark': auc_bench, 'auc_contextual': auc_ctx,
                'auc_structural': auc_str, 'auc_joint': auc_jnt
            })

            opt.zero_grad()
            loss.backward()
            opt.step()

        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.1f}s | Best AUC — Bench: {best['benchmark']:.2f}, "
              f"Ctx: {best['contextual']:.2f}, Str: {best['structural']:.2f}, "
              f"Jnt: {best['joint']:.2f}")

        all_results.append({
            'run': run_id, 'best': best, 'epoch_log': epoch_log, 'time': elapsed
        })

    return all_results


def aggregate_results(all_results):
    """Compute mean ± std across runs."""
    metrics = ['benchmark', 'contextual', 'structural', 'joint', 'struct_joint']
    agg = {}
    for m in metrics:
        vals = [r['best'][m] for r in all_results]
        agg[m] = {'mean': np.mean(vals), 'std': np.std(vals), 'max': np.max(vals), 'values': vals}
    return agg


# ======================== PLOTTING ========================
def plot_training_curves(all_results, dataset_str):
    """Plot loss and AUC curves across epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in all_results:
        epochs = [e['epoch'] for e in r['epoch_log']]
        losses = [e['loss'] for e in r['epoch_log']]
        aucs = [e['auc_benchmark'] for e in r['epoch_log']]
        axes[0].plot(epochs, losses, alpha=0.6, label=f"Run {r['run']+1}")
        axes[1].plot(epochs, aucs, alpha=0.6, label=f"Run {r['run']+1}")

    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{dataset_str} — Training Loss'); axes[0].legend()
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC (%)')
    axes[1].set_title(f'{dataset_str} — Benchmark AUC'); axes[1].legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f'{dataset_str}_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_anomaly_type_comparison(all_agg, dataset_names):
    """Bar chart comparing AUC across anomaly types for all datasets."""
    types = ['benchmark', 'contextual', 'structural', 'joint']
    labels = ['Benchmark', 'Contextual', 'Structural', 'Joint-type']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(dataset_names))
    width = 0.2

    for i, (t, label) in enumerate(zip(types, labels)):
        means = [all_agg[d][t]['mean'] for d in dataset_names]
        stds = [all_agg[d][t]['std'] for d in dataset_names]
        ax.bar(x + i * width, means, width, yerr=stds, label=label, capsize=3)

    ax.set_xlabel('Dataset'); ax.set_ylabel('AUC (%)')
    ax.set_title('GAD-NR: AUC across Anomaly Types and Datasets')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(dataset_names)
    ax.legend(); ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    path = os.path.join(OUTPUT_DIR, 'anomaly_type_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_paper_comparison(all_agg, dataset_names):
    """Compare our reproduced results vs paper's reported results (Table 2)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dataset_names))
    width = 0.35

    ours_mean = [all_agg[d]['benchmark']['mean'] for d in dataset_names]
    ours_std = [all_agg[d]['benchmark']['std'] for d in dataset_names]
    paper_mean = [PAPER_TABLE2[d]['avg'] for d in dataset_names]
    paper_std = [PAPER_TABLE2[d]['std'] for d in dataset_names]

    ax.bar(x - width/2, paper_mean, width, yerr=paper_std, label='Paper (Table 2)', capsize=4, color='steelblue')
    ax.bar(x + width/2, ours_mean, width, yerr=ours_std, label='Our Reproduction', capsize=4, color='coral')

    ax.set_xlabel('Dataset'); ax.set_ylabel('AUC (%)')
    ax.set_title('Benchmark Anomaly Detection: Paper vs Our Reproduction')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)
    ax.legend(); ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (p, o) in enumerate(zip(paper_mean, ours_mean)):
        ax.text(i - width/2, p + 2, f'{p:.1f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, o + 2, f'{o:.1f}', ha='center', va='bottom', fontsize=8)

    path = os.path.join(OUTPUT_DIR, 'paper_vs_reproduction.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ======================== RESULTS TABLE ========================
def print_results_table(all_agg, dataset_names):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("TABLE 2 REPRODUCTION: Benchmark Anomaly Detection (ROC-AUC %)")
    print("=" * 90)
    print(f"{'Dataset':<12} {'Our Mean±Std':>18} {'Our Best':>10} {'Paper Mean±Std':>18} {'Paper Best':>12}")
    print("-" * 90)
    for d in dataset_names:
        ours = all_agg[d]['benchmark']
        paper = PAPER_TABLE2[d]
        print(f"{d:<12} {ours['mean']:>8.2f} ± {ours['std']:<6.2f}  {ours['max']:>8.2f}   "
              f"{paper['avg']:>8.2f} ± {paper['std']:<6.2f}  {paper['best']:>8.2f}")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("TABLE 3 REPRODUCTION: Contextual vs Structural+Joint AUC (%)")
    print("=" * 90)
    print(f"{'Dataset':<12} {'Contextual':>14} {'Structural':>14} {'Joint-type':>14} {'Struct+Joint':>14}")
    print("-" * 90)
    for d in dataset_names:
        a = all_agg[d]
        print(f"{d:<12} {a['contextual']['mean']:>8.2f}±{a['contextual']['std']:<4.2f}"
              f" {a['structural']['mean']:>8.2f}±{a['structural']['std']:<4.2f}"
              f" {a['joint']['mean']:>8.2f}±{a['joint']['std']:<4.2f}"
              f" {a['struct_joint']['mean']:>8.2f}±{a['struct_joint']['std']:<4.2f}")
    print("=" * 90)


def save_results_csv(all_agg, dataset_names):
    """Save results to CSV."""
    path = os.path.join(OUTPUT_DIR, 'results_summary.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Metric', 'Mean', 'Std', 'Max', 'Paper_Mean', 'Paper_Std', 'Paper_Best'])
        for d in dataset_names:
            for m in ['benchmark', 'contextual', 'structural', 'joint', 'struct_joint']:
                paper = PAPER_TABLE2.get(d, {})
                writer.writerow([
                    d, m,
                    f"{all_agg[d][m]['mean']:.2f}", f"{all_agg[d][m]['std']:.2f}", f"{all_agg[d][m]['max']:.2f}",
                    paper.get('avg', ''), paper.get('std', ''), paper.get('best', '')
                ])
    print(f"\nResults CSV saved: {path}")


# ======================== MAIN ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAD-NR Paper Reproduction')
    parser.add_argument('--dataset', type=str, default='inj_cora',
                        help='Dataset name or "all" for all 6 datasets')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs per dataset')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--epochs', type=int, default=None, help='Override epoch count')
    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = ['inj_cora', 'disney', 'books', 'enron', 'reddit', 'weibo']
    else:
        datasets = [args.dataset]

    all_agg = {}

    print("=" * 70)
    print("GAD-NR Paper Reproduction — Task T3")
    print(f"Device: {device}")
    print(f"Datasets: {datasets}")
    print(f"Runs per dataset: {args.num_runs}")
    print("=" * 70)

    for ds in datasets:
        config = PAPER_CONFIGS[ds].copy()
        if args.epochs:
            config['epoch_num'] = args.epochs

        print(f"\n{'='*60}")
        print(f"DATASET: {ds} (dim={config['dimension']}, epochs={config['epoch_num']})")
        print(f"{'='*60}")

        results = run_single_experiment(ds, config, num_runs=args.num_runs)
        agg = aggregate_results(results)
        all_agg[ds] = agg

        # Plot training curves
        plot_training_curves(results, ds)

        # Print per-dataset summary
        print(f"\n  {ds} Summary:")
        print(f"    Benchmark:  {agg['benchmark']['mean']:.2f} ± {agg['benchmark']['std']:.2f} (best: {agg['benchmark']['max']:.2f})")
        print(f"    Contextual: {agg['contextual']['mean']:.2f} ± {agg['contextual']['std']:.2f}")
        print(f"    Structural: {agg['structural']['mean']:.2f} ± {agg['structural']['std']:.2f}")
        print(f"    Joint-type: {agg['joint']['mean']:.2f} ± {agg['joint']['std']:.2f}")

    # Ablation study
    if args.ablation and len(datasets) >= 1:
        ds = datasets[0]
        config = PAPER_CONFIGS[ds].copy()
        if args.epochs:
            config['epoch_num'] = args.epochs

        print(f"\n{'='*60}")
        print(f"ABLATION STUDY on {ds}")
        print(f"{'='*60}")

        ablation_results = {}
        for name, df, dd, dn in [
            ('Full GAD-NR', False, False, False),
            ('w/o feat recon', True, False, False),
            ('w/o degree recon', False, True, False),
            ('w/o neighbor recon', False, False, True),
        ]:
            print(f"\n  >> {name}")
            res = run_single_experiment(ds, config, num_runs=args.num_runs,
                                        disable_feat=df, disable_deg=dd, disable_neigh=dn)
            ablation_results[name] = aggregate_results(res)

        print(f"\n{'='*60}")
        print(f"ABLATION RESULTS — {ds}")
        print(f"{'='*60}")
        print(f"{'Variant':<25} {'Benchmark':>12} {'Contextual':>12} {'Struct+Joint':>14}")
        print("-" * 65)
        for name, agg in ablation_results.items():
            print(f"{name:<25} {agg['benchmark']['mean']:>8.2f}±{agg['benchmark']['std']:<4.2f}"
                  f" {agg['contextual']['mean']:>8.2f}±{agg['contextual']['std']:<4.2f}"
                  f" {agg['struct_joint']['mean']:>8.2f}±{agg['struct_joint']['std']:<4.2f}")

    # Final tables and plots
    if len(all_agg) > 0:
        run_datasets = list(all_agg.keys())
        print_results_table(all_agg, run_datasets)
        save_results_csv(all_agg, run_datasets)

        if len(run_datasets) > 1:
            plot_anomaly_type_comparison(all_agg, run_datasets)
            plot_paper_comparison(all_agg, run_datasets)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("Done!")
