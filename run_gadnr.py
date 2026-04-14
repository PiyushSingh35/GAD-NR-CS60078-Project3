"""
GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction
================================================================
Standalone training script converted from the original Jupyter notebooks.
Setup: Python 3.12, PyTorch 2.11, PyTorch Geometric 2.5.3, pygod 0.3.1

Usage:
    python run_gadnr.py --dataset inj_cora --epoch_num 100
    python run_gadnr.py --dataset weibo --epoch_num 100 --dimension 16
    python run_gadnr.py --dataset reddit --epoch_num 100 --dimension 16
    python run_gadnr.py --dataset disney --epoch_num 100 --dimension 16
    python run_gadnr.py --dataset books --epoch_num 100 --dimension 16
    python run_gadnr.py --dataset enron --epoch_num 100 --dimension 16
"""

import sys
import types

# ====== Mock C++ extension modules (torch_sparse, torch_scatter, etc.) ======
# These are needed by pygod's model imports but NOT used by our actual GAD-NR code path.
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
    # For torch_sparse specific attrs
    _mock.matmul = _noop
    _mock.fill_diag = _noop
    _mock.sum = _noop
    _mock.mul = _noop
    _mock.set_diag = _noop
    _mock.remove_diag = _noop
    sys.modules[_mod_name] = _mock

import os
import zipfile
import argparse
import random
import math
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb
import networkx as nx

from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv
from torch_geometric.utils import add_self_loops

from pygod.metrics import eval_roc_auc
from pygod.generator import gen_contextual_outliers, gen_structural_outliers
from pygod.utils.utility import check_parameter


# ======================== DATA LOADING ========================
DATA_DIR = "/home/claude/pygod_data/extracted"

def load_data_local(dataset_str):
    """Load dataset from locally extracted .pt files (no internet needed)."""
    pt_path = os.path.join(DATA_DIR, f"{dataset_str}.pt")
    if not os.path.exists(pt_path):
        # Try extracting from zip
        zip_path = os.path.join(os.path.dirname(DATA_DIR), f"{dataset_str}.pt.zip")
        if os.path.exists(zip_path):
            os.makedirs(DATA_DIR, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(DATA_DIR)
        else:
            raise FileNotFoundError(f"Dataset {dataset_str} not found at {pt_path} or {zip_path}")
    data = torch.load(pt_path, weights_only=False)
    return data


# ======================== UTILITY FUNCTIONS ========================

def gen_joint_structural_outliers(data, m, n, random_state=None):
    """
    Generate joint-type structural outliers.
    Randomly select n nodes as anomalies and connect each to m random other nodes.
    """
    if not isinstance(data, Data):
        raise TypeError("data should be torch_geometric.data.Data")
    check_parameter(m, low=0, high=data.num_nodes, param_name='m')
    check_parameter(n, low=0, high=data.num_nodes, param_name='n')
    check_parameter(m * n, low=0, high=data.num_nodes, param_name='m*n')

    if random_state:
        np.random.seed(random_state)

    outlier_idx = np.random.choice(data.num_nodes, size=n, replace=False)
    new_edges = []
    for i in range(0, n):
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

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn_val = x1.shape[0]
    h_dim = x1.shape[1]

    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / max((nn_val - 1), 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / max((nn_val - 1), 1)

    eye = torch.eye(h_dim)
    cov_x1 = cov_x1 + eye
    cov_x2 = cov_x2 + eye

    KL_loss = 0.5 * (
        math.log(torch.det(cov_x1) / torch.det(cov_x2)) - h_dim
        + torch.trace(torch.inverse(cov_x2).matmul(cov_x1))
        + (mean_x2 - mean_x1).reshape(1, -1).matmul(torch.inverse(cov_x2)).matmul(mean_x2 - mean_x1)
    )
    KL_loss = KL_loss.to(device)
    return KL_loss


def W2_neighbor_loss(predictions, targets, mask_len):
    x1 = predictions.squeeze().cpu().detach()
    x2 = targets.squeeze().cpu().detach()

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)
    nn_val = x1.shape[0]

    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / (nn_val - 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / (nn_val - 1)

    W2_loss = torch.square(mean_x1 - mean_x2).sum() + torch.trace(
        cov_x1 + cov_x2 + 2 * sqrtm(sqrtm(cov_x1) @ (cov_x2.numpy()) @ (sqrtm(cov_x1)))
    )
    return W2_loss


# ======================== MODEL LAYERS ========================

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class MLP_generator(nn.Module):
    def __init__(self, input_dim, output_dim, sample_size):
        super(MLP_generator, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.linear3 = nn.Linear(output_dim, output_dim)
        self.linear4 = nn.Linear(output_dim, output_dim)

    def forward(self, embedding, device):
        neighbor_embedding = F.relu(self.linear(embedding))
        neighbor_embedding = F.relu(self.linear2(neighbor_embedding))
        neighbor_embedding = F.relu(self.linear3(neighbor_embedding))
        neighbor_embedding = self.linear4(neighbor_embedding)
        return neighbor_embedding


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=10):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(F.relu(x))
        return x


# ======================== GAD-NR MODEL ========================

class GNNStructEncoder(nn.Module):
    def __init__(self, in_dim0, in_dim, hidden_dim, layer_num, sample_size, device,
                 neighbor_num_list, GNN_name="GIN", norm_mode="PN-SCS", norm_scale=20,
                 lambda_loss1=0.01, lambda_loss2=0.001, lambda_loss3=0.0001):
        super(GNNStructEncoder, self).__init__()

        self.mlp0 = nn.Linear(in_dim0, hidden_dim)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.out_dim = hidden_dim
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3

        # GNN Encoder
        if GNN_name == "GIN":
            self.linear1 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv1 = GINConv(self.linear1)
            self.linear2 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv2 = GINConv(self.linear2)
        elif GNN_name == "GCN":
            self.graphconv1 = GCNConv(hidden_dim, hidden_dim)
            self.graphconv2 = GCNConv(hidden_dim, hidden_dim)
        elif GNN_name == "GAT":
            self.graphconv1 = GATConv(hidden_dim, hidden_dim)
            self.graphconv2 = GATConv(hidden_dim, hidden_dim)
        else:
            self.graphconv1 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
            self.graphconv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')

        self.neighbor_num_list = neighbor_num_list
        self.neighbor_generator = MLP_generator(hidden_dim, hidden_dim, sample_size).to(device)

        self.gaussian_mean = nn.Parameter(
            torch.FloatTensor(sample_size, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(sample_size, hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.m = torch.distributions.Normal(
            torch.zeros(sample_size, hidden_dim), torch.ones(sample_size, hidden_dim))
        self.m_h = torch.distributions.Normal(
            torch.zeros(sample_size, hidden_dim), 50 * torch.ones(sample_size, hidden_dim))

        self.mlp_gaussian_mean = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.mlp_gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.mlp_m = torch.distributions.Normal(torch.zeros(hidden_dim), torch.ones(hidden_dim))

        self.mlp_mean = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_sigma = nn.Linear(hidden_dim, hidden_dim)

        self.layer1_generator = MLP_generator(hidden_dim, hidden_dim, sample_size)

        # Decoders
        self.degree_decoder = FNN(hidden_dim, hidden_dim, 1, 4)
        self.feature_decoder = FNN(hidden_dim, hidden_dim, in_dim, 3)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()
        self.in_dim = in_dim
        self.sample_size = sample_size
        self.init_projection = FNN(in_dim, hidden_dim, hidden_dim, 1)

    def forward_encoder(self, x, edge_index):
        h0 = self.mlp0(x)
        l1 = self.graphconv1(h0, edge_index)
        return l1, h0

    def sample_neighbors(self, indexes, neighbor_dict, gt_embeddings):
        sampled_embeddings_list = []
        mark_len_list = []
        for index in indexes:
            sampled_embeddings = []
            neighbor_indexes = neighbor_dict[index]
            if len(neighbor_indexes) < self.sample_size:
                mask_len = len(neighbor_indexes)
                sample_indexes = neighbor_indexes
            else:
                sample_indexes = random.sample(neighbor_indexes, self.sample_size)
                mask_len = self.sample_size
            for idx in sample_indexes:
                sampled_embeddings.append(gt_embeddings[idx].tolist())
            if len(sampled_embeddings) < self.sample_size:
                for _ in range(self.sample_size - len(sampled_embeddings)):
                    sampled_embeddings.append(torch.zeros(self.out_dim).tolist())
            sampled_embeddings_list.append(sampled_embeddings)
            mark_len_list.append(mask_len)
        return sampled_embeddings_list, mark_len_list

    def reconstruction_neighbors(self, FNN_generator, neighbor_indexes, neighbor_dict,
                                  from_layer, to_layer, device):
        local_index_loss = 0
        local_index_loss_per_node = []
        sampled_embeddings_list, mark_len_list = self.sample_neighbors(
            neighbor_indexes, neighbor_dict, to_layer)

        for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
            index = neighbor_indexes[i]
            mask_len1 = mark_len_list[i]
            mean = from_layer[index].repeat(self.sample_size, 1)
            mean = self.mlp_mean(mean)
            sigma = from_layer[index].repeat(self.sample_size, 1)
            sigma = self.mlp_sigma(sigma)
            std_z = self.m.sample().to(device)
            var = mean + sigma.exp() * std_z
            nhij = FNN_generator(var, device)

            generated_neighbors = nhij
            generated_neighbors = torch.unsqueeze(generated_neighbors, dim=0).to(device)
            target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0).to(device)

            if args.neigh_loss == "KL":
                loss_val = KL_neighbor_loss(generated_neighbors, target_neighbors, mask_len1)
            else:
                loss_val = W2_neighbor_loss(generated_neighbors, target_neighbors, mask_len1)

            local_index_loss += loss_val
            local_index_loss_per_node.append(loss_val)

        local_index_loss_per_node = torch.stack(local_index_loss_per_node)
        return local_index_loss, local_index_loss_per_node

    def neighbor_decoder(self, gij, ground_truth_degree_matrix, h0, neighbor_dict, device, h):
        tot_nodes = gij.shape[0]
        degree_logits = self.degree_decoding(gij)
        ground_truth_degree_matrix = torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix.float())
        degree_loss_per_node = (degree_logits - ground_truth_degree_matrix).pow(2)

        loss_list = []
        loss_list_per_node = []
        feature_loss_list = []

        for _ in range(3):
            indexes = []
            h0_prime = self.feature_decoder(gij)
            feature_losses_per_node = (h0 - h0_prime).pow(2).mean(1)
            feature_loss_list.append(feature_losses_per_node)

            for i1 in range(len(gij)):
                indexes.append(i1)
            local_index_loss, local_index_loss_per_node = self.reconstruction_neighbors(
                self.layer1_generator, indexes, neighbor_dict, gij, h0, device)
            loss_list.append(local_index_loss)
            loss_list_per_node.append(local_index_loss_per_node)

        loss_list = torch.stack(loss_list)
        h_loss = torch.mean(loss_list)

        loss_list_per_node = torch.stack(loss_list_per_node)
        h_loss_per_node = torch.mean(loss_list_per_node, dim=0)

        feature_loss_per_node = torch.mean(torch.stack(feature_loss_list), dim=0)
        feature_loss = torch.mean(torch.stack(feature_loss_list))

        h_loss_per_node = h_loss_per_node.reshape(tot_nodes, 1)
        degree_loss_per_node = degree_loss_per_node.reshape(tot_nodes, 1)
        feature_loss_per_node = feature_loss_per_node.reshape(tot_nodes, 1)

        loss = (self.lambda_loss1 * h_loss
                + degree_loss * self.lambda_loss3
                + self.lambda_loss2 * feature_loss)
        loss_per_node = (self.lambda_loss1 * h_loss_per_node
                         + degree_loss_per_node * self.lambda_loss3
                         + self.lambda_loss2 * feature_loss_per_node)

        return loss, loss_per_node, h_loss_per_node, degree_loss_per_node, feature_loss_per_node

    def degree_decoding(self, node_embeddings):
        degree_logits = F.relu(self.degree_decoder(node_embeddings))
        return degree_logits

    def forward(self, edge_index, x, ground_truth_degree_matrix, neighbor_dict, device):
        l1, h0 = self.forward_encoder(x, edge_index)
        loss, loss_per_node, h_loss, degree_loss, feature_loss = self.neighbor_decoder(
            l1, ground_truth_degree_matrix, h0, neighbor_dict, device, x)
        return loss, loss_per_node, h_loss, degree_loss, feature_loss


# ======================== TRAINING ========================

def train(data, y, yc, ys, yj, ysj, lr, epoch, device, encoder,
          lambda_loss1, lambda_loss2, lambda_loss3, hidden_dim,
          sample_size=10, loss_step=20, real_loss=False,
          calculate_contextual=False, calculate_structural=False):

    in_nodes = data.edge_index[0, :]
    out_nodes = data.edge_index[1, :]

    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))
    neighbor_num_list = torch.tensor(neighbor_num_list).to(device)

    in_dim = data.x.shape[1]
    GNNModel = GNNStructEncoder(
        in_dim, hidden_dim, hidden_dim, 2, sample_size, device=device,
        neighbor_num_list=neighbor_num_list, GNN_name=encoder,
        lambda_loss1=lambda_loss1, lambda_loss2=lambda_loss2, lambda_loss3=lambda_loss3)
    GNNModel.to(device)

    degree_params = list(map(id, GNNModel.degree_decoder.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params, GNNModel.parameters())

    opt = torch.optim.Adam(
        [{'params': base_params},
         {'params': GNNModel.degree_decoder.parameters(), 'lr': 1e-2}],
        lr=lr, weight_decay=0.0003)

    min_loss = float('inf')
    arg_min_loss_per_node = None

    best_auc = 0
    best_auc_contextual = 0
    best_auc_dense_structural = 0
    best_auc_joint_structural = 0
    best_auc_structure_type = 0

    results_log = []

    for i in tqdm(range(epoch), desc="Training"):
        if i % loss_step == 0:
            GNNModel.lambda_loss2 = GNNModel.lambda_loss2 + 0.5
            GNNModel.lambda_loss3 = GNNModel.lambda_loss3 / 2

        loss, loss_per_node, h_loss, degree_loss, feature_loss = GNNModel(
            data.edge_index, data.x, neighbor_num_list, neighbor_dict, device=device)

        loss_per_node = loss_per_node.cpu().detach()
        h_loss = h_loss.cpu().detach()
        degree_loss = degree_loss.cpu().detach()
        feature_loss = feature_loss.cpu().detach()

        h_loss_norm = h_loss / (torch.max(h_loss) - torch.min(h_loss) + 1e-8)
        degree_loss_norm = degree_loss / (torch.max(degree_loss) - torch.min(degree_loss) + 1e-8)
        feature_loss_norm = feature_loss / (torch.max(feature_loss) - torch.min(feature_loss) + 1e-8)

        comb_loss = (args.h_loss_weight * h_loss_norm
                     + args.degree_loss_weight * degree_loss_norm
                     + args.feature_loss_weight * feature_loss_norm)

        if real_loss:
            comp_loss = loss_per_node
        else:
            comp_loss = comb_loss

        auc_score = eval_roc_auc(y.numpy(), comp_loss.numpy()) * 100

        if len(yc) > 0:
            contextual_auc_score = eval_roc_auc(yc.numpy(), comp_loss.numpy()) * 100
        else:
            contextual_auc_score = 0.0

        if len(ys) > 0:
            dense_structural_auc_score = eval_roc_auc(ys.numpy(), comp_loss.numpy()) * 100
            joint_structural_auc_score = eval_roc_auc(yj.numpy(), comp_loss.numpy()) * 100
            structure_type_auc_score = eval_roc_auc(ysj.numpy(), comp_loss.numpy()) * 100
        else:
            dense_structural_auc_score = 0.0
            joint_structural_auc_score = 0.0
            structure_type_auc_score = 0.0

        best_auc = max(best_auc, auc_score)
        best_auc_contextual = max(best_auc_contextual, contextual_auc_score)
        best_auc_dense_structural = max(best_auc_dense_structural, dense_structural_auc_score)
        best_auc_joint_structural = max(best_auc_joint_structural, joint_structural_auc_score)
        best_auc_structure_type = max(best_auc_structure_type, structure_type_auc_score)

        if i % 10 == 0 or i == epoch - 1:
            print(f"\n[Epoch {i}/{epoch}] Loss: {loss.item():.4f}")
            print(f"  AUC (benchmark): {auc_score:.2f}  (best: {best_auc:.2f})")
            if len(yc) > 0:
                print(f"  AUC (contextual): {contextual_auc_score:.2f}  (best: {best_auc_contextual:.2f})")
            if len(ys) > 0:
                print(f"  AUC (structural): {dense_structural_auc_score:.2f}  (best: {best_auc_dense_structural:.2f})")
                print(f"  AUC (joint-type): {joint_structural_auc_score:.2f}  (best: {best_auc_joint_structural:.2f})")
                print(f"  AUC (struct+joint): {structure_type_auc_score:.2f}  (best: {best_auc_structure_type:.2f})")

        results_log.append({
            'epoch': i, 'loss': loss.item(),
            'auc_benchmark': auc_score,
            'auc_contextual': contextual_auc_score,
            'auc_structural': dense_structural_auc_score,
            'auc_joint': joint_structural_auc_score,
            'auc_struct_joint': structure_type_auc_score,
        })

        if loss < min_loss:
            min_loss = loss
            arg_min_loss_per_node = loss_per_node

        opt.zero_grad()
        loss.backward()
        opt.step()
        loss = loss.cpu().detach()

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS for {args.dataset}")
    print(f"  Best AUC (benchmark/combined): {best_auc:.2f}")
    print(f"  Best AUC (contextual):         {best_auc_contextual:.2f}")
    print(f"  Best AUC (structural):         {best_auc_dense_structural:.2f}")
    print(f"  Best AUC (joint-type):         {best_auc_joint_structural:.2f}")
    print(f"  Best AUC (struct+joint):       {best_auc_structure_type:.2f}")
    print("=" * 70)

    return min_loss.item(), arg_min_loss_per_node.cpu().detach(), results_log


def train_real_datasets(dataset_str, epoch_num=10, lr=5e-6, encoder="GCN",
                        lambda_loss1=1e-2, lambda_loss2=1e-3, lambda_loss3=1e-3,
                        sample_size=8, loss_step=20, hidden_dim=None,
                        real_loss=False, calculate_contextual=False, calculate_structural=False):

    data = load_data_local(dataset_str)
    node_features = data.x
    node_features_min = node_features.min()
    node_features_max = node_features.max()
    node_features = (node_features - node_features_min) / (node_features_max + 1e-8)
    data.x = node_features

    yc = []
    ys = []
    yj = []

    if calculate_contextual:
        if dataset_str == "inj_cora":
            yc = data.y >> 0 & 1
        else:
            data, yc = gen_contextual_outliers(data=data, n=args.contextual_n, k=args.contextual_k)
        yc = yc.cpu().detach()

    if calculate_structural:
        if dataset_str == "inj_cora":
            ys = data.y >> 1 & 1
        else:
            data, ys = gen_structural_outliers(data=data, n=args.structural_n, m=args.structural_m, p=0.2)
        ys = ys.cpu().detach()
        data, yj = gen_joint_structural_outliers(data=data, n=args.structural_n, m=args.structural_m)

    if args.use_combine_outlier:
        data.y = torch.logical_or(ys, yc).int()

    ysj = torch.logical_or(ys, yj).int() if len(ys) > 0 and len(yj) > 0 else []
    y = data.y.bool()
    y = y.cpu().detach()

    edge_index = data.edge_index.cpu()
    num_nodes = node_features.shape[0]
    self_edges = torch.tensor([[i for i in range(num_nodes)], [i for i in range(num_nodes)]])
    edge_index = torch.cat([edge_index, self_edges], dim=1)
    data.edge_index = edge_index
    data = data.to(device)

    print(f"\nDataset: {dataset_str}")
    print(f"  Nodes: {num_nodes}, Edges: {edge_index.shape[1]}, Features: {data.x.shape[1]}")
    print(f"  Encoder: {encoder}, Hidden dim: {hidden_dim}, Sample size: {sample_size}")
    print(f"  Lambda: loss1(neigh)={lambda_loss1}, loss2(feat)={lambda_loss2}, loss3(deg)={lambda_loss3}")
    print()

    loss, loss_per_node, results_log = train(
        data, y, yc, ys, yj, ysj, lr=lr, epoch=epoch_num, device=device,
        encoder=encoder, lambda_loss1=lambda_loss1, lambda_loss2=lambda_loss2,
        lambda_loss3=lambda_loss3, hidden_dim=hidden_dim, sample_size=sample_size,
        loss_step=loss_step, real_loss=real_loss,
        calculate_contextual=calculate_contextual, calculate_structural=calculate_structural)

    return results_log


# ======================== MAIN ========================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction')
    parser.add_argument('--dataset', type=str, default="inj_cora")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--lambda_loss1', type=float, default=1e-2)
    parser.add_argument('--lambda_loss2', type=float, default=0.5)
    parser.add_argument('--lambda_loss3', type=float, default=0.8)
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--encoder', type=str, default="GCN")
    parser.add_argument('--loss_step', type=int, default=30)
    parser.add_argument('--real_loss', type=bool, default=False)
    parser.add_argument('--neigh_loss', type=str, default="KL")
    parser.add_argument('--h_loss_weight', type=float, default=1.0)
    parser.add_argument('--feature_loss_weight', type=float, default=2.0)
    parser.add_argument('--degree_loss_weight', type=float, default=1.0)
    parser.add_argument('--calculate_contextual', type=bool, default=True)
    parser.add_argument('--contextual_n', type=int, default=70)
    parser.add_argument('--contextual_k', type=int, default=10)
    parser.add_argument('--calculate_structural', type=bool, default=True)
    parser.add_argument('--structural_n', type=int, default=70)
    parser.add_argument('--structural_m', type=int, default=10)
    parser.add_argument('--use_combine_outlier', type=bool, default=False)

    args = parser.parse_args()

    print("=" * 70)
    print("GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}, Encoder: {args.encoder}, Dim: {args.dimension}")
    print(f"LR: {args.lr}, Epochs: {args.epoch_num}")

    results = train_real_datasets(
        dataset_str=args.dataset, lr=args.lr, epoch_num=args.epoch_num,
        lambda_loss1=args.lambda_loss1, lambda_loss2=args.lambda_loss2,
        lambda_loss3=args.lambda_loss3, encoder=args.encoder,
        sample_size=args.sample_size, loss_step=args.loss_step,
        hidden_dim=args.dimension, real_loss=args.real_loss,
        calculate_contextual=args.calculate_contextual,
        calculate_structural=args.calculate_structural)
