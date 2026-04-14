# GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction
## Project 3 — CS60078 Complex Network Theory, Spring 2025-26

### Setup Instructions

#### 1. Prerequisites
- Python >= 3.10
- pip (Python package manager)

#### 2. Clone the Repository
```bash
git clone https://github.com/Graph-COM/GAD-NR.git
cd GAD-NR
```

#### 3. Install Dependencies
```bash
pip install torch torchvision
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv  # Optional C++ extensions
pip install pygod==0.3.1 networkx matplotlib scipy seaborn scikit-learn requests tqdm
```

**Note on C++ Extensions:** The optional PyG C++ extensions (`torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv`) require compilation and may need specific CUDA versions. If compilation fails (e.g., due to memory limits), the provided `run_gadnr.py` script includes runtime mocks for these modules — the core GAD-NR model does **not** use them.

#### 4. Download Datasets
The datasets are loaded via `pygod.utils.load_data()`. If internet access is restricted, you can clone the data repo manually:
```bash
git clone https://github.com/pygod-team/data.git pygod_data
cd pygod_data
# Extract each dataset
python -c "import zipfile; [zipfile.ZipFile(f'{d}.pt.zip').extractall('extracted/') for d in ['inj_cora','weibo','reddit','disney','books','enron']]"
```
Then update the `DATA_DIR` path in `run_gadnr.py` to point to the `extracted/` folder.

#### 5. Running the Code

**Quick run (inj_cora dataset with default hyperparameters):**
```bash
python run_gadnr.py --dataset inj_cora --epoch_num 100 --dimension 128 --encoder GCN
```

**Run on other datasets (use hidden dimension 16 as per paper):**
```bash
python run_gadnr.py --dataset weibo --epoch_num 100 --dimension 16
python run_gadnr.py --dataset reddit --epoch_num 100 --dimension 16
python run_gadnr.py --dataset disney --epoch_num 100 --dimension 16
python run_gadnr.py --dataset books --epoch_num 100 --dimension 16
python run_gadnr.py --dataset enron --epoch_num 100 --dimension 16
```

**Alternatively, use the original Jupyter notebooks:**
```bash
jupyter notebook GAD-NR_inj_cora.ipynb
```

#### 6. Key Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `inj_cora` | Dataset name |
| `--encoder` | `GCN` | GNN encoder: GCN, GIN, GraphSAGE, GAT |
| `--dimension` | `128` | Hidden dimension (128 for Cora, 16 for others) |
| `--epoch_num` | `100` | Number of training epochs |
| `--lr` | `0.01` | Learning rate |
| `--lambda_loss1` | `0.01` | Neighbor reconstruction loss weight |
| `--lambda_loss2` | `0.5` | Feature reconstruction loss weight |
| `--lambda_loss3` | `0.8` | Degree reconstruction loss weight |
| `--neigh_loss` | `KL` | Neighborhood loss type: KL or W2 |
| `--sample_size` | `10` | Number of neighbors to sample |

#### 7. Output
The script prints per-epoch AUC scores for:
- **Benchmark anomaly detection** (combined labels)
- **Contextual anomaly detection**
- **Structural anomaly detection**
- **Joint-type anomaly detection**
- **Structural + Joint-type combined**

#### 8. Libraries Used
- `torch` (PyTorch) — deep learning framework
- `torch_geometric` (PyG) — graph neural network layers (GCN, GIN, SAGE, GAT)
- `pygod` — graph anomaly detection utilities (data loading, outlier injection, AUC evaluation)
- `scipy` — matrix operations (sqrtm for W2 loss)
- `scikit-learn` — evaluation metrics
- `matplotlib`, `seaborn` — visualization
- `networkx` — graph utilities
- `tqdm` — progress bars

---

## Endterm: GAD-NR++ (Enhanced GAD-NR)

### Running GAD-NR++
```bash
python gadnr_plus.py --dataset disney --epochs 50 --num_runs 3
python gadnr_plus.py --dataset books --epochs 50 --num_runs 1
python gadnr_plus.py --dataset inj_cora --epochs 100 --num_runs 3
```

### Improvements over GAD-NR
1. **Multi-head GAT encoder** (4 heads, 2 layers, residual + LayerNorm)
2. **GMM neighbourhood decoder** (K=2 components, NLL loss)
3. **Learnable anomaly score fusion** (MLP replaces heuristic weights)

### Additional Libraries
- `torch_geometric.nn.GATConv` — Graph Attention Network layer
