import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from torch_scatter import scatter_add, scatter_softmax

__all__ = ["scGFformer", "info_nce_loss"]

def generalized_kernel(
    data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=1e-3, normalize_data=True
):
    *_, d = data.shape
    data_normalizer = (d ** -0.25) if normalize_data else 1.0
    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon
    data_dash = F.linear(data_normalizer * data, projection_matrix)
    return kernel_fn(data_dash) + kernel_epsilon

def build_knn_graph(x, *, n_neighbors=10, max_neighbors=5, sampling_method="random", metric="cos"):
    N = x.size(0)
    device = x.device
    if metric == "cos":
        x_norm = F.normalize(x, dim=-1)
        sim = x_norm @ x_norm.T
        _, idx = torch.topk(sim, n_neighbors + 1, dim=-1)
    elif metric == "l2":
        dist = torch.cdist(x, x, p=2)
        _, idx = torch.topk(-dist, n_neighbors + 1, dim=-1)
    else:
        raise ValueError("metric must be 'cos' or 'l2'")
    idx = idx[:, 1:]
    if max_neighbors and max_neighbors < n_neighbors:
        if sampling_method == "topk":
            idx = idx[:, :max_neighbors]
        elif sampling_method == "random":
            rand_idx = torch.rand_like(idx.float()).argsort(dim=-1)
            idx = idx.gather(1, rand_idx[:, :max_neighbors])
        else:
            raise ValueError("sampling_method must be 'topk' or 'random'")
    row = torch.arange(N, device=device).unsqueeze(1).expand_as(idx)
    edge_index = torch.stack([row.reshape(-1), idx.reshape(-1)], dim=0)
    return edge_index

class GraphformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, *, heads=4, dropout=0.1, use_performer=False,
                 performer_dim=None, kernel_fn=nn.ReLU(), kernel_epsilon=1e-3):
        super().__init__()
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.use_performer = use_performer
        self.kernel_fn = kernel_fn
        self.kernel_epsilon = kernel_epsilon
        self.q_proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.out_proj = nn.Linear(out_dim * heads, out_dim)
        if use_performer:
            proj_dim = performer_dim if performer_dim is not None else out_dim
            self.register_buffer(
                "rand_proj",
                torch.randn(proj_dim, out_dim) * (1.0 / math.sqrt(out_dim)),
                persistent=False,
            )
        else:
            self.rand_proj = None

    def forward(self, x):
        N, H = x.size(0), self.heads
        d_head = self.out_proj.in_features // H
        Q = self.q_proj(x).view(N, H, d_head)
        K = self.k_proj(x).view(N, H, d_head)
        V = self.v_proj(x).view(N, H, d_head)
        if not self.use_performer:
            attn = torch.einsum("nhd,mhd->nhm", Q, K) / math.sqrt(d_head)
            attn = self.dropout(torch.softmax(attn, dim=-1))
            out = torch.einsum("nhm,mhd->nhd", attn, V)
        else:
            Qp = generalized_kernel(Q, projection_matrix=self.rand_proj, kernel_fn=self.kernel_fn,
                                    kernel_epsilon=self.kernel_epsilon)
            Kp = generalized_kernel(K, projection_matrix=self.rand_proj, kernel_fn=self.kernel_fn,
                                    kernel_epsilon=self.kernel_epsilon)
            K_sum = Kp.sum(dim=0)
            KV = torch.einsum("nhm,nhd->hmd", Kp, V)
            z = 1. / (torch.einsum("nhm,hm->nh", Qp, K_sum) + 1e-6)
            out = torch.einsum("nhm,hmd,nh->nhd", Qp, KV, z)
        return self.out_proj(out.reshape(N, H * d_head))

class SparseGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.1, concat=True):
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(1, heads, out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(1, heads, out_dim))
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        N, H, D = x.size(0), self.heads, self.out_dim
        x_proj = self.lin(x).view(N, H, D)
        row, col = edge_index
        x_i, x_j = x_proj[row], x_proj[col]
        alpha = (x_i * self.attn_l).sum(-1) + (x_j * self.attn_r).sum(-1)
        alpha = self.leaky_relu(alpha)
        alpha = torch.exp(alpha - alpha.max())
        alpha = scatter_softmax(alpha, row, dim=0)
        alpha = self.dropout(alpha)
        out = x_j * alpha.unsqueeze(-1)
        out = scatter_add(out, row, dim=0, dim_size=N)
        return out.reshape(N, H * D) if self.concat else out.mean(dim=1)

class GeneSE(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 2:
            z = x.mean(dim=0, keepdim=True)
        else:
            z = x.mean(dim=1, keepdim=True)
        s = self.sigmoid(self.fc2(self.relu(self.fc1(z))))
        return x * s if x.dim() == 2 else x * s.expand_as(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_multiplier=4, dropout=0.1):
        super().__init__()
        hidden = dim * hidden_multiplier
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer("pe", None, persistent=False)
        self.max_len = 0

    def _update_pe(self, needed_len, device):
        if self.pe is not None and self.max_len >= needed_len and self.pe.device == device:
            return
        pos = torch.arange(0, needed_len, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.dim, 2, device=device) * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(needed_len, self.dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe, self.max_len = pe.unsqueeze(0), needed_len

    def forward(self, x):
        device = x.device
        needed_len = x.size(0) if x.dim() == 2 else x.size(1)
        self._update_pe(needed_len, device)
        if x.dim() == 2:
            return self.pe[0, :needed_len]
        return self.pe[:, :needed_len].expand(x.size(0), -1, -1)

class MultiScaleAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, *, heads=4, dropout=0.1, use_performer=False,
                 performer_dim=None, use_gat=False, use_gene_att=False):
        super().__init__()
        self.use_gat = use_gat
        self.use_gene_att = use_gene_att
        if use_gat:
            self.gat = SparseGraphAttention(in_dim, out_dim, heads=4, dropout=dropout, concat=False)
            self.gate = nn.Parameter(torch.tensor(0.5))
        self.ln1 = nn.LayerNorm(in_dim)
        self.attn = GraphformerLayer(
            in_dim,
            out_dim,
            heads=heads,
            dropout=dropout,
            use_performer=use_performer,
            performer_dim=performer_dim,
        )
        self.proj_residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(out_dim)
        self.ffn = FeedForward(out_dim, dropout=dropout)
        if use_gene_att:
            self.gene_att = GeneSE(out_dim)

    def forward(self, x, edge_index=None, edge_weight=None):
        if self.use_gat and edge_index is not None:
            x = x + self.gate.tanh() * self.gat(x, edge_index)
        x = self.proj_residual(x) + self.dropout(self.attn(self.ln1(x)))
        x = x + self.ffn(self.ln2(x))
        if self.use_gene_att:
            x = self.gene_att(x)
        return x

class scGFformer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        *,
        num_layers=2,
        heads=4,
        dropout=0.1,
        use_performer=False,
        performer_dim=None,
        use_gat=False,
        use_pe=False,
        use_gene_att=False,
        dynamic_graph=False,
        n_neighbors=10,
        max_neighbors=5,
        sampling_method="random",
        metric="cos",
        rebuild_interval=10,
        contrastive=False,
        proj_dim=128
    ):
        super().__init__()
        self.use_pe = use_pe
        if use_pe:
            self.pos_encoder = SinusoidalPositionalEmbedding(in_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.layers = nn.ModuleList([
            MultiScaleAttentionLayer(
                hidden_dim, hidden_dim, heads=heads, dropout=dropout,
                use_performer=use_performer, performer_dim=performer_dim,
                use_gat=use_gat, use_gene_att=use_gene_att)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.contrastive = contrastive
        if contrastive:
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, proj_dim)
            )
        self.dynamic = dynamic_graph and use_gat
        self.n_neighbors, self.max_neighbors = n_neighbors, max_neighbors
        self.sampling_method, self.metric = sampling_method, metric
        self.rebuild_interval = max(1, rebuild_interval)
        self._edge_cpu = None
        self._step = 0

    def _maybe_rebuild_graph(self, h_detached: torch.Tensor):
        if not self.dynamic:
            return None
        if self._edge_cpu is None or self._step % self.rebuild_interval == 0:
            with torch.no_grad():
                self._edge_cpu = build_knn_graph(
                    h_detached.cpu(),
                    n_neighbors=self.n_neighbors,
                    max_neighbors=self.max_neighbors,
                    sampling_method=self.sampling_method,
                    metric=self.metric,
                )
        self._step += 1
        return self._edge_cpu

    def forward(self, x: torch.Tensor, edge_index=None, edge_weight=None, return_projection=False):
        if self.use_pe:
            x = x + self.pos_encoder(x) / math.sqrt(x.size(-1))
        h = self.input_proj(x)
        edge_cpu = self._maybe_rebuild_graph(h.detach())
        ei = edge_cpu.to(x.device) if edge_cpu is not None else edge_index
        for layer in self.layers:
            h = layer(h, ei, edge_weight)
        h_norm = self.norm(h)
        logits = self.classifier(h_norm)
        if self.contrastive and return_projection:
            proj = F.normalize(self.projector(h_norm), dim=-1)
            return logits, proj
        return logits

def info_nce_loss(z1, z2, temperature=0.5):
    N = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(N, device=z1.device)
    loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)
    return loss * 0.5