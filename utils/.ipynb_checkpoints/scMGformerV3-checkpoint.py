import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

__all__ = [
    "GraphformerWithPos",
    "Graphformer",
    "SinusoidalPositionalEmbedding",
    "ProjectionHead",
    "SupConLoss",
]

############################################################
# Utility: generalized kernel (used by Performer attention)
############################################################

def generalized_kernel(
    data: torch.Tensor,
    *,
    projection_matrix: torch.Tensor | None,
    kernel_fn=nn.ReLU(),
    kernel_epsilon: float = 1e-3,
    normalize_data: bool = True,
) -> torch.Tensor:
    """Random feature mapping for Performer attention.
    If *projection_matrix* is *None*, falls back to the input-space kernel.
    """
    *_, d = data.shape
    data_normalizer = (d ** -0.25) if normalize_data else 1.0

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    data_dash = F.linear(data_normalizer * data, projection_matrix)  # (..., m)
    return kernel_fn(data_dash) + kernel_epsilon

############################################################
# Core building blocks
############################################################

class GraphformerLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int = 4,
        dropout: float = 0.1,
        use_performer: bool = False,
        performer_dim: int | None = None,
        kernel_fn=nn.ReLU(),
        kernel_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.use_performer = use_performer
        self.kernel_fn = kernel_fn
        self.kernel_epsilon = kernel_epsilon

        # Projections
        self.q_proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.out_proj = nn.Linear(out_dim * heads, out_dim)

        # Random projection matrix for Performer
        if use_performer:
            proj_dim = performer_dim if performer_dim is not None else out_dim
            self.register_buffer(
                "rand_proj",
                torch.randn(proj_dim, out_dim) * (1.0 / math.sqrt(out_dim)),
                persistent=False,
            )
        else:
            self.rand_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        H = self.heads
        d_head = self.out_proj.in_features // H

        # Project to heads
        Q = self.q_proj(x).view(N, H, d_head)
        K = self.k_proj(x).view(N, H, d_head)
        V = self.v_proj(x).view(N, H, d_head)

        if not self.use_performer:
            attn_scores = torch.einsum("nhd,mhd->nhm", Q, K) / math.sqrt(d_head)
            attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
            out = torch.einsum("nhm,mhd->nhd", attn_probs, V)
        else:
            # FAVOR+ (Performer) attention
            Q_prime = generalized_kernel(Q, projection_matrix=self.rand_proj, kernel_fn=self.kernel_fn,
                                         kernel_epsilon=self.kernel_epsilon)
            K_prime = generalized_kernel(K, projection_matrix=self.rand_proj, kernel_fn=self.kernel_fn,
                                         kernel_epsilon=self.kernel_epsilon)
            K_sum = K_prime.sum(dim=0)  # (H, m)
            KV = torch.einsum("nhm,nhd->hmd", K_prime, V)
            z = 1.0 / (torch.einsum("nhm,hm->nh", Q_prime, K_sum) + 1e-6)
            out = torch.einsum("nhm,hmd,nh->nhd", Q_prime, KV, z)

        return self.out_proj(out.reshape(N, H * d_head))


class SimpleGCN(nn.Module):
    """Single-hop symmetric-normalized GCN layer (no parameters)."""
    def forward(self, x: torch.Tensor, edge_index, edge_weight=None):
        if edge_index is None:
            return x
        N = x.size(0)
        row, col = edge_index
        deg = degree(col, N, dtype=x.dtype).clamp(min=1)
        norm = (deg[row].pow(-0.5) * deg[col].pow(-0.5)).view(-1)
        value = norm if edge_weight is None else norm * edge_weight
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        return matmul(adj, x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_multiplier: int = 4, dropout: float = 0.1):
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
    """Fixed sinusoidal positional embeddings (same formulation as in the Transformer paper)."""
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return positional embeddings matching the first (sequence) dimension of *x*."""
        if x.dim() == 2:  # [N, D]
            return self.pe[0, :x.size(0), :]
        elif x.dim() == 3:  # [B, N, D]
            return self.pe[:, :x.size(1), :].expand(x.size(0), -1, -1)
        else:
            raise ValueError("Unsupported input shape for position encoding")


class GraphformerEncoderLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int = 4,
        dropout: float = 0.1,
        use_performer: bool = False,
        performer_dim: int | None = None,
        use_gcn: bool = False,
    ) -> None:
        super().__init__()
        self.use_gcn = use_gcn
        if use_gcn:
            self.gcn = SimpleGCN()
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

    def forward(self, x, edge_index=None, edge_weight=None):
        # Optionally blend GCN message passing
        if self.use_gcn and edge_index is not None:
            x = x + self.gate.tanh() * self.gcn(x, edge_index, edge_weight)

        # Self‑attention
        attn_out = self.attn(self.ln1(x))
        x = self.proj_residual(x) + self.dropout(attn_out)

        # Feed‑forward network
        x = x + self.ffn(self.ln2(x))
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),  # features lie on unit sphere
        )

    def forward(self, x):
        return self.net(x)


class SupConLoss(nn.Module):
    """Supervised contrastive loss (SimCLR-style)."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        features = F.normalize(features, dim=-1)
        sim = torch.matmul(features, features.T) / self.T  # (N, N)
        # Logits stabilization trick
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).to(features.device)
        mask.fill_diagonal_(False)

        pos_exp = exp_sim * mask
        neg_exp = exp_sim * (~mask)
        pos_sum = pos_exp.sum(dim=1)
        denom = pos_sum + neg_exp.sum(dim=1)
        loss = -torch.log((pos_sum + 1e-8) / (denom + 1e-8))
        return loss.mean()


class Graphformer(nn.Module):
    """Stack of Graphformer encoder layers (without positional concat)."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        use_performer: bool = False,
        performer_dim: int | None = None,
        use_gcn: bool = False,
        proj_dim: int | None = 128,  # set *None* to disable SupCon branch
    ) -> None:
        super().__init__()
        self.use_projection = proj_dim is not None

        self.layers = nn.ModuleList([
            GraphformerEncoderLayer(
                in_dim if i == 0 else hidden_dim,
                hidden_dim,
                heads=heads,
                dropout=dropout,
                use_performer=use_performer,
                performer_dim=performer_dim,
                use_gcn=use_gcn,
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

        if self.use_projection:
            self.proj_head = ProjectionHead(hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor, edge_index=None, edge_weight=None):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        h = self.norm(h)
        logits = self.classifier(h)

        if self.use_projection:
            proj = self.proj_head(h)
            return logits, proj
        return logits

############################################################
# NEW: GraphformerWithPos – concatenates positional encodings
############################################################

class GraphformerWithPos(nn.Module):
    """Wraps *Graphformer*: concatenates fixed positional embeddings to node features."""
    def __init__(
        self,
        node_feat_dim: int,
        pos_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        max_nodes: int = 10000,
        **graphformer_kwargs,
    ) -> None:
        """
        Args:
            node_feat_dim: dimension of raw node attributes.
            pos_dim: size of sinusoidal positional embeddings.
            hidden_dim / out_dim: Graphformer hyper‑parameters (as before).
            max_nodes: longest graph you expect (for sinusoidal table size).
            **graphformer_kwargs: forwarded to :class:`Graphformer` (e.g., num_layers,…).
        """
        super().__init__()
        self.pos_encoder = SinusoidalPositionalEmbedding(pos_dim, max_len=max_nodes)
        self.model = Graphformer(
            in_dim=node_feat_dim + pos_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            **graphformer_kwargs,
        )

    def forward(self, x: torch.Tensor, edge_index=None, edge_weight=None):
        """x: [N, node_feat_dim] – returns same outputs as *Graphformer*."""
        pos = self.pos_encoder(x)  # [N, pos_dim]
        x_cat = torch.cat([x, pos.to(x.dtype)], dim=-1).to(device)
        return self.model(x_cat, edge_index, edge_weight)