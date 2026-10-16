"""
Spatial Graph Convolutional Network (GCN) Layer and Architecture.


Mathematical Formulation:
--------------------------
For a graph with N nodes, feature matrix X of shape [N, F], and adjacency matrix A of shape [N, N]:
1. Add self-loops to ensure each team retains its own features:
       A_tilde = A + I_N
   If edge weights (ICCS) are provided:
       A_tilde[i, j] = w_ij  for i != j
       A_tilde[i, i] = 1.0   (self-loop weight)

2. Compute the diagonal degree matrix D_tilde:
       D_tilde[i, i] = sum_j A_tilde[i, j]

3. Compute the symmetric normalized adjacency matrix A_hat:
       A_hat[i, j] = A_tilde[i, j] / sqrt(D_tilde[i, i] * D_tilde[j, j])

4. Message Passing & Linear Transformation:
       H = Activation( A_hat @ X @ W + b )
   where W is a learnable weight matrix of shape [F, D] and b is a bias vector of shape [D].
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_dense_normalized_adjacency(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor] = None,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    Constructs the symmetric normalized adjacency matrix A_hat in dense tensor form.

    Formula:
        A_hat = D^(-1/2) * (A + I) * D^(-1/2)
    """
    if num_nodes == 0:
        return torch.zeros((0, 0), dtype=torch.float32)

    device = edge_index.device if edge_index.numel() > 0 else torch.device("cpu")
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)

    # 1. Fill edges from edge_index
    if edge_index.numel() > 0:
        src = edge_index[0]
        dst = edge_index[1]
        if edge_weights is not None and edge_weights.numel() > 0:
            for s, d, w in zip(src, dst, edge_weights):
                A[s, d] = w
        else:
            A[src, dst] = 1.0

    # 2. Add self-loops (diagonal = 1.0)
    if add_self_loops:
        A = A + torch.eye(num_nodes, device=device)

    # 3. Compute degree D_ii = sum_j A_ij
    deg = torch.sum(A, dim=1)  # shape [N]
    # Prevent division by zero
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
    deg_inv_sqrt[deg == 0] = 0.0

    # 4. Symmetric normalization: D^(-1/2) @ A @ D^(-1/2)
    # Using broadcasting: A_hat[i, j] = deg_inv_sqrt[i] * A[i, j] * deg_inv_sqrt[j]
    A_hat = deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)
    return A_hat


class SimpleGCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer.

    Equation:
        H^(l+1) = Activation( A_hat @ H^(l) @ W + b )
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight matrix W and bias b
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.linear.weight, a=1.0)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features tensor of shape [N, in_features].
            edge_index: Edge indices tensor of shape [2, E].
            edge_weights: Edge weight tensor of shape [E].

        Returns:
            Output node embeddings of shape [N, out_features].
        """
        num_nodes = x.size(0)
        if num_nodes == 0:
            return torch.zeros((0, self.out_features), dtype=x.dtype, device=x.device)

        # 1. Compute normalized adjacency A_hat of shape [N, N]
        A_hat = build_dense_normalized_adjacency(num_nodes, edge_index, edge_weights)

        # 2. Linear projection: X @ W + b
        # shape: [N, out_features]
        projected = self.linear(x)

        # 3. Message passing via matrix multiplication: A_hat @ projected
        # shape: [N, out_features]
        out = torch.matmul(A_hat, projected)
        return out


class SpatialGCN(nn.Module):
    """
    Two-layer Spatial Graph Convolutional Network.

    Architecture:
        Layer 1: GCN(in_features -> hidden_dim) + ReLU + Dropout
        Layer 2: GCN(hidden_dim -> out_dim)

    Maps raw clique features [K, F] into spatial clique embeddings [K, D].
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden_dim: int = 16,
        out_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = SimpleGCNLayer(in_features, hidden_dim)
        self.conv2 = SimpleGCNLayer(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [N, in_features]
            edge_index: [2, E]
            edge_weights: [E]

        Returns:
            Clique embeddings H of shape [N, out_dim]
        """
        h = self.conv1(x, edge_index, edge_weights)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weights)
        return h
