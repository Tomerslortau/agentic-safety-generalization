"""Residual MLP student for learning the control mappings.

Paper configuration for the reported runs: 3 layers, width 2048, GELU, LayerNorm.
The defaults below reproduce the original architecture (5 layers, ReLU, no
LayerNorm); pass --num-layers 3 --activation gelu --use-layernorm to match the paper.
"""

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=5,
                 activation='relu', use_layernorm=False):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = (nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
                      if use_layernorm else None)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"unknown activation: {activation}")

    def forward(self, x):
        x = self.act(self.input_proj(x))
        for i, layer in enumerate(self.layers):
            inp = self.norms[i](x) if self.norms is not None else x
            x = x + self.act(layer(inp))
        return self.output_proj(x)
