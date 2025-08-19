import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, time_steps, n_features, device, n=10000): # n value taken from Attention is All you need
        super(PositionalEncoding, self).__init__()
        self.time_steps = time_steps
        self.n_features = n_features
        self.device = device
        self.n = n

    def get_positional_encoding(self):
        P = torch.zeros(self.time_steps, self.n_features, device=self.device)
        for f in range(int(self.time_steps/2)):
            for i in range(self.n_features):
                den = np.power(self.n, 2*i/self.n_features)
                P[2*f, i] = np.sin(f/den)
                P[2*f+1, i] = np.cos(f/den)
        return P

    def forward(self, x):
        positional_matrix = self.get_positional_encoding()
        return x + positional_matrix


class TransformerBlock(nn.Module):
    def __init__(self, time_steps, n_features, n_heads, N, dropout):
        super(TransformerBlock, self).__init__()
        self.time_steps = time_steps
        self.n_features = n_features
        self.n_heads = n_heads
        self.N = N
        self.dropout = dropout

        self.trans_block = nn.ModuleList([nn.MultiheadAttention(self.n_features, self.n_heads, batch_first=True, dropout=self.dropout) for _ in range(self.N)])

        self.layer_norm = torch.nn.LayerNorm(self.n_features)

    def forward(self, x):
        res = x
        for t in self.trans_block:
            x = self.layer_norm(x)
            x = t(x, x, x)[0] # SelfAttention returns also weights
            x = x + res
            res = x
        return x
