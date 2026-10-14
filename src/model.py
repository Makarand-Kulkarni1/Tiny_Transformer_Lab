from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int = 73     # number of distinct tokens (from our tokenizer)
    block_size: int = 272    # longest sequence the model can read at once
    n_layer: int = 4         # how many transformer blocks are stacked
    n_head: int = 4          # attention heads per block
    n_embd: int = 128        # size of each token's vector (C)
    dropout: float = 0.1     # fraction of values randomly zeroed during training

class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # lower-triangular matrix of ones: the causal mask
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)       # (B, T, head_size)
        q = self.query(x)     # (B, T, head_size)
        v = self.value(x)     # (B, T, head_size)

        # 1: score every query against every key, scaled
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)      # (B, T, T)

        # 2: causal mask: forbidden positions get -infinity
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # 3: softmax turns each row into weights that sum to 1, then dropout
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 4: blend the values using those weights
        out = wei @ v                                                 # (B, T, head_size)
        return out

class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)   # what the token is
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)   # where it is
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)       # [0, 1, ..., T-1]
        x = self.tok_emb(idx) + self.pos_emb(pos)      # (B,T,C) + (T,C) -> (B,T,C)
        return self.drop(x)