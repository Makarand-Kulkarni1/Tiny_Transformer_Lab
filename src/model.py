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