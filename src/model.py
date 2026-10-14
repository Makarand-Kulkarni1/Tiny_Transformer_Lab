from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)       # (B, T, T) scores
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # hide the future
        wei = torch.softmax(wei, dim=-1)                              # rows sum to 1
        wei = self.dropout(wei)
        return wei @ v                                                # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, joined and mixed."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must divide evenly by n_head"
        head_size = config.n_embd // config.n_head            # 128 // 4 = 32
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)   # mixes the heads' outputs
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, C)
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    """Per-token feed-forward network: expand, non-linearity, shrink."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),   # 128 -> 512
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),   # 512 -> 128
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention then MLP, each with a residual connection."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)   # what the token is
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)   # where it is
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Small random starting weights, as in GPT-2."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, C)
        x = self.blocks(x)                                     # (B, T, C)
        x = self.ln_f(x)                                       # (B, T, C)
        logits = self.head(x)                                  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # flatten to (B*T, vocab_size)
                targets.view(-1),                   # flatten to (B*T,)
                ignore_index=-100,                  # skip the masked question and padding
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_id=None, temperature=0.0):
        """Extend idx (shape (B, T)) one token at a time. Call model.eval() first.
        temperature=0 means greedy: always pick the most likely token."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]        # never exceed block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                          # scores for the next token only
            if temperature == 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx