from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------
batch_size = 64
block_size = 256
max_iters =500
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embed = 384
n_head = 6
n_layer = 12
droupout = 0.1

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x




@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 6
    n_embed: int = 768



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):

