"""Karpathy's Zero-to-Hero GPT (gpt-dev notebook), as a plain script with knobs to break it.

It knows nothing about FlowScope. Watch it with:
    uv run flowscope run examples/gpt.py --layers 12 --no-residual
"""
import argparse, os, urllib.request

import torch
import torch.nn as nn
from torch.nn import functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, cfg, head_size):
        super().__init__()
        self.cfg = cfg
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5 if self.cfg.scale_attention else 1.0)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, cfg, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, cfg):
        super().__init__()
        self.residual = cfg.residual
        self.sa = MultiHeadAttention(cfg, cfg.n_embd // cfg.n_head)
        self.ffwd = FeedFoward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd) if cfg.layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(cfg.n_embd) if cfg.layernorm else nn.Identity()

    def forward(self, x):
        if self.residual:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.sa(self.ln1(x))
            x = self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        return logits, F.cross_entropy(logits.view(B * T, C), targets.view(B * T))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", dest="n_layer", type=int, default=4)
    ap.add_argument("--embd", dest="n_embd", type=int, default=64)
    ap.add_argument("--heads", dest="n_head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--no-residual", dest="residual", action="store_false")
    ap.add_argument("--no-layernorm", dest="layernorm", action="store_false")
    ap.add_argument("--no-scale", dest="scale_attention", action="store_false", help="drop the C**-0.5 in attention")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt"))
    return ap.parse_args(argv)


def main(argv=None):
    cfg = parse_args(argv)
    torch.manual_seed(1337)
    if not os.path.exists(cfg.data):
        os.makedirs(os.path.dirname(cfg.data), exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, cfg.data)
    text = open(cfg.data, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    train_data = data[:int(0.9 * len(data))]

    def get_batch():
        ix = torch.randint(len(train_data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([train_data[i:i + cfg.block_size] for i in ix])
        y = torch.stack([train_data[i + 1:i + cfg.block_size + 1] for i in ix])
        return x.to(cfg.device), y.to(cfg.device)

    model = GPT(cfg, len(chars)).to(cfg.device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for it in range(cfg.iters):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if it % 200 == 0 or it == cfg.iters - 1:
            print(f"step {it}: loss {loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()
