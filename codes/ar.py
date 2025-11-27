import math
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from pathlib import Path

# --- 1. Configuration & Data Loading ---
batch_size = 64                # match MDLM
block_size = 256
max_iters = 5000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set AR model to match MDLM parameterization
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.1

print(f"Using device: {device}")

# Create model directory
Path("./models").mkdir(parents=True, exist_ok=True)

# Download TinyShakespeare
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = requests.get(url).text

# Character-level tokenizer
chars = sorted(list(set(text)))
chars.append('[MASK]')  # keep same vocab as MDLM for fair embeddings comparison
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
mask_token_id = stoi['[MASK]']

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loader
def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i+block_size] for i in ix])
    y = torch.stack([data_[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, model_type='ar'):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_interval):
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu',
                          dtype=torch.float16, enabled=(device == 'cuda')):
                X, Y = get_batch(split)
                _, loss = model(X, targets=Y)
            losses.append(loss.item())
        out[split] = float(sum(losses) / len(losses))
    model.train()
    return out

# --- 2. Model 1: Autoregressive (AR) Transformer ---

class AutoRegressiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Use similar TransformerEncoderLayer settings to MDLM: gelu, pre-LN (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # tie weights like MDLM
        self.lm_head.weight = self.token_embedding_table.weight

        # init weights similar to MDLM
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        float_mask = torch.zeros(sz, sz, device=device).float()
        float_mask.masked_fill_(mask, float('-inf'))
        return float_mask

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb

        causal_mask = self._generate_square_subsequent_mask(T)
        output = self.transformer_encoder(src=x, mask=causal_mask)
        output = self.ln_f(output)
        logits = self.lm_head(output)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:].to(device)
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu',
                          dtype=torch.float16, enabled=(device == 'cuda')):
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

# --- 3. Instantiate and print parameter count ---

print("\n--- Building AR model with MDLM-like paramization ---")
ar_model = AutoRegressiveModel().to(device)

total_params = sum(p.numel() for p in ar_model.parameters())
print(f"AR model parameters: {total_params/1e6:.2f}M")

# --- 4. Training ---

ar_optimizer = torch.optim.AdamW(ar_model.parameters(), lr=learning_rate)
ar_scaler = GradScaler(enabled=(device == 'cuda'))
ar_scheduler = CosineAnnealingLR(ar_optimizer, T_max=max_iters)

print("\n--- Training Autoregressive (AR) Model ---")
for it in tqdm(range(max_iters), desc="AR training"):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(ar_model, model_type='ar')
        print(f"AR Model: step {it}, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    ar_optimizer.zero_grad(set_to_none=True)
    with autocast(device_type='cuda' if device == 'cuda' else 'cpu',
                  dtype=torch.float16, enabled=(device == 'cuda')):
        xb, yb = get_batch('train')
        _, loss = ar_model(xb, yb)

    ar_scaler.scale(loss).backward()
    ar_scaler.step(ar_optimizer)
    ar_scaler.update()
    ar_scheduler.step()

print("Training finished. Saving AR model...")
torch.save(ar_model.state_dict(), './models/ar_model.pth')
print("AR model saved to ./models/ar_model.pth")

# --- 5. Generate ---

print("\n--- Generating from AR Model ---")
start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
ar_result = ar_model.generate(start_context, max_new_tokens=200)[0].tolist()
print(decode(ar_result))
print("---------------------------------")
