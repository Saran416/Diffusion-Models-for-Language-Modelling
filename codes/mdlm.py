import math
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
batch_size = 64
block_size = 256
max_iters = 50000
eval_interval = 500
learning_rate = 3e-4
warmup_iters = 2000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.1

# MDLM specific: continuous time formulation
sigma_min = 1e-4
sigma_max = 1.0

print(f"Device: {device}")
print(f"Model: {n_layer} layers, {n_embd} dim, {n_head} heads")

Path("./models").mkdir(parents=True, exist_ok=True)

# --- Data Loading ---
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = requests.get(url).text

chars = sorted(list(set(text)))
chars.append('[MASK]')
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
mask_token_id = stoi['[MASK]']

print(f"Vocab size: {vocab_size}, Mask ID: {mask_token_id}")

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i+block_size] for i in ix])
    return x.to(device)

# --- MDLM Model (SUBS Parameterization) ---

class MDLM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer encoder (no time conditioning for SUBS)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Tie weights (like BERT)
        self.head.weight = self.tok_emb.weight
        
        self.apply(self._init_weights)
        print(f"Parameters: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")

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

    def forward(self, x):
        B, T = x.shape
        
        tok_emb = self.tok_emb(x)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_emb(pos)
        
        x = self.drop(tok_emb + pos_emb)
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

    def q_sample(self, x0, t):
        B, T = x0.shape
        
        # Compute sigma (noise level) for this timestep
        log_sigma = torch.log(torch.tensor(sigma_min)) + t * torch.log(torch.tensor(sigma_max / sigma_min))
        sigma = torch.exp(log_sigma).to(x0.device)
        
        # Masking probability: 1 - exp(-sigma)
        # This is the "absorbing state" formulation
        mask_prob = 1.0 - torch.exp(-sigma)
        mask_prob = mask_prob.view(B, 1)
        
        # Generate mask
        should_mask = torch.rand(B, T, device=x0.device) < mask_prob
        
        # Apply mask
        xt = x0.clone()
        xt[should_mask] = mask_token_id
        
        return xt, should_mask

    def compute_loss(self, x0):
        B, T = x0.shape
        
        # Sample random timesteps uniformly in [0, 1]
        t = torch.rand(B, device=x0.device)
        
        # Forward process: add noise (mask tokens)
        xt, mask = self.q_sample(x0, t)
        
        # Predict original tokens
        logits = self(xt)  # (B, T, V)
        
        # Compute log probabilities of ground truth
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = torch.gather(log_probs, 2, x0.unsqueeze(-1)).squeeze(-1)
        
        # SUBS parameterization: simplified loss
        # Only compute on masked positions, weight by dsigma / (exp(sigma) - 1)
        
        # Compute sigma and its derivative
        log_sigma = torch.log(torch.tensor(sigma_min)) + t * torch.log(torch.tensor(sigma_max / sigma_min))
        sigma = torch.exp(log_sigma).to(x0.device)
        dsigma = sigma * torch.log(torch.tensor(sigma_max / sigma_min))
        
        # Weight for each sample in batch
        weights = dsigma / torch.expm1(sigma)  # dsigma / (exp(sigma) - 1)
        weights = weights.view(B, 1)
        
        # Compute loss: negative log probability, weighted
        loss = -target_log_probs * weights
        
        # Only average over masked positions
        num_masked = mask.sum()
        if num_masked > 0:
            loss = (loss * mask.float()).sum() / num_masked
        else:
            loss = loss.mean()  # Fallback
        
        return loss

    @torch.no_grad()
    def sample(self, num_samples=1, steps=1000, temperature=1.0):
        self.eval()
        
        # Start fully masked
        x = torch.full((num_samples, block_size), mask_token_id, dtype=torch.long, device=device)

        timesteps = torch.linspace(1.0, 0.0, steps + 1)
        
        for i in range(steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            
            if i % 100 == 0:
                n_masked = (x == mask_token_id).sum().item()
                print(f"  Step {i}/{steps}, masked: {n_masked}/{num_samples * block_size}")
            
            # Get predictions
            logits = self(x) / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample from predictions
            sampled = torch.multinomial(probs.view(-1, vocab_size), 1).view(num_samples, block_size)
            
            # Determine which positions to update
            # Compute mask probabilities for current and next timesteps
            sigma_cur = sigma_min * (sigma_max / sigma_min) ** t_cur
            sigma_next = sigma_min * (sigma_max / sigma_min) ** t_next if t_next > 0 else 0
            
            mask_prob_cur = 1.0 - torch.exp(torch.tensor(-sigma_cur))
            mask_prob_next = 1.0 - torch.exp(torch.tensor(-sigma_next))
            
            # Update masked positions with samples
            is_masked = (x == mask_token_id)
            x = torch.where(is_masked, sampled, x)
            
            # Remask for next iteration (except last step)
            if i < steps - 1 and mask_prob_next > 0:
                remask = torch.rand(num_samples, block_size, device=device) < mask_prob_next
                x = torch.where(remask, mask_token_id, x)
        
        self.train()
        return x

# --- Training ---

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(50):
            x = get_batch(split)
            loss = model.compute_loss(x)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

print("\n--- Training MDLM ---")
model = MDLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), weight_decay=0.01)

# Learning rate schedule with warmup
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay after warmup
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * coeff

# Early stopping
patience = 10
min_delta = 0.001
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

print("Starting training...\n")

for it in tqdm(range(max_iters), desc="Training"):
    # Update LR
    lr = get_lr(it)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Eval
    if it % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"\nStep {it}: train={losses['train']:.4f}, val={losses['val']:.4f}, lr={lr:.6f}")
        
        # Early stopping
        if losses['val'] < best_val_loss - min_delta:
            best_val_loss = losses['val']
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"✓ Best: {best_val_loss:.4f}")
            torch.save(model.state_dict(), './models/mdlm_best.pth')
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping at step {it}")
                if best_model_state:
                    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                break
        
        # Sample
        if it > 0 and it % 5000 == 0:
            print("\n--- Sample ---")
            sample = model.sample(num_samples=1, steps=100, temperature=0.9)[0]
            print(decode(sample.tolist())[:200])
    
    # Training step
    x = get_batch('train')
    optimizer.zero_grad(set_to_none=True)
    loss = model.compute_loss(x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

print("\n✓ Training complete!")
torch.save(model.state_dict(), './models/mdlm_final.pth')

# --- Generate Samples ---

print("\n" + "="*70)
print("FINAL SAMPLES (100 steps)")
print("="*70)

for i in range(3):
    print(f"\n--- Sample {i+1} ---")
    sample = model.sample(num_samples=1, steps=100, temperature=0.85)[0]
    print(decode(sample.tolist()))
    print("-"*70)