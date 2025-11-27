import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import time
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import gc

# --- 1. Configuration ---
batch_size = 32
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MDLM hyperparameters
mdlm_n_embd = 512
mdlm_n_head = 8
mdlm_n_layer = 8
dropout = 0.1

# MDLM specific: continuous time formulation
sigma_min = 1e-4
sigma_max = 1.0

print(f"Using device: {device}")

# Tokenizer Setup
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

# --- 2. MDLM Model Definition ---

class MDLM(nn.Module):
    """
    Masked Diffusion Language Model with SUBS parameterization.
    Based on the official implementation from Sahoo et al. (NeurIPS 2024)
    """
    def __init__(self, n_embd=512, n_head=8, n_layer=8):
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
        
        # Tie weights
        self.head.weight = self.tok_emb.weight
        
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

    def forward(self, x):
        """
        x: (B, T) token indices
        Returns: (B, T, vocab_size) logits
        """
        B, T = x.shape
        
        tok_emb = self.tok_emb(x)
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_emb(pos)
        
        x = self.drop(tok_emb + pos_emb)
        x = self.transformer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

    @torch.no_grad()
    def sample(self, num_samples=1, steps=100, temperature=1.0, verbose=False):
        """
        Ancestral sampling (DDPM-style) for MDLM.
        Start from all masks, iteratively denoise.
        """
        self.eval()
        
        # Start fully masked
        x = torch.full((num_samples, block_size), mask_token_id, dtype=torch.long, device=device)
        
        # Timesteps: go from 1.0 (fully noisy) to 0.0 (clean)
        timesteps = torch.linspace(1.0, 0.0, steps + 1)
        
        for i in range(steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            
            if verbose and i % 20 == 0:
                n_masked = (x == mask_token_id).sum().item()
                print(f"  Step {i}/{steps}, masked: {n_masked}/{num_samples * block_size}")
            
            # Get predictions
            logits = self(x) / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample from predictions
            sampled = torch.multinomial(probs.view(-1, vocab_size), 1).view(num_samples, block_size)
            
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


# --- 3. Evaluation Helpers ---

def calculate_self_bleu(texts):
    """Calculate Self-BLEU score (lower = more diverse)"""
    if len(texts) <= 1:
        return 0.0
    total_bleu = 0.0
    chencherry = SmoothingFunction().method1
    for i in tqdm(range(len(texts)), desc="Self-BLEU"):
        candidate = texts[i].split()
        references = [texts[j].split() for j in range(len(texts)) if i != j]
        if not candidate or not any(references):
            continue
        score = corpus_bleu([references], [candidate], smoothing_function=chencherry)
        total_bleu += score
    return total_bleu / len(texts)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MASKED DIFFUSION LANGUAGE MODEL (MDLM) EVALUATION")
    print("="*70)
    
    # --- Load Model ---
    print("\n--- Loading MDLM Model ---")
    mdlm_model = MDLM(n_embd=mdlm_n_embd, n_head=mdlm_n_head, n_layer=mdlm_n_layer).to(device)
    
    try:
        # Try loading from best checkpoint first
        try:
            mdlm_model.load_state_dict(torch.load('./models/mdlm_best.pth', map_location=device))
            print("✓ Successfully loaded mdlm_best.pth")
        except FileNotFoundError:
            mdlm_model.load_state_dict(torch.load('./models/mdlm_final.pth', map_location=device))
            print("✓ Successfully loaded mdlm_final.pth")
        
        print(f"  MDLM Model: {mdlm_n_layer} layers, {mdlm_n_embd} dim, {mdlm_n_head} heads")
        total_params = sum(p.numel() for p in mdlm_model.parameters())
        print(f"  Total Parameters: {total_params/1e6:.2f}M")
    except FileNotFoundError:
        print("ERROR: No MDLM model found. Please train the model first.")
        exit()
    except RuntimeError as e:
        print(f"ERROR: Could not load MDLM model: {e}")
        exit()
    
    mdlm_model.eval()
    
    # Clear cache
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    # --- 1. Inference Time Comparison ---
    print("\n" + "="*70)
    print("1. INFERENCE TIME")
    print("="*70)
    num_inference_runs = 20
    sampling_steps = 100
    
    print(f"\nTesting MDLM Model ({num_inference_runs} runs, {block_size} tokens, {sampling_steps} steps)...")
    mdlm_times = []
    
    for _ in tqdm(range(num_inference_runs), desc="MDLM inference"):
        if device == 'cuda': 
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _ = mdlm_model.sample(num_samples=1, steps=sampling_steps)
        if device == 'cuda': 
            torch.cuda.synchronize()
        mdlm_times.append(time.time() - start_time)
        
        # Clear cache after each run
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    mdlm_avg_time = sum(mdlm_times) / len(mdlm_times)
    mdlm_speed = block_size / mdlm_avg_time
    
    print(f"\nMDLM Model Avg Time: {mdlm_avg_time:.4f} seconds for {block_size} tokens")
    print(f"MDLM Model Speed: {mdlm_speed:.2f} tokens/second")
    print(f"Sampling Steps Used: {sampling_steps}")
    
    # --- 2. Generation Quality ---
    print("\n" + "="*70)
    print("2. GENERATION QUALITY SAMPLES")
    print("="*70)
    
    print("\n--- MDLM Generated Samples ---")
    with torch.no_grad():
        sample_results = mdlm_model.sample(num_samples=3, steps=sampling_steps, temperature=0.85)
    
    for i in range(3):
        text = decode(sample_results[i].tolist())
        print(f"\nSample {i+1}:")
        print(text[:500])  # Limit output length
        print("-"*70)
    
    # Clear cache
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    # --- 3. Generation Diversity (Self-BLEU) ---
    print("\n" + "="*70)
    print("3. GENERATION DIVERSITY (Self-BLEU)")
    print("="*70)
    print("Lower Self-BLEU = More diverse generations")
    
    num_samples_for_bleu = 50
    print(f"\nGenerating {num_samples_for_bleu} samples from MDLM model...")
    
    # Generate in smaller batches to avoid OOM
    batch_gen_size = 10
    mdlm_texts = []
    
    for batch_idx in range(0, num_samples_for_bleu, batch_gen_size):
        current_batch_size = min(batch_gen_size, num_samples_for_bleu - batch_idx)
        
        with torch.no_grad():
            batch_results = mdlm_model.sample(num_samples=current_batch_size, steps=sampling_steps).tolist()
        
        mdlm_texts.extend([decode(l) for l in batch_results])
        
        # Clear cache after each batch
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"  Generated {len(mdlm_texts)}/{num_samples_for_bleu} samples")
    
    print("\nCalculating MDLM Self-BLEU...")
    mdlm_self_bleu = calculate_self_bleu(mdlm_texts)
    print(f"MDLM Model Self-BLEU: {mdlm_self_bleu:.4f}")
    
    # --- Summary ---
    print("\n" + "="*70)
    print("MDLM MODEL EVALUATION SUMMARY")
    print("="*70)
    print(f"Model Configuration: {mdlm_n_layer} layers, {mdlm_n_embd} dim, {mdlm_n_head} heads")
    print(f"Total Parameters:    {total_params/1e6:.2f}M")
    print(f"Sampling Steps:      {sampling_steps}")
    print(f"Inference Speed:     {mdlm_speed:.2f} tokens/sec")
    print(f"Self-BLEU Score:     {mdlm_self_bleu:.4f}")
    print("="*70)
    
    # Save results to file
    with open('./mdlm_eval_results.txt', 'w') as f:
        f.write("MDLM MODEL EVALUATION RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Model Configuration: {mdlm_n_layer} layers, {mdlm_n_embd} dim, {mdlm_n_head} heads\n")
        f.write(f"Total Parameters: {total_params/1e6:.2f}M\n")
        f.write(f"Sampling Steps: {sampling_steps}\n")
        f.write(f"Inference Speed: {mdlm_speed:.2f} tokens/sec\n")
        f.write(f"Average Time per {block_size} tokens: {mdlm_avg_time:.4f} seconds\n")
        f.write(f"Self-BLEU Score: {mdlm_self_bleu:.4f}\n")
        f.write("="*70 + "\n\n")
        f.write("Sample Generations:\n\n")
        for i in range(min(3, len(mdlm_texts))):
            f.write(f"Sample {i+1}:\n")
            f.write(mdlm_texts[i][:500] + "\n")
            f.write("-"*70 + "\n\n")
    
    print("\nResults saved to ./mdlm_eval_results.txt")