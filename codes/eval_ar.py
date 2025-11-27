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
gen_tokens = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# AR hyperparameters
ar_n_embd = 512
ar_n_head = 8
ar_n_layer = 8
dropout = 0.1

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

print(f"Vocab size: {vocab_size}")

# --- 2. AR Model Definition ---

class AutoRegressiveModel(nn.Module):
    """Standard autoregressive transformer"""
    def __init__(self, n_embd=512, n_head=8, n_layer=8):
        super().__init__()
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
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
        
        # Tie weights
        self.lm_head.weight = self.token_embedding_table.weight

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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:].to(device)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


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
    print("AUTOREGRESSIVE MODEL EVALUATION")
    print("="*70)
    
    # --- Load Model ---
    print("\n--- Loading AR Model ---")
    ar_model = AutoRegressiveModel(n_embd=ar_n_embd, n_head=ar_n_head, n_layer=ar_n_layer).to(device)
    
    try:
        ar_model.load_state_dict(torch.load('./models/ar_model.pth', map_location=device))
        print("✓ Successfully loaded ar_model.pth")
        print(f"  AR Model: {ar_n_layer} layers, {ar_n_embd} dim, {ar_n_head} heads")
        total_params = sum(p.numel() for p in ar_model.parameters())
        print(f"  Total Parameters: {total_params/1e6:.2f}M")
    except FileNotFoundError:
        print("ERROR: ar_model.pth not found. Please train the model first.")
        exit()
    except RuntimeError as e:
        print(f"ERROR: Could not load ar_model.pth: {e}")
        exit()
    
    ar_model.eval()
    
    # Clear cache
    if device == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
    
    # --- 1. Inference Time Comparison ---
    print("\n" + "="*70)
    print("1. INFERENCE TIME")
    print("="*70)
    num_inference_runs = 20
    
    print(f"\nTesting AR Model ({num_inference_runs} runs, {gen_tokens} tokens)...")
    ar_times = []
    start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    for _ in tqdm(range(num_inference_runs), desc="AR inference"):
        if device == 'cuda': 
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _ = ar_model.generate(start_context, max_new_tokens=gen_tokens)
        if device == 'cuda': 
            torch.cuda.synchronize()
        ar_times.append(time.time() - start_time)
        
        # Clear cache after each run
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    ar_avg_time = sum(ar_times) / len(ar_times)
    ar_speed = gen_tokens / ar_avg_time
    
    print(f"\nAR Model Avg Time: {ar_avg_time:.4f} seconds for {gen_tokens} tokens")
    print(f"AR Model Speed: {ar_speed:.2f} tokens/second")
    
    # --- 2. Generation Quality ---
    print("\n" + "="*70)
    print("2. GENERATION QUALITY SAMPLES")
    print("="*70)
    
    print("\n--- AR Generated Samples ---")
    start_context = torch.zeros((3, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        ar_results = ar_model.generate(start_context, max_new_tokens=block_size)
    
    for i in range(3):
        text = decode(ar_results[i].tolist())
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
    print(f"\nGenerating {num_samples_for_bleu} samples from AR model...")
    
    # Generate in smaller batches to avoid OOM
    batch_gen_size = 10
    ar_texts = []
    
    for batch_idx in range(0, num_samples_for_bleu, batch_gen_size):
        current_batch_size = min(batch_gen_size, num_samples_for_bleu - batch_idx)
        start_context_batch = torch.zeros((current_batch_size, 1), dtype=torch.long, device=device)
        
        with torch.no_grad():
            batch_results = ar_model.generate(start_context_batch, max_new_tokens=100).tolist()
        
        ar_texts.extend([decode(l) for l in batch_results])
        
        # Clear cache after each batch
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        print(f"  Generated {len(ar_texts)}/{num_samples_for_bleu} samples")
    
    print("\nCalculating AR Self-BLEU...")
    ar_self_bleu = calculate_self_bleu(ar_texts)
    print(f"AR Model Self-BLEU: {ar_self_bleu:.4f}")
    
    # --- Summary ---
    print("\n" + "="*70)
    print("AR MODEL EVALUATION SUMMARY")
    print("="*70)
    print(f"Model Configuration: {ar_n_layer} layers, {ar_n_embd} dim, {ar_n_head} heads")
    print(f"Total Parameters:    {total_params/1e6:.2f}M")
    print(f"Inference Speed:     {ar_speed:.2f} tokens/sec")
    print(f"Self-BLEU Score:     {ar_self_bleu:.4f}")
    print("="*70)
    
    # Save results to file
    with open('./ar_eval_results.txt', 'w') as f:
        f.write("AR MODEL EVALUATION RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Model Configuration: {ar_n_layer} layers, {ar_n_embd} dim, {ar_n_head} heads\n")
        f.write(f"Total Parameters: {total_params/1e6:.2f}M\n")
        f.write(f"Inference Speed: {ar_speed:.2f} tokens/sec\n")
        f.write(f"Average Time per {gen_tokens} tokens: {ar_avg_time:.4f} seconds\n")
        f.write(f"Self-BLEU Score: {ar_self_bleu:.4f}\n")
        f.write("="*70 + "\n\n")
        f.write("Sample Generations:\n\n")
        for i in range(min(3, len(ar_texts))):
            f.write(f"Sample {i+1}:\n")
            f.write(ar_texts[i][:500] + "\n")
            f.write("-"*70 + "\n\n")
    
    print("\nResults saved to ./ar_eval_results.txt")