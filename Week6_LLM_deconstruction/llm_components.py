import torch
import torch.nn as nn

# 1. The ADC Pipeline
class BasicTokenizerAndEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.dummy_vocab = {"<PAD>": 0, "learning": 1, "deep": 2, "is": 3, "awesome": 4}

    def encode(self, text):
        words = text.lower().split()
        return [self.dummy_vocab.get(word, 0) for word in words]

    def forward(self, input_ids):
        return self.token_embedding_table(input_ids)

#2. Production Normalization
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

# 3. Positional Upgrades
def apply_rotary_emb_dummy(q, k):
    B, T, C = q.shape
    position = torch.arange(T, dtype=torch.float).unsqueeze(1)
    theta = position * 0.1 # Arbitrary rotation scaling for demonstration
    cos_theta = torch.cos(theta).unsqueeze(0) 
    sin_theta = torch.sin(theta).unsqueeze(0) 

    q_rotated = q * cos_theta + torch.roll(q, shifts=1, dims=-1) * sin_theta
    k_rotated = k * cos_theta + torch.roll(k, shifts=1, dims=-1) * sin_theta
    
    return q_rotated, k_rotated

if __name__ == "__main__":
    print("--- Executing Week 6: LLM Component Deconstruction ---")
    
    # 1. Text to Vectors
    pipeline = BasicTokenizerAndEmbedding(vocab_size=10, embed_dim=16)
    raw_text = "learning deep is awesome"
    
    token_ids = torch.tensor([pipeline.encode(raw_text)])
    print(f"Raw Text: '{raw_text}'")
    print(f"Discrete Token IDs: {token_ids.tolist()}")
    
    embeddings = pipeline(token_ids)
    print(f"Vector Space Shape (B, T, C): {embeddings.shape}")
    
    # 2. RMSNorm Execution
    rmsnorm = RMSNorm(dim=16)
    normalized_embeddings = rmsnorm(embeddings)
    
    print("\nPre-Norm Vector Variance:", torch.var(embeddings[0, 0, :]).item())
    print("Post-RMSNorm Variance:", torch.var(normalized_embeddings[0, 0, :]).item())
    print("Notice how RMSNorm standardizes the signal scale without centering the mean.")
