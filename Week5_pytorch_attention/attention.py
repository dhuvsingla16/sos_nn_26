import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return torch.sigmoid(self.output(x))

class CausalSelfAttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, max_seq_len=1024):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) 
        k = self.key(x)  
        v = self.value(x) 

        wei = q @ k.transpose(-2, -1) 
        
        #  Scale by the square root of the head dimension
        wei = wei * (k.shape[-1] ** -0.5)
        
        #  Apply Causal Mask 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        #  Softmax to convert affinities into a clean probability distribution
        wei = F.softmax(wei, dim=-1) 
        
        #  Aggregate the Values based on the attention distribution
        out = wei @ v 
        
        return out, wei

if __name__ == "__main__":
    print("--- Executing Week 5: Math of Self-Attention ---")
    
    # arbitrary network dimensions
    batch_size = 4
    sequence_length = 8 
    embed_dim = 32      
    head_size = 16      
    attention_head = CausalSelfAttentionHead(n_embd=embed_dim, head_size=head_size)
    dummy_input = torch.randn(batch_size, sequence_length, embed_dim)
    print(f"Input Shape (B, T, C): {dummy_input.shape}")
    
    output, attention_map = attention_head(dummy_input)
    print(f"\nOutput Shape (B, T, head_size): {output.shape}")
    print(f"Attention Map Shape (B, T, T): {attention_map.shape}")
   
    print("\nAttention Distribution for Batch 0 (Notice the lower-triangular structure):")
    print(torch.round(attention_map[0] * 100) / 100)
