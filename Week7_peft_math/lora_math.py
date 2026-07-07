import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):

    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.base_layer.weight.requires_grad = False 
        self.rank = rank
        self.scaling = alpha / rank
        
        # Matrix A: Down-projects the input from 'in_features' to 'rank'
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        # Matrix B: Up-projects the rank back to 'out_features'
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        
        self.reset_parameters()

    def reset_parameters(self):
        # A is initialized with random noise (Kaiming uniform)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output

def simulate_8bit_quantization(tensor):
    abs_max = torch.max(torch.abs(tensor))
    scale = 127.0 / abs_max
    
    # Quantize to 8-bit integer space
    quantized_tensor = torch.round(tensor * scale).to(torch.int8)
    dequantized_tensor = (quantized_tensor.to(torch.float32) / scale)
    
    return quantized_tensor, dequantized_tensor

if __name__ == "__main__":
    print("--- Executing Week 7: LoRA & PEFT Math ---")
    IN_FEATURES = 4096
    OUT_FEATURES = 4096
    RANK = 8
    
    lora_layer = LoRALinear(in_features=IN_FEATURES, out_features=OUT_FEATURES, rank=RANK)
    base_params = IN_FEATURES * OUT_FEATURES
    lora_params = (IN_FEATURES * RANK) + (OUT_FEATURES * RANK)
    
    print(f"Base Matrix Parameters (Frozen): {base_params:,}")
    print(f"LoRA Matrix Parameters (Trainable): {lora_params:,}")
    print(f"Reduction: You are training only {((lora_params / base_params) * 100):.3f}% of the original parameters!")

    dummy_input = torch.randn(1, IN_FEATURES)
    output = lora_layer(dummy_input)
    print(f"\nForward Pass Successful. Output Shape: {output.shape}")

    print("\n--- Conceptual Quantization (QLoRA) ---")
    original_weights = lora_layer.base_layer.weight.data
    q_weights, dq_weights = simulate_8bit_quantization(original_weights)
    
    print(f"Original FP32 Tensor Memory: {original_weights.element_size() * original_weights.nelement() / (1024**2):.2f} MB")
    print(f"Quantized INT8 Tensor Memory: {q_weights.element_size() * q_weights.nelement() / (1024**2):.2f} MB")
