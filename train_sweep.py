import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from optitrain_fp4.nvfp4 import NVFP4Linear
from optitrain_fp4.kernels.triton_attention import FP4FlashAttention
from optitrain_fp4.optimizer import Muon, MuonClip

class ToyTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int = 128, num_heads: int = 4, 
                 micro_block_size: int = 16, stochastic: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # QKV Projections simulating NVFP4
        self.q_proj = NVFP4Linear(hidden_size, hidden_size, bias=False, 
                                  micro_block_size=micro_block_size, stochastic=stochastic)
        self.k_proj = NVFP4Linear(hidden_size, hidden_size, bias=False, 
                                  micro_block_size=micro_block_size, stochastic=stochastic)
        self.v_proj = NVFP4Linear(hidden_size, hidden_size, bias=False, 
                                  micro_block_size=micro_block_size, stochastic=stochastic)
        
        # Tag parameter names so MuonClip can recognize QK projections
        self.q_proj.weight._name = "q_proj"
        self.k_proj.weight._name = "k_proj"
        self.v_proj.weight._name = "v_proj"
        
        # Out projection
        self.out_proj = NVFP4Linear(hidden_size, hidden_size, bias=False, 
                                    micro_block_size=micro_block_size, stochastic=stochastic)
        self.out_proj.weight._name = "out_proj"
        
        # MLP Layers
        self.mlp_in = NVFP4Linear(hidden_size, hidden_size * 4, bias=False, 
                                  micro_block_size=micro_block_size, stochastic=stochastic)
        self.mlp_out = NVFP4Linear(hidden_size * 4, hidden_size, bias=False, 
                                   micro_block_size=micro_block_size, stochastic=stochastic)
        self.mlp_in.weight._name = "mlp_in"
        self.mlp_out.weight._name = "mlp_out"
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> tuple:
        B, S, C = x.shape
        
        # Attention forward
        norm_x = self.norm1(x)
        
        # Compute Q, K, V
        q = self.q_proj(norm_x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Run Fused Attention Triton kernel simulation (falls back dynamically or executes)
        # For sweep testing simplicity, we can also compute the attention matrix explicitly
        # to monitor attention entropy
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Track Attention Entropy
        # H = -sum(p * log(p))
        with torch.no_grad():
            entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-12), dim=-1).mean().item()
            
        # Standard scaled attention output
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, C)
        
        x = x + self.out_proj(attn_out)
        
        # MLP Forward
        mlp_out = self.mlp_out(F.silu(self.mlp_in(self.norm2(x))))
        x = x + mlp_out
        
        return x, entropy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum_lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--qk_max_ratio", type=float, default=0.02)
    parser.add_argument("--micro_block_size", type=int, default=16)
    parser.add_argument("--stochastic_rounding", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--output_dir", type=str, default="./results/run_000")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    
    # Build Model
    model = ToyTransformerBlock(
        micro_block_size=args.micro_block_size, 
        stochastic=args.stochastic_rounding
    ).to(device)
    
    # Initialize Muon Optimizer with MuonClip Stabilizer
    stabilizer = MuonClip(qk_max_ratio=args.qk_max_ratio)
    optimizer = Muon(
        model.parameters(),
        lr=args.lr,
        momentum_lr=args.momentum_lr,
        beta=args.beta,
        clip_stabilizer=stabilizer
    )
    
    # Generate static dummy data for target loss minimization task
    torch.manual_seed(42)
    x_input = torch.randn(8, 32, 128, device=device) # B=8, S=32, C=128
    y_target = torch.randn(8, 32, 128, device=device)
    
    steps = 50
    diverged = False
    qk_entropies = []
    losses = []
    
    print("[*] Starting training loop...")
    for step in range(steps):
        optimizer.zero_grad()
        output, entropy = model(x_input)
        
        loss = F.mse_loss(output, y_target)
        loss.backward()
        
        # Check gradient magnitudes and loss value
        if torch.isnan(loss) or loss.item() > 100.0:
            print(f"[-] Diverged at step {step}: Loss is {loss.item()}")
            diverged = True
            break
            
        # Entropy check: values below 0.1 typically signal QK collapse/sparsity issues
        if entropy < 0.05:
            print(f"[-] Attention entropy collapsed at step {step}: Entropy {entropy:.4f}")
            diverged = True
            break
            
        optimizer.step()
        
        qk_entropies.append(entropy)
        losses.append(loss.item())
        
        if step % 10 == 0:
            print(f"    Step {step:02d} | Loss: {loss.item():.4f} | Attn Entropy: {entropy:.4f}")
            
    final_stats = {
        "val_loss": losses[-1] if not diverged else 999.0,
        "qk_entropy": sum(qk_entropies) / len(qk_entropies) if qk_entropies else 0.0,
        "diverged": diverged,
        "step_count": len(losses)
    }
    
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(final_stats, f, indent=4)
        
    print(f"[+] Saved execution stats to {stats_path}")

if __name__ == "__main__":
    main()
