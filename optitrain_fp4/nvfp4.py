import torch
import torch.nn as nn
from typing import Tuple

# The standard positive representable values of OCP E2M1 FP4 format:
# s (1 bit), e (2 bits), m (1 bit)
# 000 -> 0.0
# 001 -> 0.5
# 010 -> 1.0
# 011 -> 1.5
# 100 -> 2.0
# 101 -> 3.0
# 110 -> 4.0
# 111 -> 6.0
FP4_GRID_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_GRID = sorted(list(set([-x for x in FP4_GRID_POS] + FP4_GRID_POS)))

class NVFP4Quantizer:
    """
    Simulates NVIDIA FP4 (E2M1) quantization with dual-level micro-block scaling 
    and stochastic rounding. Supports PyTorch autograd via Straight-Through Estimator (STE).
    """
    
    def __init__(self, micro_block_size: int = 16, macro_block_size: int = 4):
        """
        Args:
            micro_block_size: Number of elements sharing a micro-scale factor (e.g., 16).
            macro_block_size: Number of micro-blocks grouped to share a macro-scale factor (e.g., 4).
        """
        self.micro_block_size = micro_block_size
        self.macro_block_size = macro_block_size
        
        # Register FP4 grid tensor
        self.grid = torch.tensor(FP4_GRID, dtype=torch.float32)
        self.max_val = 6.0  # Max value in FP4 grid

    def quantize_and_scale(self, x: torch.Tensor, stochastic: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies dual-level microscaling and FP4 quantization to input tensor x.
        
        Returns:
            quantized_x: The quantized tensor in original scale.
            macro_scales: Macro-block scale factors.
            micro_scales_quant: Quantized micro-block scale factors.
        """
        device = x.device
        dtype = x.dtype
        original_shape = x.shape
        
        # 1. Flatten and pad if necessary to align with micro_block_size
        num_elements = x.numel()
        pad_size = (self.micro_block_size - (num_elements % self.micro_block_size)) % self.micro_block_size
        if pad_size > 0:
            x_padded = torch.cat([x.view(-1), torch.zeros(pad_size, device=device, dtype=dtype)])
        else:
            x_padded = x.view(-1)
            
        total_micro_blocks = x_padded.numel() // self.micro_block_size
        x_blocked = x_padded.view(total_micro_blocks, self.micro_block_size)
        
        # 2. Compute micro-block scale factors: S_0 = max(|x_block|) / max_val
        micro_scales = x_blocked.abs().max(dim=1, keepdim=True)[0] / self.max_val
        micro_scales = torch.clamp(micro_scales, min=1e-12)
        
        # 3. Dual-level grouping: group micro_scales into macro-blocks
        # S_0 scale elements: total_micro_blocks. We group them by macro_block_size.
        pad_macro_size = (self.macro_block_size - (total_micro_blocks % self.macro_block_size)) % self.macro_block_size
        if pad_macro_size > 0:
            micro_scales_padded = torch.cat([micro_scales.view(-1), torch.full((pad_macro_size,), 1e-12, device=device, dtype=dtype)])
        else:
            micro_scales_padded = micro_scales.view(-1)
            
        total_macro_blocks = micro_scales_padded.numel() // self.macro_block_size
        micro_scales_blocked = micro_scales_padded.view(total_macro_blocks, self.macro_block_size)
        
        # Macro scale factor: S_1 = max(micro_scales_in_macro_block)
        macro_scales = micro_scales_blocked.max(dim=1, keepdim=True)[0]
        macro_scales = torch.clamp(macro_scales, min=1e-12)
        
        # Quantize micro_scales to powers of 2 relative to macro_scales (conceptually matching MX formats)
        # Ratio = micro_scales_blocked / macro_scales (values in [0, 1])
        ratios = micro_scales_blocked / macro_scales
        # We quantize the ratios to nearest powers of 2: 2^0, 2^-1, 2^-2, ..., 2^-7
        # Log2 of ratios clipped to [-7, 0]
        ratios_log2 = torch.clamp(torch.round(torch.log2(ratios)), min=-7.0, max=0.0)
        micro_scales_quant_blocked = torch.pow(2.0, ratios_log2) * macro_scales
        
        # Unroll scales back to elements
        micro_scales_quant = micro_scales_quant_blocked.view(-1)[:total_micro_blocks].view(total_micro_blocks, 1)
        
        # 4. Scale inputs for FP4 quantization
        # x_scaled = x_blocked / micro_scales_quant
        x_scaled = x_blocked / micro_scales_quant
        
        # 5. Quantize to FP4 grid [-6.0, 6.0]
        grid = self.grid.to(device)
        
        # Quantization search & rounding
        x_scaled_clipped = torch.clamp(x_scaled, min=grid[0], max=grid[-1])
        
        if stochastic:
            # Stochastic Rounding
            # Find the bucket for each scaled value
            # Since grid is small and sorted, we can search using buckets
            # We can use bucketization or direct comparisons
            # Let's use searchsorted to find right indices
            indices = torch.searchsorted(grid, x_scaled_clipped.contiguous())
            # Clamp indices so that we can fetch indices and indices-1
            idx_right = torch.clamp(indices, min=1, max=len(grid)-1)
            idx_left = idx_right - 1
            
            val_left = grid[idx_left]
            val_right = grid[idx_right]
            
            # Probability of rounding to right value
            denom = val_right - val_left
            # If denom is 0, probability is 0
            prob = torch.where(denom > 0, (x_scaled_clipped - val_left) / denom, torch.zeros_like(denom))
            
            # Uniform noise
            noise = torch.rand_like(x_scaled_clipped)
            mask = noise < prob
            
            x_quant_scaled = torch.where(mask, val_right, val_left)
        else:
            # Nearest Neighbor Rounding
            indices = torch.searchsorted(grid, x_scaled_clipped.contiguous())
            idx_right = torch.clamp(indices, min=1, max=len(grid)-1)
            idx_left = idx_right - 1
            
            val_left = grid[idx_left]
            val_right = grid[idx_right]
            
            mask = (x_scaled_clipped - val_left) > (val_right - x_scaled_clipped)
            x_quant_scaled = torch.where(mask, val_right, val_left)
            
        # Re-scale back
        x_quant_blocked = x_quant_scaled * micro_scales_quant
        
        # Reshape to original size
        if pad_size > 0:
            x_quant = x_quant_blocked.view(-1)[:num_elements].view(original_shape)
        else:
            x_quant = x_quant_blocked.view(original_shape)
            
        return x_quant, macro_scales, micro_scales_quant_blocked

def get_hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Constructs a normalized Walsh-Hadamard matrix of size n x n (n must be a power of 2).
    """
    import math
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device, dtype=dtype)
    current_size = 2
    while current_size < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
        current_size *= 2
    if n == 1:
        return torch.ones((1, 1), device=device, dtype=dtype)
    return H / math.sqrt(n)

def apply_rht(grad_output_flat: torch.Tensor, input_flat: torch.Tensor, quantizer: NVFP4Quantizer = None, stochastic: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the Random Hadamard Transform along the contracting dimension (rows)
    of both grad_output_flat [M, C_out] and input_flat [M, C_in].
    Then optionally quantizes the resulting transformed matrices using NVFP4Quantizer.
    
    This disperses outliers across the sequence/token/batch dimension so that they do
    not get clipped during FP4 quantization, while preserving the exact inner product.
    """
    import math
    M = grad_output_flat.shape[0]
    # Find next power of 2 for Hadamard matrix
    N = 2 ** int(math.ceil(math.log2(M)))
    
    # Pad both matrices to size N along dimension 0
    if N > M:
        pad_grad = torch.zeros(N - M, grad_output_flat.shape[1], device=grad_output_flat.device, dtype=grad_output_flat.dtype)
        pad_input = torch.zeros(N - M, input_flat.shape[1], device=input_flat.device, dtype=input_flat.dtype)
        grad_padded = torch.cat([grad_output_flat, pad_grad], dim=0)
        input_padded = torch.cat([input_flat, pad_input], dim=0)
    else:
        grad_padded = grad_output_flat
        input_padded = input_flat
        
    # Generate random sign vector of size N
    # Fixed/cached relative to current execution context, but stochastic over steps
    sign_vector = torch.randint(0, 2, (N, 1), device=grad_output_flat.device, dtype=grad_output_flat.dtype) * 2.0 - 1.0
    
    # Apply sign vector (element-wise multiplication along columns)
    grad_signed = grad_padded * sign_vector
    input_signed = input_padded * sign_vector
    
    # Get Hadamard matrix
    H = get_hadamard_matrix(N, grad_output_flat.device, grad_output_flat.dtype)
    
    # Transform (H @ signed_matrix)
    grad_trans = H @ grad_signed
    input_trans = H @ input_signed
    
    # Quantize to FP4 if quantizer is provided
    if quantizer is not None:
        grad_trans_q, _, _ = quantizer.quantize_and_scale(grad_trans, stochastic=stochastic)
        input_trans_q, _, _ = quantizer.quantize_and_scale(input_trans, stochastic=stochastic)
        return grad_trans_q, input_trans_q
        
    return grad_trans, input_trans

class NVFP4LinearFunction(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for NVFP4 Linear Layer Forward/Backward passes.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, 
                quantizer: NVFP4Quantizer = None, stochastic: bool = True) -> torch.Tensor:
        ctx.save_for_backward(input, weight, bias)
        ctx.quantizer = quantizer
        ctx.stochastic = stochastic
        
        # Quantize weights and activations to FP4 during forward pass
        if quantizer is not None:
            input_q, _, _ = quantizer.quantize_and_scale(input, stochastic=stochastic)
            weight_q, _, _ = quantizer.quantize_and_scale(weight, stochastic=stochastic)
        else:
            input_q = input
            weight_q = weight
            
        output = torch.nn.functional.linear(input_q, weight_q, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        input, weight, bias = ctx.saved_tensors
        quantizer = ctx.quantizer
        stochastic = ctx.stochastic
        
        # Gradients with respect to input, weight, bias
        # Backprop through linear is standard, using Straight-Through Estimator (STE)
        grad_input = grad_output.matmul(weight)
        
        # Reshape grad_output and input to compute weight gradient
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        input_flat = input.reshape(-1, input.shape[-1])
        
        # Apply Random Hadamard Transform (RHT) for outlier dispersion in FP4 weight-grad GEMM
        grad_output_trans, input_trans = apply_rht(grad_output_flat, input_flat, quantizer, stochastic)
        grad_weight = grad_output_trans.t().matmul(input_trans)
        
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output_flat.sum(dim=0)
            
        return grad_input, grad_weight, grad_bias, None, None

class NVFP4Linear(nn.Module):
    """
    Linear layer simulating NVFP4 precision training with dual-level block scaling.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 micro_block_size: int = 16, macro_block_size: int = 4, stochastic: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.stochastic = stochastic
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.quantizer = NVFP4Quantizer(micro_block_size=micro_block_size, macro_block_size=macro_block_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return NVFP4LinearFunction.apply(input, self.weight, self.bias, self.quantizer, self.stochastic)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, stochastic={self.stochastic}'
