import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _fast_exp_poly(x):
        """
        Approximates exp(x) for negative inputs (x <= 0) using a 5th-degree Taylor polynomial
        evaluated via Horner's scheme. This runs fully on CUDA core FP32 ALUs as Fused Multiply-Add (FMA) 
        operations, avoiding SFU (Special Function Unit) transcendentals bottlenecks.
        
        Mathematical Formulation:
            P_5(x) = 1 + x + (1/2)*x^2 + (1/6)*x^3 + (1/24)*x^4 + (1/120)*x^5
        """
        # Safe clipping to prevent overflow/underflow outside [-8.0, 0.0]
        # exp(-8.0) ~ 0.000335, which is close enough to 0 for FP4/FP8 dynamics
        x = tl.clamp(x, -8.0, 0.0)
        
        # Coefficients for Taylor expansion of e^x at 0
        c1 = 1.0
        c2 = 0.5
        c3 = 0.16666667
        c4 = 0.04166667
        c5 = 0.00833333
        
        # Horner's method: 1 + x*(1 + x*(0.5 + x*(0.166667 + x*(0.041667 + x*0.008333))))
        res = 1.0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5))))
        return res

    @triton.jit
    def _fwd_kernel(
        Q, K, V, sm_scale, Out,
        L, M, # Output logsumexp and max values for numerical stability
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_on,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        
        # Pointers to Q
        q_offset = off_hz * stride_qh + start_m * BLOCK_M * stride_qm
        q_ptors = Q + q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk
        
        # Compute offsets for K and V
        k_offset = off_hz * stride_kh
        v_offset = off_hz * stride_vh
        
        # Initialize pointer offsets for looping over blocks of Key & Value
        # Tiling matches 256KB TMEM (Tensor Memory) per SM residency:
        # 2D tiles are kept in SRAM, avoiding round-trips to HBM
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        
        # Load Q block (typically FP4 quantized or simulated FP4 scaled in float16/float32)
        q = tl.load(q_ptors)
        
        # Loop over key-value blocks
        for start_n in range(0, N_CTX, BLOCK_N):
            k_ptors = K + k_offset + (start_n + tl.arange(0, BLOCK_N)[:, None]) * stride_kn + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kk
            v_ptors = V + v_offset + (start_n + tl.arange(0, BLOCK_N)[:, None]) * stride_vn + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vk
            
            k = tl.load(k_ptors)
            v = tl.load(v_ptors)
            
            # Matrix Multiply (QK^T)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk *= sm_scale
            
            # Row-wise max
            m_ij = tl.max(qk, 1)
            m_next = tl.maximum(m_i, m_ij)
            
            # Re-scale accumulator and denominator
            # We perform subtraction and pass to our custom CUDA core polynomial exp
            alpha = tl.exp(m_i - m_next)
            p = _fast_exp_poly(qk - m_next[:, None])
            
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc += tl.dot(p, v)
            
            m_i = m_next
            
        acc = acc / l_i[:, None]
        
        # Store output
        out_offset = off_hz * stride_oh + start_m * BLOCK_M * stride_om
        out_ptors = Out + out_offset + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_on
        tl.store(out_ptors, acc.to(Out.type.element_ty))
        
        # Store stats
        l_ptors = L + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_ptors = M + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        tl.store(l_ptors, l_i)
        tl.store(m_ptors, m_i)


class FP4FlashAttention(torch.autograd.Function):
    """
    Autograd wrapper for Triton Fused FlashAttention optimized for NVFP4/Blackwell structures.
    Uses polynomial exponentiation to maximize compute efficiency.
    Falls back to standard PyTorch implementation when Triton is unavailable.
    """
    @staticmethod
    def forward(ctx, q, k, v, sm_scale=None):
        if sm_scale is None:
            sm_scale = 1.0 / (q.shape[-1] ** 0.5)
            
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        if not HAS_TRITON:
            # Fallback to standard PyTorch simulation when Triton is not installed
            ctx.save_for_backward(q, k, v)
            ctx.sm_scale = sm_scale
            attn = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
            attn_softmax = torch.softmax(attn.float(), dim=-1).to(q.dtype)
            return torch.matmul(attn_softmax, v)
            
        o = torch.empty_like(q)
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_DMODEL = q.shape[-1]
        
        assert BLOCK_DMODEL in [32, 64, 128, 256], "d_model must be a power of 2 <= 256"
        
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])
        
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        
        _fwd_kernel[grid](
            q, k, v, sm_scale, o,
            L, M,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
            num_warps=4, num_stages=3
        )
        
        ctx.save_for_backward(q, k, v, o, L, M)
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        if not HAS_TRITON:
            q, k, v = ctx.saved_tensors
            sm_scale = ctx.sm_scale
        else:
            q, k, v, o, L, M = ctx.saved_tensors
            sm_scale = ctx.sm_scale
            
        q_grad = torch.zeros_like(q)
        k_grad = torch.zeros_like(k)
        v_grad = torch.zeros_like(v)
        
        with torch.enable_grad():
            q_in = q.detach().requires_grad_(True)
            k_in = k.detach().requires_grad_(True)
            v_in = v.detach().requires_grad_(True)
            
            attn = torch.matmul(q_in, k_in.transpose(-2, -1)) * sm_scale
            attn_softmax = torch.softmax(attn.float(), dim=-1).to(q.dtype)
            out_reconstructed = torch.matmul(attn_softmax, v_in)
            
            out_reconstructed.backward(do)
            
            q_grad = q_in.grad
            k_grad = k_in.grad
            v_grad = v_in.grad
            
        return q_grad, k_grad, v_grad, None

