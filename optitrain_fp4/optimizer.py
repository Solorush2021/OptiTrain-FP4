import torch
from torch.optim import Optimizer
from typing import Dict, List, Tuple, Union, Callable

def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Orthogonalizes matrix G using Newton-Schulz iteration:
    X_{k+1} = 0.5 * X_k * (3 * I - X_k^T * X_k)
    For M x N matrices, applies iteration to the smaller dimension to optimize compute.
    """
    M, N = G.shape
    X = G.to(torch.float32)
    
    # Convergence condition: spectral norm of X_0 < sqrt(3).
    # Normalizing by Frobenius norm ensures all singular values are <= 1.0,
    # placing them safely within the convergence basin.
    X = X / (X.norm() + eps)
    
    if M < N:
        # Rows are orthonormal: X @ X.T = I (M x M)
        for _ in range(steps):
            XXT = X @ X.t()
            eye = torch.eye(M, device=X.device, dtype=X.dtype)
            X = 0.5 * (3.0 * eye - XXT) @ X
    else:
        # Columns are orthonormal: X.T @ X = I (N x N)
        for _ in range(steps):
            XTX = X.t() @ X
            eye = torch.eye(N, device=X.device, dtype=X.dtype)
            X = 0.5 * X @ (3.0 * eye - XTX)
            
    return X.to(G.dtype)

class MuonClip:
    """
    Stabilizer to bound updates to prevent QK collapse and divergence in FP4 training.
    """
    def __init__(self, max_ratio: float = 0.05, qk_max_ratio: float = 0.02, qk_names: List[str] = None):
        """
        Args:
            max_ratio: Maximum Frobenius norm ratio of the update to the weight (||update||_F / ||weight||_F).
            qk_max_ratio: Stricter ratio applied to Q/K weight matrices to prevent self-attention entropy collapse.
            qk_names: Substrings in parameter names to identify QK projections (e.g., 'q_proj', 'k_proj', 'attn.q', 'attn.k').
        """
        self.max_ratio = max_ratio
        self.qk_max_ratio = qk_max_ratio
        self.qk_names = qk_names or ["q_proj", "k_proj", "attn.q", "attn.k", "query", "key"]

    def __call__(self, name: str, weight: torch.Tensor, update: torch.Tensor, lr: float) -> torch.Tensor:
        """
        Stabilizes and clips the update step.
        """
        # Determine if this parameter belongs to QK projections
        is_qk = any(qk_name in name.lower() for qk_name in self.qk_names)
        ratio_limit = self.qk_max_ratio if is_qk else self.max_ratio
        
        weight_norm = torch.norm(weight)
        update_step = update * lr
        update_norm = torch.norm(update_step)
        
        # If weight norm is 0, do not clip
        if weight_norm > 0:
            current_ratio = update_norm / weight_norm
            if current_ratio > ratio_limit:
                scale = ratio_limit / (current_ratio + 1e-8)
                update = update * scale
                
        return update

class Muon(Optimizer):
    """
    Muon: Newton-Schulz Orthogonalized Momentum Optimizer from Kimi K2.
    Applies Newton-Schulz orthogonalization to momentum of 2D weight matrices.
    Falls back to AdamW/SGD-style momentum for 1D tensors (biases, layernorm, embeddings).
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum_lr: float = 1e-4, # For 1D/fallback params
        beta: float = 0.95,
        steps: int = 5,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        clip_stabilizer: MuonClip = None
    ):
        """
        Args:
            params: Iterable of parameters to optimize.
            lr: Learning rate for 2D parameters (matrix updates).
            momentum_lr: Learning rate for 1D/fallback parameters.
            beta: Momentum decay rate.
            steps: Number of Newton-Schulz iterations.
            eps: Epsilon for numerical stability.
            weight_decay: Weight decay factor.
            clip_stabilizer: MuonClip stabilizer instance.
        """
        defaults = dict(
            lr=lr,
            momentum_lr=momentum_lr,
            beta=beta,
            steps=steps,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
        self.clip_stabilizer = clip_stabilizer or MuonClip()
        
    @torch.no_grad()
    def step(self, closure: Callable = None) -> Union[float, None]:
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            lr = group['lr']
            momentum_lr = group['momentum_lr']
            beta = group['beta']
            steps = group['steps']
            eps = group['eps']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                # Fetch parameter name if registered, else use default string
                param_name = getattr(p, '_name', 'parameter')
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)
                    # For 1D parameters, we track variance for AdamW fallback
                    if p.ndim < 2:
                        state['variance'] = torch.zeros_like(p.data)
                        
                state['step'] += 1
                momentum = state['momentum']
                
                # 1. Update momentum: m_t = beta * m_{t-1} + (1 - beta) * grad
                momentum.mul_(beta).add_(grad, alpha=1.0 - beta)
                
                # 2. Check if we apply Muon or fallback
                if p.ndim >= 2:
                    # Apply weight decay
                    if wd != 0:
                        p.data.mul_(1.0 - lr * wd)
                        
                    # Orthogonalize momentum using Newton-Schulz
                    # To perform orthogonalization efficiently, reshape to 2D
                    orig_shape = p.shape
                    momentum_2d = momentum.view(orig_shape[0], -1)
                    
                    orthogonal_momentum = newton_schulz_orthogonalize(momentum_2d, steps=steps, eps=eps)
                    orthogonal_momentum = orthogonal_momentum.view(orig_shape)
                    
                    # 3. Apply MuonClip stabilizer to the update
                    stabilized_momentum = self.clip_stabilizer(param_name, p.data, orthogonal_momentum, lr)
                    
                    # 4. Apply update
                    p.data.add_(stabilized_momentum, alpha=-lr)
                else:
                    # Fallback to AdamW update for 1D tensors (e.g. bias, scale, embeddings)
                    if wd != 0:
                        p.data.mul_(1.0 - momentum_lr * wd)
                        
                    variance = state['variance']
                    # Update variance: v_t = 0.999 * v_{t-1} + 0.001 * grad^2
                    variance.mul_(0.999).addcmul_(grad, grad, value=0.001)
                    
                    # Bias corrections
                    bias_correction1 = 1.0 - beta ** state['step']
                    bias_correction2 = 1.0 - 0.999 ** state['step']
                    
                    step_size = momentum_lr * (bias_correction2 ** 0.5) / bias_correction1
                    denom = (variance.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                    
                    p.data.addcdiv_(momentum, denom, value=-step_size)
                    
        return loss
