"""
Utility functions for HAMT
"""
import torch
import numpy as np
from typing import Dict, Any


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'total_millions': total_params / 1e6,
        'trainable_millions': trainable_params / 1e6
    }


def get_lr_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
):
    """Cosine learning rate schedule with warmup"""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def compute_flops_per_token(config, seq_len: int) -> Dict[str, float]:
    """
    Estimate FLOPs per token for HAMT vs standard Transformer.
    This is a rough approximation.
    """
    d_model = config.hidden_dim
    d_hcm = config.hcm_dim
    d_ff = config.intermediate_dim
    n_layers = config.num_layers
    n_slots = config.num_slots
    
    # Standard Transformer FLOPs (per layer, per token)
    # Self-attention: O(N^2 * d) for QKV computation + attention weights
    # For full sequence of length N, per token: O(N * d)
    attention_flops = 2 * seq_len * d_model * d_model  # QK^T + attention @ V
    ffn_flops = 2 * d_model * d_ff  # Two linear layers
    standard_per_layer = attention_flops + ffn_flops
    standard_total = n_layers * standard_per_layer
    
    # HAMT FLOPs (per layer, per token)
    # Item projection: d_model -> d_hcm
    item_proj_flops = 2 * d_model * d_hcm
    
    # Binding: O(d_hcm) for elementwise or O(d_hcm log d_hcm) for FFT
    binding_flops = d_hcm  # elementwise
    
    # Retrieval head: query -> unbinding key (2-layer MLP)
    retrieval_flops = 2 * d_model * 2 * d_hcm + 2 * 2 * d_hcm * d_hcm
    
    # Unbinding from n_slots: O(n_slots * d_hcm)
    unbinding_flops = n_slots * d_hcm
    
    # Memory update: O(n_slots * d_hcm)
    update_flops = n_slots * d_hcm
    
    # Output projection: (d_model + d_hcm) -> d_model
    output_proj_flops = 2 * (d_model + d_hcm) * d_model
    
    # FFN (same as standard)
    hamt_per_layer = (
        item_proj_flops + binding_flops + retrieval_flops + 
        unbinding_flops + update_flops + output_proj_flops + ffn_flops
    )
    hamt_total = n_layers * hamt_per_layer
    
    return {
        'standard_per_token': standard_total,
        'hamt_per_token': hamt_total,
        'reduction_ratio': standard_total / hamt_total,
        'standard_gflops': standard_total / 1e9,
        'hamt_gflops': hamt_total / 1e9
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary for logging"""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.4f}")
        else:
            formatted.append(f"{key}: {value}")
    return " | ".join(formatted)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
