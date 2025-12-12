"""
Configuration classes for HAMT models
"""
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class HAMTConfig:
    """Configuration for Holographic Associative Memory Transformer"""
    
    # Model dimensions
    hidden_dim: int = 768
    hcm_dim: int = 4096  # Holographic Context Memory dimension
    num_layers: int = 12
    num_attention_heads: int = 12  # For hybrid models
    intermediate_dim: int = 3072
    
    # HCM settings
    num_slots: int = 8  # Multi-slot memory
    binding_type: Literal["elementwise", "circular_conv"] = "elementwise"
    use_hierarchical_memory: bool = False
    fast_memory_update_rate: float = 1.0
    slow_memory_update_rate: float = 0.1
    
    # Gating and normalization
    use_gating: bool = True
    decay_rate: float = 1e-4  # Passive forgetting
    normalization_type: Literal["rms", "layer", "none"] = "rms"
    
    # Retrieval
    retrieval_head_hidden_dim: Optional[int] = None  # Defaults to 2*hcm_dim
    retrieval_dropout: float = 0.1
    
    # Training
    max_position_embeddings: int = 2048
    vocab_size: int = 50257
    dropout: float = 0.1
    use_auxiliary_loss: bool = True
    aux_loss_weight: float = 0.05
    
    # TBPTT settings
    bptt_horizon: int = 256
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        if self.retrieval_head_hidden_dim is None:
            self.retrieval_head_hidden_dim = 2 * self.hcm_dim
        
        # Validate dimensions
        assert self.hidden_dim % self.num_attention_heads == 0, \
            "hidden_dim must be divisible by num_attention_heads"
        assert self.hcm_dim >= self.hidden_dim, \
            "hcm_dim should be >= hidden_dim for effective holographic encoding"
