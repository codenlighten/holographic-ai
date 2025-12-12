"""
Holographic memory operations: binding, unbinding, and superposition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HolographicMemory(nn.Module):
    """
    Manages the Holographic Context Memory (HCM) with binding and unbinding operations.
    Implements both elementwise product and circular convolution binding.
    """
    
    def __init__(
        self,
        hcm_dim: int,
        num_slots: int = 8,
        binding_type: str = "elementwise",
        decay_rate: float = 1e-4,
        normalization: str = "rms"
    ):
        super().__init__()
        self.hcm_dim = hcm_dim
        self.num_slots = num_slots
        self.binding_type = binding_type
        self.decay_rate = decay_rate
        self.normalization = normalization
        
        # Learnable scaling factor for normalization
        self.scale = nn.Parameter(torch.ones(1))
        
    def generate_positional_keys(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate high-dimensional positional keys.
        For elementwise binding, use bipolar {-1, +1} vectors.
        For circular convolution, use random phase unitary vectors.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensors on
            
        Returns:
            Positional keys of shape [seq_len, hcm_dim]
        """
        if self.binding_type == "elementwise":
            # Bipolar random vectors
            keys = torch.randint(0, 2, (seq_len, self.hcm_dim), device=device) * 2 - 1
            return keys.float()
        else:  # circular_conv
            # Random phase unitary vectors (implemented in real domain for simplicity)
            keys = torch.randn(seq_len, self.hcm_dim, device=device)
            keys = F.normalize(keys, p=2, dim=-1)
            return keys
    
    def bind(self, item: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """
        Bind item vector with positional key.
        
        Args:
            item: Item vector [batch, seq_len, hcm_dim] or [batch, hcm_dim]
            position: Position key [seq_len, hcm_dim] or [hcm_dim]
            
        Returns:
            Bound vector of same shape as item
        """
        if self.binding_type == "elementwise":
            return item * position
        else:  # circular_conv
            return self._circular_convolution(item, position)
    
    def unbind(self, memory: torch.Tensor, unbinding_key: torch.Tensor) -> torch.Tensor:
        """
        Unbind/retrieve from memory using unbinding key.
        
        Args:
            memory: Memory state [batch, num_slots, hcm_dim] or [batch, hcm_dim]
            unbinding_key: Unbinding key [hcm_dim] or [batch, hcm_dim]
            
        Returns:
            Retrieved context vector
        """
        if self.binding_type == "elementwise":
            # For bipolar keys: p^{-1} = p, so unbinding is just multiplication
            # Handle different dimensionalities
            if memory.dim() == 3:  # [batch, num_slots, hcm_dim]
                if unbinding_key.dim() == 1:  # [hcm_dim]
                    # Broadcast to [batch, num_slots, hcm_dim]
                    return memory * unbinding_key.view(1, 1, -1)
                elif unbinding_key.dim() == 2:  # [batch, hcm_dim]
                    return memory * unbinding_key.unsqueeze(1)
            else:  # memory is [batch, hcm_dim]
                if unbinding_key.dim() == 1:  # [hcm_dim]
                    return memory * unbinding_key.view(1, -1)
                else:  # [batch, hcm_dim]
                    return memory * unbinding_key
        else:  # circular_conv
            return self._circular_correlation(memory, unbinding_key)
    
    def _circular_convolution(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Circular convolution via FFT"""
        # Move to complex domain
        a_fft = torch.fft.fft(a, dim=-1)
        b_fft = torch.fft.fft(b, dim=-1)
        result_fft = a_fft * b_fft
        return torch.fft.ifft(result_fft, dim=-1).real
    
    def _circular_correlation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Circular correlation (unbinding operation for circular convolution)"""
        # Correlation = convolution with conjugate = element-wise multiply with conjugate in freq domain
        a_fft = torch.fft.fft(a, dim=-1)
        b_fft = torch.fft.fft(b, dim=-1)
        result_fft = a_fft * torch.conj(b_fft)
        return torch.fft.ifft(result_fft, dim=-1).real
    
    def normalize_memory(self, memory: torch.Tensor) -> torch.Tensor:
        """Apply normalization to memory state"""
        if self.normalization == "rms":
            return self._rms_norm(memory) * self.scale
        elif self.normalization == "layer":
            return F.layer_norm(memory, (memory.shape[-1],)) * self.scale
        else:
            return memory
    
    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Root Mean Square normalization"""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
        return x / rms
    
    def apply_decay(self, memory: torch.Tensor) -> torch.Tensor:
        """Apply passive decay to memory"""
        return memory * (1 - self.decay_rate)
    
    def update_memory(
        self,
        old_memory: torch.Tensor,
        new_item: torch.Tensor,
        gate: torch.Tensor
    ) -> torch.Tensor:
        """
        Update memory with gated superposition.
        
        Args:
            old_memory: Previous memory state [batch, num_slots, hcm_dim]
            new_item: New bound item to superpose [batch, hcm_dim]
            gate: Gating values [batch, num_slots, hcm_dim] or [batch, num_slots, 1]
            
        Returns:
            Updated memory state
        """
        # Apply decay to old memory
        old_memory = self.apply_decay(old_memory)
        
        # Expand new_item for multi-slot
        new_item = new_item.unsqueeze(1)  # [batch, 1, hcm_dim]
        
        # Gated update
        updated = (1 - gate) * old_memory + gate * new_item
        
        # Normalize
        return self.normalize_memory(updated)
    
    def initialize_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize empty memory state"""
        return torch.zeros(batch_size, self.num_slots, self.hcm_dim, device=device)
