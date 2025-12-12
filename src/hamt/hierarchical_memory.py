"""
Hierarchical Holographic Memory - Fast and Slow Memory Systems
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .memory import HolographicMemory


class HierarchicalMemory(nn.Module):
    """
    Two-tier memory system:
    - Fast memory: High update rate, short-term retention
    - Slow memory: Low update rate, long-term retention
    """
    
    def __init__(
        self,
        hcm_dim: int,
        num_fast_slots: int = 4,
        num_slow_slots: int = 4,
        binding_type: str = "elementwise",
        fast_update_rate: float = 1.0,
        slow_update_rate: float = 0.1,
        consolidation_threshold: float = 0.5
    ):
        super().__init__()
        self.hcm_dim = hcm_dim
        self.num_fast_slots = num_fast_slots
        self.num_slow_slots = num_slow_slots
        
        # Fast memory - high turnover
        self.fast_memory = HolographicMemory(
            hcm_dim=hcm_dim,
            num_slots=num_fast_slots,
            binding_type=binding_type,
            decay_rate=0.001  # Higher decay for fast memory
        )
        
        # Slow memory - long-term storage
        self.slow_memory = HolographicMemory(
            hcm_dim=hcm_dim,
            num_slots=num_slow_slots,
            binding_type=binding_type,
            decay_rate=0.0001  # Lower decay for slow memory
        )
        
        self.fast_update_rate = fast_update_rate
        self.slow_update_rate = slow_update_rate
        self.consolidation_threshold = consolidation_threshold
        
        # Consolidation network - decides what moves to slow memory
        self.consolidation_gate = nn.Sequential(
            nn.Linear(hcm_dim * 2, hcm_dim),
            nn.GELU(),
            nn.Linear(hcm_dim, 1),
            nn.Sigmoid()
        )
    
    def initialize_memory(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize both fast and slow memory"""
        fast_state = self.fast_memory.initialize_memory(batch_size, device)
        slow_state = self.slow_memory.initialize_memory(batch_size, device)
        return fast_state, slow_state
    
    def consolidate(
        self,
        fast_state: torch.Tensor,
        slow_state: torch.Tensor,
        item: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        Decide whether to consolidate from fast to slow memory
        
        Args:
            fast_state: Current fast memory [batch, num_fast_slots, hcm_dim]
            slow_state: Current slow memory [batch, num_slow_slots, hcm_dim]
            item: New item to potentially consolidate [batch, hcm_dim]
            
        Returns:
            Updated slow_state, consolidation_occurred
        """
        batch_size = item.shape[0]
        
        # Compute consolidation scores
        # Average fast memory representation
        fast_summary = fast_state.mean(dim=1)  # [batch, hcm_dim]
        
        # Combine item with fast memory context
        consolidation_input = torch.cat([item, fast_summary], dim=-1)  # [batch, 2*hcm_dim]
        
        # Decide whether to consolidate
        consolidation_score = self.consolidation_gate(consolidation_input)  # [batch, 1]
        
        # If score exceeds threshold, add to slow memory
        should_consolidate = consolidation_score.squeeze(-1) > self.consolidation_threshold
        
        if should_consolidate.any():
            # Create gates for slow memory update
            slow_gates = torch.zeros(batch_size, self.num_slow_slots, 1, device=item.device)
            slow_gates[:, :, 0] = consolidation_score * self.slow_update_rate
            
            # Update slow memory
            slow_state = self.slow_memory.update_memory(slow_state, item, slow_gates)
            
        return slow_state, should_consolidate.any().item()
    
    def retrieve(
        self,
        query: torch.Tensor,
        fast_state: torch.Tensor,
        slow_state: torch.Tensor,
        unbinding_key: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve from both fast and slow memory
        
        Args:
            query: Query vector [batch, hidden_dim]
            fast_state: Fast memory state [batch, num_fast_slots, hcm_dim]
            slow_state: Slow memory state [batch, num_slow_slots, hcm_dim]
            unbinding_key: Key for unbinding [batch, hcm_dim]
            
        Returns:
            Combined retrieval [batch, hcm_dim]
        """
        # Retrieve from fast memory
        fast_retrieval = self.fast_memory.unbind(fast_state, unbinding_key)
        fast_retrieval = fast_retrieval.mean(dim=1)  # Average across slots
        
        # Retrieve from slow memory
        slow_retrieval = self.slow_memory.unbind(slow_state, unbinding_key)
        slow_retrieval = slow_retrieval.mean(dim=1)  # Average across slots
        
        # Weighted combination (could be learned)
        alpha = 0.7  # Weight for fast memory
        combined = alpha * fast_retrieval + (1 - alpha) * slow_retrieval
        
        return combined
    
    def update(
        self,
        fast_state: torch.Tensor,
        slow_state: torch.Tensor,
        item: torch.Tensor,
        fast_gate: torch.Tensor,
        do_consolidation: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Update hierarchical memory
        
        Args:
            fast_state: Current fast memory
            slow_state: Current slow memory
            item: New item to store
            fast_gate: Gate values for fast memory
            do_consolidation: Whether to attempt consolidation
            
        Returns:
            Updated fast_state, updated slow_state, consolidation_occurred
        """
        # Always update fast memory
        fast_state = self.fast_memory.update_memory(fast_state, item, fast_gate)
        
        # Optionally consolidate to slow memory
        consolidation_occurred = False
        if do_consolidation:
            slow_state, consolidation_occurred = self.consolidate(fast_state, slow_state, item)
        
        return fast_state, slow_state, consolidation_occurred


class AdaptiveMemoryController(nn.Module):
    """
    Adaptive controller that decides memory operations based on input
    """
    
    def __init__(self, hidden_dim: int, hcm_dim: int):
        super().__init__()
        
        # Controller network
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim + hcm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # [write_strength, read_strength, consolidate]
            nn.Sigmoid()
        )
    
    def forward(
        self,
        query: torch.Tensor,
        memory_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate control signals
        
        Args:
            query: Current query [batch, hidden_dim]
            memory_state: Memory summary [batch, hcm_dim]
            
        Returns:
            write_strength, read_strength, consolidate_flag
        """
        # Concatenate query and memory state
        controller_input = torch.cat([query, memory_state], dim=-1)
        
        # Generate control signals
        controls = self.controller(controller_input)  # [batch, 3]
        
        write_strength = controls[:, 0:1]
        read_strength = controls[:, 1:2]
        consolidate_flag = controls[:, 2:3]
        
        return write_strength, read_strength, consolidate_flag
