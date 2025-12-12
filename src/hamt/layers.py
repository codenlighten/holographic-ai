"""
HAMT Layer implementation with gating, retrieval, and auxiliary loss support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .memory import HolographicMemory
from .config import HAMTConfig


class RetrievalHead(nn.Module):
    """
    Generates unbinding key from query and retrieves from HCM.
    Two-stage process: query -> unbinding_key -> retrieval
    """
    
    def __init__(self, hidden_dim: int, hcm_dim: int, hidden_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hcm_dim = hcm_dim
        
        # MLP to generate unbinding key from query
        self.key_generator = nn.Sequential(
            nn.Linear(hidden_dim, hcm_dim * hidden_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hcm_dim * hidden_factor, hcm_dim)
        )
        
        # Slot attention weights (learn to combine multiple memory slots)
        self.slot_attention = nn.Linear(hcm_dim, 1)
        
    def forward(
        self,
        query: torch.Tensor,
        memory: HolographicMemory,
        hcm_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query vector [batch, seq_len, hidden_dim]
            memory: HolographicMemory module
            hcm_state: Current HCM state [batch, num_slots, hcm_dim]
            
        Returns:
            context_vector: Retrieved context [batch, seq_len, hcm_dim]
            unbinding_keys: Generated unbinding keys [batch, seq_len, hcm_dim]
        """
        batch_size, seq_len, _ = query.shape
        num_slots = hcm_state.shape[1]
        
        # Generate unbinding keys from queries
        unbinding_keys = self.key_generator(query)  # [batch, seq_len, hcm_dim]
        
        # Normalize unbinding keys (ensure they're in proper space)
        unbinding_keys = F.normalize(unbinding_keys, p=2, dim=-1)
        
        # Retrieve from each slot
        # Expand for broadcasting: [batch, seq_len, num_slots, hcm_dim]
        hcm_expanded = hcm_state.unsqueeze(1).expand(-1, seq_len, -1, -1)
        keys_expanded = unbinding_keys.unsqueeze(2).expand(-1, -1, num_slots, -1)
        
        # Unbind from each slot
        retrieved_per_slot = memory.unbind(hcm_expanded, keys_expanded)  # [batch, seq_len, num_slots, hcm_dim]
        
        # Compute slot attention weights
        slot_scores = self.slot_attention(retrieved_per_slot).squeeze(-1)  # [batch, seq_len, num_slots]
        slot_weights = F.softmax(slot_scores, dim=-1)
        
        # Weighted combination of slots
        context_vector = torch.einsum('bsnh,bsn->bsh', retrieved_per_slot, slot_weights)
        
        return context_vector, unbinding_keys


class GatingNetwork(nn.Module):
    """
    Learns gating values for memory update.
    Takes query, item, and memory statistics to decide how much to write.
    """
    
    def __init__(self, hidden_dim: int, hcm_dim: int, num_slots: int):
        super().__init__()
        
        # Input: [query, item, memory_stats]
        input_dim = hidden_dim + hcm_dim + num_slots
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_slots),
            nn.Sigmoid()  # Gate values in [0, 1]
        )
    
    def forward(
        self,
        query: torch.Tensor,
        item: torch.Tensor,
        hcm_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len, hidden_dim]
            item: [batch, seq_len, hcm_dim]
            hcm_state: [batch, num_slots, hcm_dim]
            
        Returns:
            gates: [batch, seq_len, num_slots, 1]
        """
        batch_size, seq_len = query.shape[:2]
        
        # Compute memory statistics (L2 norm per slot as capacity indicator)
        memory_stats = torch.norm(hcm_state, p=2, dim=-1)  # [batch, num_slots]
        memory_stats = memory_stats.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, num_slots]
        
        # Concatenate inputs
        gate_input = torch.cat([
            query,
            item,
            memory_stats
        ], dim=-1)  # [batch, seq_len, hidden_dim + hcm_dim + num_slots]
        
        # Compute gates
        gates = self.gate_mlp(gate_input)  # [batch, seq_len, num_slots]
        
        return gates.unsqueeze(-1)  # [batch, seq_len, num_slots, 1]


class HAMTLayer(nn.Module):
    """
    Single HAMT layer replacing traditional self-attention.
    Maintains holographic context memory and performs associative retrieval.
    """
    
    def __init__(self, config: HAMTConfig):
        super().__init__()
        self.config = config
        
        # Holographic memory
        self.memory = HolographicMemory(
            hcm_dim=config.hcm_dim,
            num_slots=config.num_slots,
            binding_type=config.binding_type,
            decay_rate=config.decay_rate,
            normalization=config.normalization_type
        )
        
        # Project hidden state to HCM dimension (item vector)
        self.item_projection = nn.Linear(config.hidden_dim, config.hcm_dim)
        
        # Query projection
        self.query_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Retrieval head
        self.retrieval_head = RetrievalHead(
            hidden_dim=config.hidden_dim,
            hcm_dim=config.hcm_dim,
            hidden_factor=2,
            dropout=config.retrieval_dropout
        )
        
        # Gating network
        if config.use_gating:
            self.gating_network = GatingNetwork(
                hidden_dim=config.hidden_dim,
                hcm_dim=config.hcm_dim,
                num_slots=config.num_slots
            )
        else:
            self.gating_network = None
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim + config.hcm_dim, config.hidden_dim)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        hcm_state: Optional[torch.Tensor] = None,
        positional_keys: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_dim]
            hcm_state: Previous HCM state [batch, num_slots, hcm_dim] or None
            positional_keys: Positional keys [seq_len, hcm_dim] or None
            return_aux_loss: Whether to compute auxiliary reconstruction loss
            
        Returns:
            output: Output hidden states [batch, seq_len, hidden_dim]
            new_hcm_state: Updated HCM state [batch, num_slots, hcm_dim]
            aux_loss: Auxiliary reconstruction loss (if return_aux_loss=True)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Initialize HCM if not provided
        if hcm_state is None:
            hcm_state = self.memory.initialize_memory(batch_size, device)
        
        # Generate positional keys if not provided
        if positional_keys is None:
            positional_keys = self.memory.generate_positional_keys(seq_len, device)
        
        # Project to item vectors
        item_vectors = self.item_projection(hidden_states)  # [batch, seq_len, hcm_dim]
        
        # Project queries
        queries = self.query_projection(hidden_states)  # [batch, seq_len, hidden_dim]
        
        # Process each token sequentially (for proper recurrent HCM update)
        outputs = []
        aux_losses = []
        
        for t in range(seq_len):
            # Current item and query
            current_item = item_vectors[:, t, :]  # [batch, hcm_dim]
            current_query = queries[:, t:t+1, :]  # [batch, 1, hidden_dim]
            current_pos_key = positional_keys[t, :]  # [hcm_dim]
            
            # 1. Bind item with position
            bound_item = self.memory.bind(current_item, current_pos_key)  # [batch, hcm_dim]
            
            # 2. Retrieve from current HCM
            context_vector, unbinding_key = self.retrieval_head(
                current_query, self.memory, hcm_state
            )  # [batch, 1, hcm_dim], [batch, 1, hcm_dim]
            
            # 3. Compute gating values
            if self.gating_network is not None:
                gates = self.gating_network(
                    current_query,
                    current_item.unsqueeze(1),
                    hcm_state
                )  # [batch, 1, num_slots, 1]
                gates = gates.squeeze(1)  # [batch, num_slots, 1]
            else:
                gates = torch.ones(batch_size, self.config.num_slots, 1, device=device)
            
            # 4. Update HCM with new bound item
            hcm_state = self.memory.update_memory(hcm_state, bound_item, gates)
            
            # 5. Combine query and context for output
            combined = torch.cat([current_query.squeeze(1), context_vector.squeeze(1)], dim=-1)
            output_t = self.output_projection(combined)  # [batch, hidden_dim]
            outputs.append(output_t)
            
            # 6. Auxiliary loss: try to reconstruct item from HCM using position
            if return_aux_loss and self.training:
                # Unbind using position key to recover item
                reconstructed_item = self.memory.unbind(hcm_state, current_pos_key)
                # Take mean across slots
                reconstructed_item = reconstructed_item.mean(dim=1)  # [batch, hcm_dim]
                # Reconstruction loss
                aux_loss_t = F.mse_loss(reconstructed_item, current_item.detach())
                aux_losses.append(aux_loss_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden_dim]
        
        # Residual connection and normalization
        output = self.layer_norm(hidden_states + self.dropout(output))
        
        # Compute average auxiliary loss
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        
        return output, hcm_state, aux_loss
