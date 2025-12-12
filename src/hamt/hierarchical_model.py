"""
Hierarchical HAMT Model
Integrates two-tier memory system into HAMT architecture
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import HAMTConfig
from .hierarchical_memory import HierarchicalMemory
from .layers import RetrievalHead, GatingNetwork
from .model import HAMTBlock


class HierarchicalHAMTLayer(nn.Module):
    """
    HAMT Layer with hierarchical two-tier memory
    """
    
    def __init__(self, config: HAMTConfig):
        super().__init__()
        self.config = config
        
        # Hierarchical memory (fast + slow)
        slots_per_tier = config.num_slots // 2
        self.memory = HierarchicalMemory(
            hcm_dim=config.hcm_dim,
            num_fast_slots=slots_per_tier,
            num_slow_slots=slots_per_tier,
            binding_type=config.binding_type
        )
        self.num_slots_per_tier = slots_per_tier
        
        # Project hidden state to HCM dimension
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
        
        # Gating network for memory updates
        if config.use_gating:
            # Total slots = fast_slots + slow_slots
            total_slots = slots_per_tier * 2
            self.gating_network = GatingNetwork(
                hidden_dim=config.hidden_dim,
                hcm_dim=config.hcm_dim,
                num_slots=total_slots
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
        fast_hcm_state: Optional[torch.Tensor] = None,
        slow_hcm_state: Optional[torch.Tensor] = None,
        positional_keys: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            fast_hcm_state: [batch, num_slots, hcm_dim]
            slow_hcm_state: [batch, num_slots, hcm_dim]
            positional_keys: [seq_len, hcm_dim]
            return_aux_loss: bool
            
        Returns:
            output: [batch, seq_len, hidden_dim]
            new_fast_hcm_state: [batch, num_slots, hcm_dim]
            new_slow_hcm_state: [batch, num_slots, hcm_dim]
            aux_loss: Optional[Tensor]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Generate positional keys if not provided
        if positional_keys is None:
            positional_keys = self.memory.fast_memory.generate_positional_keys(seq_len, device)
        
        # Project to item space
        items = self.item_projection(hidden_states)  # [batch, seq_len, hcm_dim]
        
        # Generate query
        query = self.query_projection(hidden_states)  # [batch, seq_len, hidden_dim]
        
        # Generate unbinding keys using retrieval head
        unbinding_keys = self.retrieval_head(items)  # [batch, seq_len, hcm_dim]
        
        # Retrieve from hierarchical memory
        # For now, average unbinding keys across sequence
        avg_key = unbinding_keys.mean(dim=1)  # [batch, hcm_dim]
        
        # Initialize states if None
        if fast_hcm_state is None:
            fast_hcm_state = torch.zeros(batch_size, self.memory.num_fast_slots, 
                                        self.config.hcm_dim, device=device)
        if slow_hcm_state is None:
            slow_hcm_state = torch.zeros(batch_size, self.memory.num_slow_slots,
                                        self.config.hcm_dim, device=device)
        
        # Retrieve for each position in sequence
        retrieved_list = []
        for i in range(seq_len):
            ret = self.memory.retrieve(
                query=query[:, i, :],  # [batch, hidden_dim]
                fast_state=fast_hcm_state,
                slow_state=slow_hcm_state,
                unbinding_key=unbinding_keys[:, i, :]  # [batch, hcm_dim]
            )
            retrieved_list.append(ret)
        
        retrieved = torch.stack(retrieved_list, dim=1)  # [batch, seq_len, hcm_dim]
        
        # Compute gating if enabled
        if self.gating_network is not None:
            # Compute gates for both tiers combined
            gate_input = torch.cat([hidden_states, retrieved], dim=-1)
            gates = self.gating_network(gate_input)  # [batch, seq_len, total_slots]
            
            # Split gates for fast and slow memory
            fast_gates = gates[:, :, :self.num_slots_per_tier]  # [batch, seq_len, num_fast_slots]
        else:
            fast_gates = torch.ones(batch_size, seq_len, self.num_slots_per_tier, device=device)
        
        # Update hierarchical memory (process each sequence position)
        new_fast_state = fast_hcm_state.clone()
        new_slow_state = slow_hcm_state.clone()
        
        for i in range(seq_len):
            new_fast_state, new_slow_state, _ = self.memory.update(
                fast_state=new_fast_state,
                slow_state=new_slow_state,
                item=items[:, i, :],  # [batch, hcm_dim]
                fast_gate=fast_gates[:, i, :],  # [batch, num_fast_slots]
                do_consolidation=(i % 10 == 0)  # Consolidate every 10 steps
            )
        
        # Combine query and retrieved information
        combined = torch.cat([query, retrieved], dim=-1)
        output = self.output_projection(combined)
        output = self.dropout(output)
        
        # Residual connection and normalization
        output = self.layer_norm(hidden_states + output)
        
        # Auxiliary loss (optional)
        aux_loss = None
        if return_aux_loss:
            # Reconstruction loss: can we retrieve what we just stored?
            reconstructed = self.memory.retrieve(
                query=items,
                fast_memory_state=new_fast_state,
                slow_memory_state=new_slow_state,
                positional_keys=positional_keys
            )
            aux_loss = nn.functional.mse_loss(reconstructed, items)
        
        return output, new_fast_state, new_slow_state, aux_loss


class HierarchicalHAMTBlock(nn.Module):
    """
    Transformer block with hierarchical HAMT layer
    """
    
    def __init__(self, config: HAMTConfig):
        super().__init__()
        self.config = config
        
        # Hierarchical HAMT layer
        self.hamt_layer = HierarchicalHAMTLayer(config)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.intermediate_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        fast_hcm_state: Optional[torch.Tensor] = None,
        slow_hcm_state: Optional[torch.Tensor] = None,
        positional_keys: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False
    ):
        """Forward pass through hierarchical block"""
        # Hierarchical HAMT layer
        hidden_states, new_fast_state, new_slow_state, aux_loss = self.hamt_layer(
            hidden_states,
            fast_hcm_state=fast_hcm_state,
            slow_hcm_state=slow_hcm_state,
            positional_keys=positional_keys,
            return_aux_loss=return_aux_loss
        )
        
        # FFN with residual connection
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_layer_norm(hidden_states + ffn_output)
        
        return hidden_states, new_fast_state, new_slow_state, aux_loss


class HierarchicalHAMTModel(nn.Module):
    """
    Complete HAMT model with hierarchical two-tier memory
    """
    
    def __init__(self, config: HAMTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Positional embeddings
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_dim)
        
        # Dropout
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Hierarchical HAMT blocks
        self.blocks = nn.ModuleList([
            HierarchicalHAMTBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.hidden_dim)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        fast_hcm_states: Optional[list] = None,
        slow_hcm_states: Optional[list] = None,
        labels: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True
    ):
        """
        Args:
            input_ids: [batch, seq_len]
            fast_hcm_states: List of [batch, num_slots, hcm_dim] for each layer
            slow_hcm_states: List of [batch, num_slots, hcm_dim] for each layer
            labels: [batch, seq_len] for language modeling loss
            return_aux_loss: Whether to compute auxiliary losses
            
        Returns:
            Dictionary with logits, loss, fast_states, slow_states, aux_loss
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)
        hidden_states = self.embedding_dropout(token_embeds + position_embeds)
        
        # Initialize memory states if not provided
        if fast_hcm_states is None:
            fast_hcm_states = [None] * self.config.num_layers
        if slow_hcm_states is None:
            slow_hcm_states = [None] * self.config.num_layers
        
        # Track new states and auxiliary losses
        new_fast_states = []
        new_slow_states = []
        aux_losses = []
        
        # Pass through blocks
        for i, block in enumerate(self.blocks):
            hidden_states, fast_state, slow_state, aux_loss = block(
                hidden_states,
                fast_hcm_state=fast_hcm_states[i],
                slow_hcm_state=slow_hcm_states[i],
                return_aux_loss=return_aux_loss
            )
            new_fast_states.append(fast_state)
            new_slow_states.append(slow_state)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Add auxiliary loss if enabled
            if return_aux_loss and aux_losses:
                total_aux_loss = sum(aux_losses) / len(aux_losses)
                loss = loss + self.config.auxiliary_loss_weight * total_aux_loss
        
        return {
            'logits': logits,
            'loss': loss,
            'fast_hcm_states': new_fast_states,
            'slow_hcm_states': new_slow_states,
            'aux_loss': sum(aux_losses) / len(aux_losses) if aux_losses else None
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = no filtering)
            
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        fast_hcm_states = None
        slow_hcm_states = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self(
                input_ids,
                fast_hcm_states=fast_hcm_states,
                slow_hcm_states=slow_hcm_states,
                return_aux_loss=False
            )
            
            # Update memory states
            fast_hcm_states = outputs['fast_hcm_states']
            slow_hcm_states = outputs['slow_hcm_states']
            
            # Get logits for last token
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


if __name__ == "__main__":
    # Test hierarchical model
    from .config import HAMTConfig
    
    config = HAMTConfig(
        hidden_dim=256,
        hcm_dim=1024,
        num_layers=2,
        num_slots=8,  # Will be split: 4 fast, 4 slow
        vocab_size=1000
    )
    
    model = HierarchicalHAMTModel(config)
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 16))
    outputs = model(input_ids)
    
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Fast states: {len(outputs['fast_hcm_states'])} layers")
    print(f"Slow states: {len(outputs['slow_hcm_states'])} layers")
    if outputs['aux_loss'] is not None:
        print(f"Aux loss: {outputs['aux_loss'].item():.4f}")
    
    # Test generation
    generated = model.generate(input_ids[:1, :8], max_new_tokens=10)
    print(f"Generated shape: {generated.shape}")
    print("✅ Hierarchical HAMT model working!")
