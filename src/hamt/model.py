"""
Complete HAMT Transformer Model
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from .config import HAMTConfig
from .layers import HAMTLayer


class HAMTBlock(nn.Module):
    """
    Single transformer block with HAMT layer + FFN
    """
    
    def __init__(self, config: HAMTConfig):
        super().__init__()
        self.config = config
        
        # HAMT layer (replaces self-attention)
        self.hamt_layer = HAMTLayer(config)
        
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
        hcm_state: Optional[torch.Tensor] = None,
        positional_keys: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False
    ):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            hcm_state: [batch, num_slots, hcm_dim]
            positional_keys: [seq_len, hcm_dim]
            return_aux_loss: bool
            
        Returns:
            output: [batch, seq_len, hidden_dim]
            new_hcm_state: [batch, num_slots, hcm_dim]
            aux_loss: Optional[Tensor]
        """
        # HAMT layer with residual connection (already included in HAMTLayer)
        hidden_states, new_hcm_state, aux_loss = self.hamt_layer(
            hidden_states,
            hcm_state=hcm_state,
            positional_keys=positional_keys,
            return_aux_loss=return_aux_loss
        )
        
        # FFN with residual connection
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.ffn_layer_norm(hidden_states + ffn_output)
        
        return hidden_states, new_hcm_state, aux_loss


class HAMTModel(nn.Module):
    """
    Complete HAMT Transformer model for language modeling
    """
    
    def __init__(self, config: HAMTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Traditional positional embeddings (for input, separate from HCM positional keys)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_dim)
        
        # Dropout
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # HAMT blocks
        self.blocks = nn.ModuleList([
            HAMTBlock(config) for _ in range(config.num_layers)
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
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hcm_states: Optional[List[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        return_aux_loss: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            hcm_states: List of HCM states per layer (for recurrent processing)
            labels: [batch, seq_len] for computing loss
            return_aux_loss: Whether to compute auxiliary loss (defaults to config)
            
        Returns:
            Dictionary containing:
                - logits: [batch, seq_len, vocab_size]
                - loss: scalar (if labels provided)
                - aux_loss: scalar (if return_aux_loss=True)
                - hcm_states: List of final HCM states per layer
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if return_aux_loss is None:
            return_aux_loss = self.config.use_auxiliary_loss and self.training
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, hidden_dim]
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        position_embeds = self.position_embedding(positions)
        
        hidden_states = self.embedding_dropout(token_embeds + position_embeds)
        
        # Generate HCM positional keys (shared across all layers)
        # These are different from input position embeddings
        positional_keys = self.blocks[0].hamt_layer.memory.generate_positional_keys(seq_len, device)
        
        # Initialize HCM states if not provided
        if hcm_states is None:
            hcm_states = [None] * self.config.num_layers
        
        # Process through blocks
        new_hcm_states = []
        total_aux_loss = 0.0
        aux_loss_count = 0
        
        for layer_idx, (block, hcm_state) in enumerate(zip(self.blocks, hcm_states)):
            hidden_states, new_hcm_state, aux_loss = block(
                hidden_states,
                hcm_state=hcm_state,
                positional_keys=positional_keys,
                return_aux_loss=return_aux_loss
            )
            new_hcm_states.append(new_hcm_state)
            
            if aux_loss is not None:
                total_aux_loss += aux_loss
                aux_loss_count += 1
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Add auxiliary loss if computed
            if aux_loss_count > 0:
                avg_aux_loss = total_aux_loss / aux_loss_count
                loss = loss + self.config.aux_loss_weight * avg_aux_loss
        
        # Prepare output
        output = {
            'logits': logits,
            'hcm_states': new_hcm_states
        }
        
        if loss is not None:
            output['loss'] = loss
        
        if aux_loss_count > 0:
            output['aux_loss'] = total_aux_loss / aux_loss_count
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        hcm_states: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
            hcm_states: Initial HCM states (for continuing generation)
            
        Returns:
            generated_ids: [batch, seq_len + max_new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(input_ids, hcm_states=hcm_states, return_aux_loss=False)
                logits = outputs['logits']
                hcm_states = outputs['hcm_states']
            
            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
