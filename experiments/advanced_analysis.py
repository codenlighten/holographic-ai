"""
Advanced analysis of trained HAMT model
Uses visualization tools to analyze memory behavior
"""
import torch
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from hamt import HAMTModel, HAMTConfig
from hamt.visualization import (
    plot_memory_evolution,
    plot_slot_attention,
    plot_retrieval_similarity,
    create_analysis_dashboard
)


def load_trained_model(checkpoint_path: str):
    """Load trained HAMT model"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Config might already be a HAMTConfig object
    config_data = checkpoint['config']
    if isinstance(config_data, HAMTConfig):
        config = config_data
    else:
        config = HAMTConfig(**config_data)
    
    model = HAMTModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def analyze_memory_dynamics(
    model: HAMTModel,
    input_text: str,
    tokenizer_vocab: int = 5000,
    output_dir: str = "advanced_analysis"
):
    """
    Analyze memory dynamics during sequence processing
    
    Args:
        model: Trained HAMT model
        input_text: Text to analyze
        tokenizer_vocab: Vocab size for simple tokenization
        output_dir: Output directory for plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Simple tokenization (split by chars for demo)
    tokens = [ord(c) % tokenizer_vocab for c in input_text]
    input_ids = torch.tensor([tokens])
    
    print(f"Analyzing sequence of length {len(tokens)}...")
    print(f"Input text: {input_text[:100]}...")
    
    # Track memory states through layers
    memory_states_by_layer = {i: [] for i in range(model.config.num_layers)}
    attention_weights_by_layer = {}
    
    # Hook to capture memory states
    layer_outputs = {}
    
    def hook_fn(layer_idx):
        def hook(module, input, output):
            # Store output
            layer_outputs[layer_idx] = output
        return hook
    
    # Register hooks
    hooks = []
    for i, block in enumerate(model.blocks):
        hook = block.hamt_layer.register_forward_hook(hook_fn(i))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs['logits']
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"Forward pass complete. Output shape: {logits.shape}")
    
    # Get memory states from model forward pass
    # We need to track memory through the forward pass
    memory_states = []
    
    with torch.no_grad():
        # Forward with memory tracking
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        token_embeds = model.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeds = model.position_embedding(positions)
        hidden_states = model.embedding_dropout(token_embeds + position_embeds)
        
        hcm_state = None
        
        for layer_idx, block in enumerate(model.blocks):
            # Use block's forward method
            hidden_states, hcm_state, _ = block(hidden_states, hcm_state)
            
            # Store memory state
            memory_states_by_layer[layer_idx].append(hcm_state.clone())
            
            print(f"\nLayer {layer_idx}:")
            print(f"  Memory shape: {hcm_state.shape}")
            print(f"  Memory norm: {hcm_state.norm():.2f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Plot memory evolution for first and last layer
    for layer_idx in [0, model.config.num_layers - 1]:
        if memory_states_by_layer[layer_idx]:
            # Extract first batch: [batch, num_slots, hcm_dim] -> [num_slots, hcm_dim]
            states = [s[0] for s in memory_states_by_layer[layer_idx]]
            
            # Duplicate state for visualization (since we only have one state per layer)
            states = states * 10
            
            plot_memory_evolution(
                states,
                title=f"Layer {layer_idx} Memory State",
                output_file=str(output_path / f"layer_{layer_idx}_memory.png")
            )
    
    # Analyze slot usage
    print("\nSlot Usage Analysis:")
    for layer_idx in range(model.config.num_layers):
        if memory_states_by_layer[layer_idx]:
            memory = memory_states_by_layer[layer_idx][0][0]  # [batch, num_slots, hcm_dim] -> first batch
            
            # Compute norm for each slot
            slot_norms = memory.norm(dim=-1)
            print(f"  Layer {layer_idx}: {slot_norms.tolist()}")
    
    # Save analysis results
    results = {
        'input_length': len(tokens),
        'num_layers': model.config.num_layers,
        'num_slots': model.config.num_slots,
        'hcm_dim': model.config.hcm_dim,
        'output_shape': list(logits.shape),
        'layer_memory_norms': {}
    }
    
    for layer_idx in range(model.config.num_layers):
        if memory_states_by_layer[layer_idx]:
            memory = memory_states_by_layer[layer_idx][0][0]  # [batch, num_slots, hcm_dim] -> first batch
            results['layer_memory_norms'][f'layer_{layer_idx}'] = float(memory.norm())
    
    with open(output_path / "analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    
    return results


def analyze_sequence_processing(
    model: HAMTModel,
    sequence_length: int = 100,
    output_dir: str = "sequence_analysis"
):
    """
    Analyze how model processes sequences step by step
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Analyzing sequence processing (length={sequence_length})...")
    
    # Generate random sequence
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, sequence_length))
    
    # Track memory evolution
    memory_evolution = []
    
    with torch.no_grad():
        # Process sequence in chunks to see evolution
        chunk_size = 10
        hcm_state = None
        
        for i in range(0, sequence_length, chunk_size):
            chunk = input_ids[:, i:i+chunk_size]
            
            # Forward pass through first block only
            batch_size, seq_len = chunk.shape
            device = chunk.device
            
            token_embeds = model.token_embedding(chunk)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            position_embeds = model.position_embedding(positions)
            hidden_states = model.embedding_dropout(token_embeds + position_embeds)
            
            # Pass through first block
            hidden_states, hcm_state, _ = model.blocks[0](hidden_states, hcm_state)
            
            # Capture memory state
            memory_evolution.append(hcm_state.clone())
    
    # Visualize memory evolution (extract first batch)
    memory_states_unbatched = [m[0] for m in memory_evolution]
    
    plot_memory_evolution(
        memory_states_unbatched,
        title="Memory Evolution Over Sequence",
        output_file=str(output_path / "sequence_memory_evolution.png")
    )
    
    print(f"Sequence analysis saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced HAMT Analysis")
    parser.add_argument("--checkpoint", type=str, 
                        default="checkpoints/quick_train/hamt_quick_100steps.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--text", type=str,
                        default="The quick brown fox jumps over the lazy dog. This is a test of holographic memory.",
                        help="Text to analyze")
    parser.add_argument("--output_dir", type=str, default="advanced_analysis",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Please train a model first using experiments/quick_train.py")
        sys.exit(1)
    
    print("Loading trained model...")
    model, config = load_trained_model(args.checkpoint)
    
    print(f"Model loaded: {config.num_layers}L, {config.hidden_dim}H, {config.num_slots} slots")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")
    
    # Run analysis
    results = analyze_memory_dynamics(
        model,
        args.text,
        tokenizer_vocab=config.vocab_size,
        output_dir=args.output_dir
    )
    
    # Additional sequence analysis
    analyze_sequence_processing(
        model,
        sequence_length=100,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to {args.output_dir}/")
    print("\nGenerated files:")
    for f in sorted(Path(args.output_dir).glob("*")):
        print(f"  - {f.name}")
