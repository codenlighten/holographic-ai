"""
Quick demo of HAMT model
"""
import torch
from hamt import HAMTConfig, HAMTModel

def main():
    print("=" * 60)
    print("HAMT (Holographic Associative Memory Transformer) Demo")
    print("=" * 60)
    
    # Create a small config for demo
    config = HAMTConfig(
        hidden_dim=256,
        hcm_dim=1024,
        num_layers=4,
        num_slots=8,
        num_attention_heads=8,
        vocab_size=1000,
        max_position_embeddings=256,
        binding_type="elementwise",
        use_auxiliary_loss=True
    )
    
    print("\nModel Configuration:")
    print(f"  Hidden Dimension: {config.hidden_dim}")
    print(f"  HCM Dimension: {config.hcm_dim}")
    print(f"  Number of Layers: {config.num_layers}")
    print(f"  Memory Slots: {config.num_slots}")
    print(f"  Binding Type: {config.binding_type}")
    
    # Create model
    model = HAMTModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    
    # Create sample input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput Shape: {input_ids.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, return_aux_loss=True)
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    if 'aux_loss' in outputs:
        print(f"Auxiliary Loss: {outputs['aux_loss'].item():.4f}")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids[:1, :10],  # First sequence, first 10 tokens
            max_new_tokens=20,
            temperature=1.0
        )
    
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated tokens: {generated[0, :15].tolist()}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
