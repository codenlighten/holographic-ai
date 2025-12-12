"""Test hierarchical HAMT model"""
import sys
sys.path.insert(0, 'src')

from hamt.hierarchical_model import HierarchicalHAMTModel
from hamt.config import HAMTConfig
import torch

# Create config
config = HAMTConfig(
    hidden_dim=256,
    hcm_dim=1024,
    num_layers=2,
    num_slots=8,
    num_attention_heads=8,
    vocab_size=1000
)

# Create model
model = HierarchicalHAMTModel(config)

# Test forward pass
print("Testing forward pass...")
x = torch.randint(0, 1000, (2, 16))
out = model(x)

print(f'✅ Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
print(f'✅ Logits: {out["logits"].shape}')
print(f'✅ Fast states: {len(out["fast_hcm_states"])} layers')
print(f'✅ Slow states: {len(out["slow_hcm_states"])} layers')
print(f'✅ Aux loss: {out["aux_loss"].item():.4f}')

# Test generation
print("\nTesting generation...")
gen = model.generate(x[:1, :8], max_new_tokens=5, temperature=0.8)
print(f'✅ Generation: {gen.shape}')
print(f'✅ Input: {x[:1, :8].tolist()}')
print(f'✅ Generated: {gen[0].tolist()}')

print('\n🎉 Hierarchical HAMT working!')
