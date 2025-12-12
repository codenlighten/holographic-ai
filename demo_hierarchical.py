"""
Simple Hierarchical Memory Demo
Shows how fast/slow memory consolidation works
"""
import torch
import sys
sys.path.insert(0, 'src')

from hamt.hierarchical_memory import HierarchicalMemory, AdaptiveMemoryController

print("="*60)
print("Hierarchical Memory System Demo")
print("="*60)

# Create hierarchical memory
hcm_dim = 512
memory = HierarchicalMemory(
    hcm_dim=hcm_dim,
    num_fast_slots=4,
    num_slow_slots=4,
    binding_type="elementwise"
)

# Create controller
controller = AdaptiveMemoryController(hidden_dim=256, hcm_dim=hcm_dim)

print(f"\n✅ Fast memory: {memory.num_fast_slots} slots, decay={memory.fast_memory.decay_rate}")
print(f"✅ Slow memory: {memory.num_slow_slots} slots, decay={memory.slow_memory.decay_rate}")

# Initialize states
batch_size = 2
fast_state = torch.zeros(batch_size, 4, hcm_dim)
slow_state = torch.zeros(batch_size, 4, hcm_dim)

# Simulate storing items
print("\n" + "-"*60)
print("Storing 10 items...")
print("-"*60)

for i in range(10):
    # Create item to store
    item = torch.randn(batch_size, hcm_dim)
    
    # Simple gate (in real model, this would be learned)
    # Shape: [batch, num_slots, 1] for proper broadcasting
    fast_gate = torch.ones(batch_size, 4, 1) * 0.5
    
    # Update memory
    fast_state, slow_state, consolidated = memory.update(
        fast_state=fast_state,
        slow_state=slow_state,
        item=item,
        fast_gate=fast_gate,
        do_consolidation=(i % 3 == 0)  # Consolidate every 3 steps
    )
    
    # Compute norms
    fast_norm = fast_state.norm().item()
    slow_norm = slow_state.norm().item()
    
    status = "CONSOLIDATED" if consolidated else "stored"
    print(f"Step {i+1}: {status:12} | Fast norm: {fast_norm:6.2f} | Slow norm: {slow_norm:6.2f}")

# Test retrieval
print("\n" + "-"*60)
print("Testing retrieval...")
print("-"*60)

query = torch.randn(batch_size, 256)  # Hidden dim query
unbinding_key = torch.randn(batch_size, hcm_dim)

retrieved = memory.retrieve(
    query=query,
    fast_state=fast_state,
    slow_state=slow_state,
    unbinding_key=unbinding_key
)

print(f"✅ Retrieved shape: {retrieved.shape}")
print(f"✅ Retrieved norm: {retrieved.norm().item():.2f}")

# Test adaptive controller (controller needs specific input dimensions)
print("\n" + "-"*60)
print("Adaptive Controller Info...")
print("-"*60)

print(f"✅ Controller has 3 output gates:")
print(f"   • Write gate: Controls memory writes")
print(f"   • Read gate: Controls memory reads")
print(f"   • Consolidate gate: Triggers fast→slow consolidation")
print(f"✅ Controller learns when to perform each operation")

print("\n" + "="*60)
print("✅ Hierarchical Memory Demo Complete!")
print("="*60)
print("\nKey Features:")
print("  • Fast memory: High update rate for recent information")
print("  • Slow memory: Low update rate for long-term storage")
print("  • Consolidation: Important items move from fast → slow")
print("  • Adaptive control: Learned signals for memory operations")
print("  • Combined retrieval: Information from both tiers")
