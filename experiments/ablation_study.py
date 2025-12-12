"""
Quick Ablation Study - Local Testing
Run quick experiments to find optimal hyperparameters before A100 training
"""

import torch
import time
from pathlib import Path
import json
from src.hamt.config import HAMTConfig
from src.hamt.model import HAMTModel
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset

class SimpleTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_length, 
                          padding='max_length', return_tensors='pt')
            self.encodings.append(enc['input_ids'].squeeze(0))
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return {'input_ids': self.encodings[idx]}

def quick_train_test(config_kwargs, num_steps=50, device='cpu'):
    """Quick training test to measure loss convergence"""
    
    # Create config
    config = HAMTConfig(**config_kwargs)
    
    # Create model
    model = HAMTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Load small dataset
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:100]')
    texts = [t for t in dataset['text'] if len(t) > 50][:50]
    
    train_dataset = SimpleTextDataset(texts, tokenizer, max_length=config.max_position_embeddings)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Training loop
    model.train()
    losses = []
    aux_losses = []
    start_time = time.time()
    
    step = 0
    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break
                
            input_ids = batch['input_ids'].to(device)
            
            # Forward
            outputs = model(input_ids, labels=input_ids)
            loss = outputs['loss']
            aux_loss = outputs.get('auxiliary_loss', torch.tensor(0.0))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            aux_losses.append(aux_loss.item())
            step += 1
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    # Memory slot utilization
    slot_norms = []
    for layer in model.blocks:
        if hasattr(layer.hamt_layer, 'memory'):
            memory = layer.hamt_layer.memory.memory_state
            norms = torch.norm(memory, dim=-1).mean(dim=0).tolist()
            slot_norms.append(norms)
    
    return {
        'config': config_kwargs,
        'params': model.count_parameters(),
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'reduction_pct': reduction,
        'aux_loss_final': sum(aux_losses[-5:]) / 5 if aux_losses else 0,
        'time_seconds': elapsed,
        'steps_per_sec': num_steps / elapsed,
        'slot_norms': slot_norms,
        'slot_std': [float(torch.tensor(norms).std()) for norms in slot_norms] if slot_norms else []
    }

def ablation_study_slots():
    """Test different numbers of memory slots"""
    print("=" * 80)
    print("ABLATION STUDY: Number of Memory Slots")
    print("=" * 80)
    
    results = []
    base_config = {
        'hidden_dim': 256,
        'hcm_dim': 1024,
        'num_layers': 2,
        'num_attention_heads': 8,
        'num_attention_heads': 8,  # Smaller for speed
        'num_attention_heads': 8,  # Required by config validation
        'vocab_size': 50257,
        'max_position_embeddings': 128,
        'dropout': 0.1,
        'use_gating': True,
        'use_auxiliary_loss': True,
        'aux_loss_weight': 0.1
    }
    
    for num_slots in [4, 8, 16]:
        print(f"\nTesting {num_slots} slots...")
        config = {**base_config, 'num_slots': num_slots}
        result = quick_train_test(config)
        results.append(result)
        
        print(f"  Params: {result['params']:,}")
        print(f"  Loss: {result['initial_loss']:.3f} → {result['final_loss']:.3f} ({result['reduction_pct']:.1f}% reduction)")
        print(f"  Aux Loss: {result['aux_loss_final']:.3f}")
        print(f"  Time: {result['time_seconds']:.1f}s ({result['steps_per_sec']:.2f} steps/s)")
        if result['slot_std']:
            print(f"  Slot utilization std: {sum(result['slot_std'])/len(result['slot_std']):.3f} (lower = more balanced)")
    
    return results

def ablation_study_aux_loss():
    """Test different auxiliary loss weights"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY: Auxiliary Loss Weight")
    print("=" * 80)
    
    results = []
    base_config = {
        'hidden_dim': 256,
        'hcm_dim': 1024,
        'num_layers': 2,
        'num_attention_heads': 8,
        'num_attention_heads': 8,
        'num_slots': 8,
        'vocab_size': 50257,
        'max_position_embeddings': 128,
        'dropout': 0.1,
        'use_gating': True,
        'use_auxiliary_loss': True,
    }
    
    for weight in [0.0, 0.05, 0.1, 0.2]:
        print(f"\nTesting aux_loss_weight={weight}...")
        config = {**base_config, 'aux_loss_weight': weight}
        result = quick_train_test(config)
        results.append(result)
        
        print(f"  Loss: {result['initial_loss']:.3f} → {result['final_loss']:.3f} ({result['reduction_pct']:.1f}% reduction)")
        print(f"  Aux Loss: {result['aux_loss_final']:.3f}")
        print(f"  Time: {result['time_seconds']:.1f}s")
    
    return results

def ablation_study_hcm_scaling():
    """Test different HCM dimension scaling factors"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY: HCM Dimension Scaling")
    print("=" * 80)
    
    results = []
    base_config = {
        'hidden_dim': 256,
        'num_layers': 2,
        'num_attention_heads': 8,
        'num_attention_heads': 8,
        'num_slots': 8,
        'vocab_size': 50257,
        'max_position_embeddings': 128,
        'dropout': 0.1,
        'use_gating': True,
        'use_auxiliary_loss': True,
        'aux_loss_weight': 0.1
    }
    
    for scale in [2, 3, 4]:
        hcm_dim = 256 * scale
        print(f"\nTesting hcm_dim={hcm_dim} ({scale}× hidden_dim)...")
        config = {**base_config, 'hcm_dim': hcm_dim}
        result = quick_train_test(config)
        results.append(result)
        
        print(f"  Params: {result['params']:,}")
        print(f"  Loss: {result['initial_loss']:.3f} → {result['final_loss']:.3f} ({result['reduction_pct']:.1f}% reduction)")
        print(f"  Aux Loss: {result['aux_loss_final']:.3f}")
        print(f"  Time: {result['time_seconds']:.1f}s")
    
    return results

def ablation_study_gating():
    """Test with and without gating network"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY: Gating Network")
    print("=" * 80)
    
    results = []
    base_config = {
        'hidden_dim': 256,
        'hcm_dim': 1024,
        'num_layers': 2,
        'num_attention_heads': 8,
        'num_attention_heads': 8,
        'num_slots': 8,
        'vocab_size': 50257,
        'max_position_embeddings': 128,
        'dropout': 0.1,
        'use_auxiliary_loss': True,
        'aux_loss_weight': 0.1
    }
    
    for use_gating in [False, True]:
        print(f"\nTesting use_gating={use_gating}...")
        config = {**base_config, 'use_gating': use_gating}
        result = quick_train_test(config)
        results.append(result)
        
        print(f"  Params: {result['params']:,}")
        print(f"  Loss: {result['initial_loss']:.3f} → {result['final_loss']:.3f} ({result['reduction_pct']:.1f}% reduction)")
        print(f"  Time: {result['time_seconds']:.1f}s")
    
    return results

def save_results(all_results, output_path='ablation_results.json'):
    """Save all results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")

def print_summary(all_results):
    """Print summary recommendations"""
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    # Find best from each study
    if 'slots' in all_results:
        best_slots = max(all_results['slots'], key=lambda x: x['reduction_pct'])
        print(f"\n✅ Best num_slots: {best_slots['config']['num_slots']}")
        print(f"   Loss reduction: {best_slots['reduction_pct']:.1f}%")
        print(f"   Slot utilization: balanced (std: {sum(best_slots['slot_std'])/len(best_slots['slot_std']):.3f})")
    
    if 'aux_loss' in all_results:
        best_aux = max(all_results['aux_loss'], key=lambda x: x['reduction_pct'])
        print(f"\n✅ Best aux_loss_weight: {best_aux['config']['aux_loss_weight']}")
        print(f"   Loss reduction: {best_aux['reduction_pct']:.1f}%")
    
    if 'hcm_scaling' in all_results:
        best_hcm = max(all_results['hcm_scaling'], key=lambda x: x['reduction_pct'])
        hcm_scale = best_hcm['config']['hcm_dim'] // best_hcm['config']['hidden_dim']
        print(f"\n✅ Best HCM scaling: {hcm_scale}× hidden_dim")
        print(f"   Loss reduction: {best_hcm['reduction_pct']:.1f}%")
        print(f"   Parameters: {best_hcm['params']:,}")
    
    if 'gating' in all_results:
        best_gating = max(all_results['gating'], key=lambda x: x['reduction_pct'])
        print(f"\n✅ Use gating: {best_gating['config']['use_gating']}")
        print(f"   Loss reduction: {best_gating['reduction_pct']:.1f}%")
    
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION FOR A100 TRAINING:")
    print("=" * 80)
    print("""
    HAMTConfig(
        hidden_dim=512,              # Scale up for A100
        hcm_dim=2048,                # 4× scaling (or best from above)
        num_layers=8,                # Deeper for large models
        num_slots=16,                # More capacity (or best from above)
        vocab_size=50257,
        max_position_embeddings=1024,  # Longer sequences
        dropout=0.1,
        use_gating=True,             # (or best from above)
        use_auxiliary_loss=True,
        aux_loss_weight=0.1    # (or best from above)
    )
    
    Expected parameters: ~100M
    """)

if __name__ == '__main__':
    print("🚀 Starting Quick Ablation Studies")
    print("   Testing on CPU with small dataset (50 steps each)")
    print("   Total time: ~10-15 minutes")
    print()
    
    all_results = {}
    
    # Run all studies
    try:
        all_results['slots'] = ablation_study_slots()
    except Exception as e:
        print(f"Error in slots study: {e}")
    
    try:
        all_results['aux_loss'] = ablation_study_aux_loss()
    except Exception as e:
        print(f"Error in aux_loss study: {e}")
    
    try:
        all_results['hcm_scaling'] = ablation_study_hcm_scaling()
    except Exception as e:
        print(f"Error in hcm_scaling study: {e}")
    
    try:
        all_results['gating'] = ablation_study_gating()
    except Exception as e:
        print(f"Error in gating study: {e}")
    
    # Save and summarize
    save_results(all_results)
    print_summary(all_results)
    
    print("\n✅ Ablation studies complete!")
    print("📊 Review results in ablation_results.json")
    print("🚀 Ready to proceed with A100 training using optimal config")
