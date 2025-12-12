"""
Evaluation and analysis script for trained HAMT models
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hamt import HAMTModel


def load_model(checkpoint_path):
    """Load a trained HAMT model"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = HAMTModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def analyze_memory_states(model, input_text, tokenizer, device='cpu'):
    """Analyze HCM states during forward pass"""
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors='pt')['input_ids'].to(device)
    seq_len = input_ids.shape[1]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, return_aux_loss=False)
        hcm_states = outputs['hcm_states']
        logits = outputs['logits']
    
    # Analyze HCM states
    num_layers = len(hcm_states)
    num_slots = hcm_states[0].shape[1]
    
    # Compute norms
    layer_norms = []
    for hcm_state in hcm_states:
        norms = torch.norm(hcm_state[0], p=2, dim=-1).cpu().numpy()
        layer_norms.append(norms)
    
    layer_norms = np.array(layer_norms)  # [num_layers, num_slots]
    
    # Compute slot utilization (which slots are most active)
    slot_utilization = layer_norms.mean(axis=0)
    
    # Get predictions
    predicted_ids = torch.argmax(logits[0], dim=-1).cpu().numpy()
    predicted_tokens = [tokenizer.decode([tid]) for tid in predicted_ids]
    
    return {
        'layer_norms': layer_norms,
        'slot_utilization': slot_utilization,
        'predicted_tokens': predicted_tokens,
        'input_tokens': [tokenizer.decode([tid]) for tid in input_ids[0].cpu().numpy()],
        'seq_len': seq_len
    }


def visualize_analysis(analysis, save_path=None):
    """Create visualizations of the analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. HCM State Norms Heatmap
    ax = axes[0, 0]
    sns.heatmap(analysis['layer_norms'].T, annot=True, fmt='.2f', 
                cmap='viridis', ax=ax, cbar_kws={'label': 'L2 Norm'})
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Memory Slot', fontsize=11)
    ax.set_title('HCM State Norms Across Layers', fontsize=12, fontweight='bold')
    
    # 2. Slot Utilization
    ax = axes[0, 1]
    slots = np.arange(len(analysis['slot_utilization']))
    bars = ax.bar(slots, analysis['slot_utilization'], color='steelblue', alpha=0.7)
    ax.set_xlabel('Memory Slot', fontsize=11)
    ax.set_ylabel('Average Norm', fontsize=11)
    ax.set_title('Memory Slot Utilization', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight most/least used slots
    max_idx = np.argmax(analysis['slot_utilization'])
    min_idx = np.argmin(analysis['slot_utilization'])
    bars[max_idx].set_color('green')
    bars[min_idx].set_color('red')
    
    # 3. Token predictions (first 20 tokens)
    ax = axes[1, 0]
    display_len = min(20, analysis['seq_len'])
    x = np.arange(display_len)
    
    ax.text(0.05, 0.95, 'Input vs Predicted Tokens', 
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top')
    
    for i in range(display_len):
        input_tok = analysis['input_tokens'][i]
        pred_tok = analysis['predicted_tokens'][i] if i < len(analysis['predicted_tokens']) else '?'
        match = '✓' if input_tok == pred_tok else '✗'
        
        ax.text(0.05, 0.85 - i*0.035, f"{i:2d}: {input_tok:15s} → {pred_tok:15s} {match}",
                transform=ax.transAxes, fontsize=8, family='monospace')
    
    ax.axis('off')
    
    # 4. Layer-wise norm distribution
    ax = axes[1, 1]
    for layer_idx in range(analysis['layer_norms'].shape[0]):
        ax.plot(analysis['layer_norms'][layer_idx], 
                marker='o', label=f'Layer {layer_idx}', alpha=0.7)
    
    ax.set_xlabel('Memory Slot', fontsize=11)
    ax.set_ylabel('L2 Norm', fontsize=11)
    ax.set_title('Norm Distribution per Layer', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def generate_samples(model, tokenizer, prompts, max_new_tokens=50, temperature=0.8, device='cpu'):
    """Generate text from multiple prompts"""
    model = model.to(device)
    model.eval()
    
    results = []
    
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50
            )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        results.append({
            'prompt': prompt,
            'generated': generated_text,
            'length': len(generated[0])
        })
    
    return results


def main(args):
    print("="*70)
    print("HAMT Model Evaluation and Analysis")
    print("="*70)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  HCM dim: {config.hcm_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Slots: {config.num_slots}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Analyze memory states
    if args.analyze:
        print("\n" + "="*70)
        print("Memory State Analysis")
        print("="*70)
        
        test_text = args.test_text or "The quick brown fox jumps over the lazy dog."
        print(f"\nAnalyzing text: '{test_text}'")
        
        analysis = analyze_memory_states(model, test_text, tokenizer)
        
        print(f"\nResults:")
        print(f"  Sequence length: {analysis['seq_len']}")
        print(f"  Most used slot: {np.argmax(analysis['slot_utilization'])}")
        print(f"  Least used slot: {np.argmin(analysis['slot_utilization'])}")
        print(f"  Utilization range: [{analysis['slot_utilization'].min():.3f}, "
              f"{analysis['slot_utilization'].max():.3f}]")
        
        # Visualize
        save_path = Path(args.output_dir) / "memory_analysis.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_analysis(analysis, save_path=save_path)
    
    # Generate samples
    if args.generate:
        print("\n" + "="*70)
        print("Text Generation")
        print("="*70)
        
        prompts = [
            "The quick brown fox",
            "Machine learning is",
            "In the beginning",
            "Once upon a time"
        ]
        
        if args.prompts:
            prompts = args.prompts.split('|')
        
        print(f"\nGenerating from {len(prompts)} prompts...\n")
        
        results = generate_samples(
            model, tokenizer, prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Prompt: {result['prompt']}")
            print(f"   Generated: {result['generated']}")
            print(f"   Length: {result['length']} tokens\n")
        
        # Save results
        if args.output_dir:
            output_file = Path(args.output_dir) / "generation_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
    
    print("\n" + "="*70)
    print("✅ Evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HAMT model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--analyze", action="store_true", default=True,
                        help="Analyze memory states")
    parser.add_argument("--generate", action="store_true", default=True,
                        help="Generate text samples")
    parser.add_argument("--test_text", type=str, default=None,
                        help="Text to analyze (default: sample text)")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Generation prompts separated by |")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Generation temperature")
    parser.add_argument("--output_dir", type=str, default="./analysis_output",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    main(args)
