"""
Baseline Comparison Script
Fair comparison of HAMT against transformer baselines (GPT-2, DistilGPT-2, Pythia)
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm
import json
from pathlib import Path
import argparse

from src.hamt.model import HAMTModel
from src.hamt.config import HAMTConfig

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.encodings = []
        for text in tqdm(texts, desc="Tokenizing"):
            if len(text) < 50:
                continue
            enc = tokenizer(text, truncation=True, max_length=max_length,
                          padding='max_length', return_tensors='pt')
            self.encodings.append(enc['input_ids'].squeeze(0))
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return {'input_ids': self.encodings[idx]}

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate_model(model, dataloader, device='cuda'):
    """Evaluate model perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    start_time = time.time()
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        
        # Handle different model outputs
        try:
            outputs = model(input_ids, labels=input_ids)
            if isinstance(outputs, dict):
                loss = outputs['loss']
            else:
                loss = outputs.loss
        except Exception as e:
            print(f"Error during evaluation: {e}")
            continue
        
        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.size(0)
    
    elapsed = time.time() - start_time
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    throughput = total_tokens * input_ids.size(1) / elapsed  # tokens per second
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'throughput_tokens_per_sec': throughput,
        'time_seconds': elapsed
    }

@torch.no_grad()
def measure_memory(model, batch_size=16, seq_length=1024, device='cuda'):
    """Measure peak memory usage"""
    model.eval()
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    input_ids = torch.randint(0, 50257, (batch_size, seq_length)).to(device)
    
    try:
        outputs = model(input_ids)
        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
    except RuntimeError as e:
        print(f"OOM at batch_size={batch_size}, seq_length={seq_length}")
        peak_memory = None
    
    torch.cuda.empty_cache()
    
    return peak_memory

def measure_speed(model, batch_size=16, seq_length=1024, device='cuda', num_runs=10):
    """Measure forward/backward pass speed"""
    model.train()
    
    input_ids = torch.randint(0, 50257, (batch_size, seq_length)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(3):
        outputs = model(input_ids, labels=input_ids)
        if isinstance(outputs, dict):
            loss = outputs['loss']
        else:
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    # Measure forward
    forward_times = []
    for _ in range(num_runs):
        start = time.time()
        outputs = model(input_ids)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start)
    
    # Measure backward
    backward_times = []
    for _ in range(num_runs):
        outputs = model(input_ids, labels=input_ids)
        if isinstance(outputs, dict):
            loss = outputs['loss']
        else:
            loss = outputs.loss
        
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append(time.time() - start)
        optimizer.zero_grad()
    
    return {
        'forward_ms': sum(forward_times) / len(forward_times) * 1000,
        'backward_ms': sum(backward_times) / len(backward_times) * 1000,
        'tokens_per_sec': (batch_size * seq_length) / (sum(forward_times) / len(forward_times))
    }

def load_hamt_model(checkpoint_path, device='cuda'):
    """Load HAMT model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = HAMTConfig(**checkpoint['config'])
    model = HAMTModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config

def load_baseline_model(model_name, device='cuda'):
    """Load baseline transformer model"""
    print(f"Loading {model_name}...")
    
    if model_name == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif model_name == 'gpt2-medium':
        model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    elif model_name == 'distilgpt2':
        model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    elif 'pythia' in model_name:
        # e.g., 'pythia-160m', 'pythia-410m'
        hf_name = f"EleutherAI/{model_name}"
        model = AutoModelForCausalLM.from_pretrained(hf_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def compare_models(hamt_checkpoint, baseline_names, dataset_name='wikitext-2', 
                  max_samples=None, device='cuda', output_dir='comparison_results'):
    """Run comprehensive comparison"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset
    print(f"\n📚 Loading {dataset_name}...")
    if dataset_name == 'wikitext-2':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    elif dataset_name == 'wikitext-103':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    test_texts = [t for t in dataset['test']['text'] if len(t) > 50]
    if max_samples:
        test_texts = test_texts[:max_samples]
    
    # Results storage
    results = {}
    
    # Evaluate HAMT
    print("\n" + "="*80)
    print("🧠 Evaluating HAMT")
    print("="*80)
    
    hamt_model, hamt_config = load_hamt_model(hamt_checkpoint, device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    test_dataset = TextDataset(test_texts, tokenizer, max_length=1024)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    hamt_params = count_parameters(hamt_model)
    hamt_eval = evaluate_model(hamt_model, test_loader, device)
    hamt_memory = measure_memory(hamt_model, device=device)
    hamt_speed = measure_speed(hamt_model, device=device)
    
    results['HAMT'] = {
        'parameters': hamt_params,
        'loss': hamt_eval['loss'],
        'perplexity': hamt_eval['perplexity'],
        'throughput': hamt_eval['throughput_tokens_per_sec'],
        'eval_time': hamt_eval['time_seconds'],
        'peak_memory_gb': hamt_memory,
        'forward_ms': hamt_speed['forward_ms'],
        'backward_ms': hamt_speed['backward_ms'],
        'speed_tokens_per_sec': hamt_speed['tokens_per_sec']
    }
    
    print(f"  Parameters: {hamt_params:,}")
    print(f"  Perplexity: {hamt_eval['perplexity']:.2f}")
    print(f"  Throughput: {hamt_eval['throughput_tokens_per_sec']:.0f} tok/s")
    print(f"  Memory: {hamt_memory:.2f} GB")
    
    # Evaluate baselines
    for model_name in baseline_names:
        print("\n" + "="*80)
        print(f"🤖 Evaluating {model_name}")
        print("="*80)
        
        try:
            model, tokenizer = load_baseline_model(model_name, device)
            
            test_dataset = TextDataset(test_texts, tokenizer, max_length=1024)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            params = count_parameters(model)
            eval_results = evaluate_model(model, test_loader, device)
            memory = measure_memory(model, device=device)
            speed = measure_speed(model, device=device)
            
            results[model_name] = {
                'parameters': params,
                'loss': eval_results['loss'],
                'perplexity': eval_results['perplexity'],
                'throughput': eval_results['throughput_tokens_per_sec'],
                'eval_time': eval_results['time_seconds'],
                'peak_memory_gb': memory,
                'forward_ms': speed['forward_ms'],
                'backward_ms': speed['backward_ms'],
                'speed_tokens_per_sec': speed['tokens_per_sec']
            }
            
            print(f"  Parameters: {params:,}")
            print(f"  Perplexity: {eval_results['perplexity']:.2f}")
            print(f"  Throughput: {eval_results['throughput_tokens_per_sec']:.0f} tok/s")
            print(f"  Memory: {memory:.2f} GB")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Error evaluating {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Print comparison table
    print("\n" + "="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Params':<12} {'Perplexity':<12} {'Throughput':<15} {'Memory (GB)':<12}")
    print("-"*80)
    
    for model_name, metrics in results.items():
        if 'error' in metrics:
            continue
        params_str = f"{metrics['parameters']/1e6:.1f}M"
        ppl_str = f"{metrics['perplexity']:.2f}"
        throughput_str = f"{metrics['throughput']:.0f} tok/s"
        memory_str = f"{metrics['peak_memory_gb']:.2f}"
        print(f"{model_name:<20} {params_str:<12} {ppl_str:<12} {throughput_str:<15} {memory_str:<12}")
    
    # Calculate efficiency metrics
    print("\n" + "="*80)
    print("⚡ EFFICIENCY METRICS (relative to HAMT)")
    print("="*80)
    
    hamt_ppl = results['HAMT']['perplexity']
    hamt_params = results['HAMT']['parameters']
    hamt_throughput = results['HAMT']['throughput']
    
    for model_name, metrics in results.items():
        if model_name == 'HAMT' or 'error' in metrics:
            continue
        
        ppl_ratio = metrics['perplexity'] / hamt_ppl
        param_ratio = metrics['parameters'] / hamt_params
        throughput_ratio = hamt_throughput / metrics['throughput']
        
        print(f"\n{model_name}:")
        print(f"  Perplexity: {ppl_ratio:.2f}× {'worse' if ppl_ratio > 1 else 'better'}")
        print(f"  Parameters: {param_ratio:.2f}× more")
        print(f"  Speed: {throughput_ratio:.2f}× slower than HAMT")
    
    # Save results
    output_file = output_dir / 'comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare HAMT with baseline models')
    parser.add_argument('--hamt_checkpoint', type=str, required=True, help='Path to HAMT checkpoint')
    parser.add_argument('--baselines', type=str, nargs='+', 
                       default=['distilgpt2', 'gpt2'],
                       help='Baseline models to compare')
    parser.add_argument('--dataset', type=str, default='wikitext-2', 
                       choices=['wikitext-2', 'wikitext-103'])
    parser.add_argument('--max_samples', type=int, default=None, help='Limit test samples')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output_dir', type=str, default='comparison_results')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("🔬 Starting Baseline Comparison")
    print(f"   HAMT checkpoint: {args.hamt_checkpoint}")
    print(f"   Baselines: {', '.join(args.baselines)}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Device: {args.device}")
    
    results = compare_models(
        args.hamt_checkpoint,
        args.baselines,
        args.dataset,
        args.max_samples,
        args.device,
        args.output_dir
    )
    
    print("\n✅ Comparison complete!")

if __name__ == '__main__':
    main()
