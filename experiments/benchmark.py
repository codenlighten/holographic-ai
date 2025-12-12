"""
Benchmarking suite for HAMT
Compare against standard transformers and measure various metrics
"""
import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hamt import HAMTConfig, HAMTModel


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    model_name: str
    seq_length: int
    batch_size: int
    forward_time_ms: float
    backward_time_ms: float
    memory_mb: float
    throughput_tokens_per_sec: float
    parameters_m: float


class HAMTBenchmark:
    """Comprehensive benchmarking for HAMT"""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.results = []
    
    def benchmark_forward_pass(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        num_iterations: int = 10
    ) -> Tuple[float, float]:
        """
        Benchmark forward pass
        
        Returns:
            (avg_time_ms, std_time_ms)
        """
        model.eval()
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        # Actual benchmark
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(input_ids)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        return np.mean(times), np.std(times)
    
    def benchmark_backward_pass(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        num_iterations: int = 10
    ) -> Tuple[float, float]:
        """
        Benchmark backward pass
        
        Returns:
            (avg_time_ms, std_time_ms)
        """
        model.train()
        times = []
        
        # Warmup
        for _ in range(3):
            outputs = model(input_ids, labels=input_ids)
            loss = outputs['loss']
            loss.backward()
            model.zero_grad()
        
        # Actual benchmark
        for _ in range(num_iterations):
            start = time.perf_counter()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs['loss']
            loss.backward()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            model.zero_grad()
            times.append((end - start) * 1000)
        
        return np.mean(times), np.std(times)
    
    def measure_memory(self, model: nn.Module, input_ids: torch.Tensor) -> float:
        """
        Measure peak memory usage
        
        Returns:
            Peak memory in MB
        """
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(input_ids)
            
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            return peak_memory
        else:
            # For CPU, return model size as proxy
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
            return param_memory
    
    def run_benchmark(
        self,
        model: nn.Module,
        model_name: str,
        seq_lengths: List[int],
        batch_size: int = 4,
        num_iterations: int = 10
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark
        
        Args:
            model: Model to benchmark
            model_name: Name for results
            seq_lengths: List of sequence lengths to test
            batch_size: Batch size
            num_iterations: Number of iterations per test
            
        Returns:
            List of BenchmarkResult
        """
        vocab_size = model.config.vocab_size if hasattr(model, 'config') else 50257
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        
        results = []
        
        for seq_len in seq_lengths:
            print(f"Benchmarking {model_name} at seq_len={seq_len}...")
            
            # Create input
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
            
            # Forward pass
            fwd_time, fwd_std = self.benchmark_forward_pass(model, input_ids, num_iterations)
            
            # Backward pass
            bwd_time, bwd_std = self.benchmark_backward_pass(model, input_ids, num_iterations)
            
            # Memory
            memory_mb = self.measure_memory(model, input_ids)
            
            # Throughput (tokens per second)
            total_time_sec = (fwd_time + bwd_time) / 1000
            total_tokens = batch_size * seq_len
            throughput = total_tokens / total_time_sec
            
            result = BenchmarkResult(
                model_name=model_name,
                seq_length=seq_len,
                batch_size=batch_size,
                forward_time_ms=fwd_time,
                backward_time_ms=bwd_time,
                memory_mb=memory_mb,
                throughput_tokens_per_sec=throughput,
                parameters_m=param_count
            )
            
            results.append(result)
            self.results.append(result)
            
            print(f"  Forward: {fwd_time:.2f}±{fwd_std:.2f} ms")
            print(f"  Backward: {bwd_time:.2f}±{bwd_std:.2f} ms")
            print(f"  Memory: {memory_mb:.2f} MB")
            print(f"  Throughput: {throughput:.0f} tokens/sec\n")
        
        return results
    
    def save_results(self, output_file: str):
        """Save benchmark results to JSON"""
        results_dict = [
            {
                'model_name': r.model_name,
                'seq_length': r.seq_length,
                'batch_size': r.batch_size,
                'forward_time_ms': r.forward_time_ms,
                'backward_time_ms': r.backward_time_ms,
                'memory_mb': r.memory_mb,
                'throughput': r.throughput_tokens_per_sec,
                'parameters_m': r.parameters_m
            }
            for r in self.results
        ]
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_comparison(self):
        """Print comparison table"""
        if not self.results:
            print("No results to compare")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON")
        print("="*80)
        print(f"{'Model':<20} {'SeqLen':<10} {'Fwd(ms)':<12} {'Bwd(ms)':<12} {'Mem(MB)':<12} {'Throughput':<15}")
        print("-"*80)
        
        for result in self.results:
            print(f"{result.model_name:<20} {result.seq_length:<10} "
                  f"{result.forward_time_ms:<12.2f} {result.backward_time_ms:<12.2f} "
                  f"{result.memory_mb:<12.2f} {result.throughput_tokens_per_sec:<15.0f}")
        
        print("="*80)


def benchmark_hamt_vs_baseline(
    hidden_dim: int = 256,
    hcm_dim: int = 1024,
    num_layers: int = 4,
    seq_lengths: List[int] = [64, 128, 256, 512],
    batch_size: int = 4,
    device: str = 'cpu'
):
    """
    Benchmark HAMT against baseline configuration
    """
    benchmark = HAMTBenchmark(device=device)
    
    # Create HAMT model
    print("Creating HAMT model...")
    hamt_config = HAMTConfig(
        hidden_dim=hidden_dim,
        hcm_dim=hcm_dim,
        num_layers=num_layers,
        num_slots=8,
        num_attention_heads=8,
        vocab_size=5000,
        binding_type="elementwise",
        use_auxiliary_loss=False  # Disable for benchmarking
    )
    
    hamt_model = HAMTModel(hamt_config).to(device)
    
    # Benchmark HAMT
    print("\n" + "="*80)
    print("BENCHMARKING HAMT")
    print("="*80 + "\n")
    
    hamt_results = benchmark.run_benchmark(
        hamt_model,
        "HAMT",
        seq_lengths,
        batch_size=batch_size,
        num_iterations=5
    )
    
    # Print comparison
    benchmark.print_comparison()
    
    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    benchmark.save_results(output_dir / "benchmark_results.json")
    
    return benchmark


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark HAMT")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hcm_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_lengths", type=str, default="64,128,256,512",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iterations", type=int, default=5)
    
    args = parser.parse_args()
    
    seq_lengths = [int(x) for x in args.seq_lengths.split(',')]
    
    print("Starting HAMT Benchmark")
    print(f"Device: {args.device}")
    print(f"Model: {args.hidden_dim}h, {args.hcm_dim}hcm, {args.num_layers}L")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Batch size: {args.batch_size}\n")
    
    benchmark_hamt_vs_baseline(
        hidden_dim=args.hidden_dim,
        hcm_dim=args.hcm_dim,
        num_layers=args.num_layers,
        seq_lengths=seq_lengths,
        batch_size=args.batch_size,
        device=args.device
    )
