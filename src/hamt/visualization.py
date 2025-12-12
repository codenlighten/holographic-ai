"""
Visualization tools for HAMT
Create plots for memory states, slot attention, and retrieval patterns
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_memory_evolution(
    memory_states: List[torch.Tensor],
    slot_indices: Optional[List[int]] = None,
    title: str = "Memory State Evolution",
    output_file: Optional[str] = None
):
    """
    Plot how memory states evolve over time
    
    Args:
        memory_states: List of memory tensors [num_slots, hcm_dim]
        slot_indices: Which slots to plot (None = all)
        title: Plot title
        output_file: Save path (None = show)
    """
    num_steps = len(memory_states)
    num_slots = memory_states[0].shape[0]
    
    if slot_indices is None:
        slot_indices = list(range(num_slots))
    
    # Convert to numpy
    memory_arrays = [m.detach().cpu().numpy() for m in memory_states]
    
    # Plot each slot's norm over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Norm of each slot over time
    ax = axes[0]
    for slot_idx in slot_indices:
        norms = [np.linalg.norm(m[slot_idx]) for m in memory_arrays]
        ax.plot(norms, label=f'Slot {slot_idx}', marker='o', markersize=3)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Memory Norm')
    ax.set_title(f'{title} - Slot Norms')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap of first slot over time
    ax = axes[1]
    if memory_arrays:
        # Take first 50 dimensions for visualization
        slot_evolution = np.array([m[slot_indices[0], :50] for m in memory_arrays])
        
        sns.heatmap(
            slot_evolution.T,
            cmap='coolwarm',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Value'}
        )
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Memory Dimension')
        ax.set_title(f'Slot {slot_indices[0]} Memory Heatmap (first 50 dims)')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_slot_attention(
    attention_weights: torch.Tensor,
    slot_names: Optional[List[str]] = None,
    title: str = "Slot Attention Weights",
    output_file: Optional[str] = None
):
    """
    Plot attention weights across memory slots
    
    Args:
        attention_weights: Tensor of shape [batch, seq_len, num_slots] or [seq_len, num_slots]
        slot_names: Names for each slot
        title: Plot title
        output_file: Save path
    """
    # Handle different input shapes
    if attention_weights.dim() == 3:
        # Take first batch
        weights = attention_weights[0].detach().cpu().numpy()
    else:
        weights = attention_weights.detach().cpu().numpy()
    
    seq_len, num_slots = weights.shape
    
    if slot_names is None:
        slot_names = [f'Slot {i}' for i in range(num_slots)]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Heatmap
    ax = axes[0]
    sns.heatmap(
        weights.T,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'},
        yticklabels=slot_names
    )
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Memory Slot')
    ax.set_title(f'{title} - Heatmap')
    
    # Plot 2: Line plot of attention per slot
    ax = axes[1]
    for slot_idx in range(num_slots):
        ax.plot(weights[:, slot_idx], label=slot_names[slot_idx], marker='o', markersize=2)
    
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'{title} - Per-Slot Attention')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_retrieval_similarity(
    retrieved: torch.Tensor,
    query: torch.Tensor,
    title: str = "Retrieval Similarity",
    output_file: Optional[str] = None
):
    """
    Plot similarity between query and retrieved memory
    
    Args:
        retrieved: Retrieved memory [batch, seq_len, hidden_dim]
        query: Query vectors [batch, seq_len, hidden_dim]
        title: Plot title
        output_file: Save path
    """
    # Compute cosine similarity
    retrieved_norm = torch.nn.functional.normalize(retrieved, dim=-1)
    query_norm = torch.nn.functional.normalize(query, dim=-1)
    
    similarity = (retrieved_norm * query_norm).sum(dim=-1)  # [batch, seq_len]
    
    # Take first batch
    sim = similarity[0].detach().cpu().numpy()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(sim, marker='o', markersize=4, linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(range(len(sim)), sim, alpha=0.3)
    
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_memory_capacity(
    num_items_list: List[int],
    recall_accuracies: List[float],
    title: str = "Memory Capacity Analysis",
    output_file: Optional[str] = None
):
    """
    Plot memory capacity vs recall accuracy
    
    Args:
        num_items_list: List of number of items stored
        recall_accuracies: Corresponding recall accuracies
        title: Plot title
        output_file: Save path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(num_items_list, recall_accuracies, marker='o', markersize=8, 
            linewidth=2, label='Recall Accuracy')
    
    # Add 50% threshold line
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Threshold')
    
    ax.set_xlabel('Number of Items Stored')
    ax.set_ylabel('Recall Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_comparative_performance(
    seq_lengths: List[int],
    hamt_times: List[float],
    baseline_times: List[float],
    metric_name: str = "Forward Pass Time (ms)",
    title: str = "HAMT vs Baseline Performance",
    output_file: Optional[str] = None
):
    """
    Plot comparative performance between HAMT and baseline
    
    Args:
        seq_lengths: List of sequence lengths
        hamt_times: HAMT metrics
        baseline_times: Baseline metrics
        metric_name: Name of the metric
        title: Plot title
        output_file: Save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute values
    ax = axes[0]
    ax.plot(seq_lengths, hamt_times, marker='o', markersize=8, 
            linewidth=2, label='HAMT')
    ax.plot(seq_lengths, baseline_times, marker='s', markersize=8, 
            linewidth=2, label='Baseline')
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{title} - Absolute')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Plot 2: Speedup
    ax = axes[1]
    speedups = [b / h if h > 0 else 1.0 for b, h in zip(baseline_times, hamt_times)]
    
    ax.plot(seq_lengths, speedups, marker='o', markersize=8, 
            linewidth=2, color='green')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Speedup')
    ax.fill_between(seq_lengths, speedups, 1, alpha=0.3, color='green')
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Speedup (Baseline / HAMT)')
    ax.set_title(f'{title} - Speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def create_analysis_dashboard(
    memory_states: List[torch.Tensor],
    attention_weights: torch.Tensor,
    training_loss: List[float],
    output_dir: str = "analysis_output"
):
    """
    Create comprehensive analysis dashboard with multiple plots
    
    Args:
        memory_states: List of memory state tensors
        attention_weights: Attention weight tensor
        training_loss: List of loss values
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Creating analysis dashboard...")
    
    # Plot 1: Memory evolution
    plot_memory_evolution(
        memory_states,
        title="Memory State Evolution",
        output_file=str(output_path / "memory_evolution.png")
    )
    
    # Plot 2: Slot attention
    plot_slot_attention(
        attention_weights,
        title="Slot Attention Weights",
        output_file=str(output_path / "slot_attention.png")
    )
    
    # Plot 3: Training loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(training_loss, linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "training_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Dashboard saved to {output_dir}/")


if __name__ == "__main__":
    # Demo with synthetic data
    print("Creating demo visualizations...")
    
    output_dir = Path("visualization_demo")
    output_dir.mkdir(exist_ok=True)
    
    # Synthetic memory states
    num_steps = 20
    num_slots = 8
    hcm_dim = 256
    
    memory_states = []
    for t in range(num_steps):
        # Simulate memory evolution
        state = torch.randn(num_slots, hcm_dim) * (1 + 0.1 * t)
        memory_states.append(state)
    
    plot_memory_evolution(
        memory_states,
        slot_indices=[0, 1, 2, 3],
        output_file=str(output_dir / "demo_memory_evolution.png")
    )
    
    # Synthetic attention weights
    seq_len = 30
    attention = torch.softmax(torch.randn(seq_len, num_slots), dim=-1)
    
    plot_slot_attention(
        attention,
        output_file=str(output_dir / "demo_slot_attention.png")
    )
    
    # Synthetic capacity analysis
    num_items = [1, 2, 5, 10, 20, 50, 100]
    accuracies = [0.99, 0.98, 0.95, 0.90, 0.75, 0.55, 0.35]
    
    plot_memory_capacity(
        num_items,
        accuracies,
        output_file=str(output_dir / "demo_capacity.png")
    )
    
    # Synthetic performance comparison
    seq_lengths = [64, 128, 256, 512, 1024]
    hamt_times = [10, 12, 15, 20, 28]
    baseline_times = [15, 30, 60, 120, 240]
    
    plot_comparative_performance(
        seq_lengths,
        hamt_times,
        baseline_times,
        output_file=str(output_dir / "demo_performance.png")
    )
    
    print(f"\nDemo visualizations saved to {output_dir}/")
