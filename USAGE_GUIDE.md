# HAMT Complete Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Training](#training)
3. [Evaluation](#evaluation)
4. [Experimentation](#experimentation)
5. [Advanced Topics](#advanced-topics)

---

## Quick Start

### 1. Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify installation
python -c "from hamt import HAMTModel; print('✅ HAMT installed correctly')"
```

### 2. Run the Demo

```bash
python experiments/demo.py
```

This will:
- Create a small HAMT model (21M params)
- Run a forward pass
- Test generation
- Display results

### 3. Run Tests

```bash
pytest tests/ -v
```

Expected: All 5 tests passing ✅

---

## Training

### Quick Training (500 steps, ~10 minutes)

```bash
python experiments/quick_train.py \
    --num_steps 500 \
    --batch_size 8 \
    --log_interval 50
```

**Output:**
- Model checkpoint: `checkpoints/quick_train/hamt_quick_500steps.pt`
- Training info: `checkpoints/quick_train/training_info.json`
- Final loss displayed

### Full Training (on WikiText-2)

```bash
python experiments/train.py \
    --hidden_dim 512 \
    --hcm_dim 2048 \
    --num_layers 8 \
    --num_slots 8 \
    --batch_size 16 \
    --num_epochs 20 \
    --output_dir ./checkpoints/wikitext
```

**Parameters:**
- `--hidden_dim`: Transformer hidden dimension
- `--hcm_dim`: Holographic memory dimension (typically 4x hidden_dim)
- `--num_layers`: Number of transformer layers
- `--num_slots`: Number of memory slots (8 recommended)
- `--binding_type`: `"elementwise"` or `"circular_conv"`
- `--use_aux_loss`: Enable auxiliary reconstruction loss
- `--aux_loss_weight`: Weight for auxiliary loss (default: 0.05)

### Training Tips

1. **Start Small**: Begin with 256 hidden_dim, 4 layers
2. **Monitor Losses**: Both main loss and aux loss should decrease
3. **Batch Size**: Adjust based on available RAM (4-16 typical)
4. **Learning Rate**: 3e-4 is a good starting point
5. **Aux Loss**: Helps memory fidelity, keep weight around 0.01-0.1

---

## Evaluation

### Analyze a Trained Model

```bash
python experiments/evaluate.py \
    --checkpoint checkpoints/quick_train/hamt_quick_500steps.pt \
    --test_text "Your test text here" \
    --max_new_tokens 30 \
    --temperature 0.8
```

**Outputs:**
- Memory state visualization: `analysis_output/memory_analysis.png`
- Generation results: `analysis_output/generation_results.json`

### What to Look For

1. **Memory State Norms**: Should be relatively uniform across slots
2. **Slot Utilization**: All slots should be used (not just 1-2)
3. **Generation Quality**: Should improve with training steps
4. **Layer-wise Patterns**: Upper layers often have higher norms

### Custom Prompts

```bash
python experiments/evaluate.py \
    --checkpoint your_checkpoint.pt \
    --prompts "First prompt|Second prompt|Third prompt" \
    --max_new_tokens 50 \
    --temperature 1.0
```

---

## Experimentation

### Interactive Notebook

Open the Jupyter notebook for hands-on exploration:

```bash
jupyter notebook notebooks/hamt_exploration.ipynb
```

**Notebook Contents:**
1. Holographic memory basics (binding/unbinding)
2. Memory superposition tests (1-slot vs 8-slot)
3. Full model forward pass
4. Memory state visualization
5. Generation examples
6. Complexity analysis (HAMT vs standard transformer)

### Key Experiments to Try

#### 1. **Binding Type Comparison**

Train two models with different binding:

```bash
# Elementwise binding (faster)
python experiments/quick_train.py --binding_type elementwise --num_steps 500

# Circular convolution (better unbinding)
python experiments/quick_train.py --binding_type circular_conv --num_steps 500
```

Compare reconstruction quality in aux_loss.

#### 2. **Slot Count Ablation**

```bash
# Test with different slot counts
for slots in 1 4 8 16; do
    python experiments/quick_train.py --num_slots $slots --num_steps 300
done
```

More slots → better capacity but more parameters.

#### 3. **HCM Dimension Scaling**

```bash
# Test different HCM dimensions
python experiments/quick_train.py --hcm_dim 512 --num_steps 300
python experiments/quick_train.py --hcm_dim 2048 --num_steps 300
python experiments/quick_train.py --hcm_dim 4096 --num_steps 300
```

Larger HCM → better holographic capacity.

#### 4. **Long Context Testing**

```bash
# Train with longer sequences
python experiments/train.py --max_length 1024 --num_epochs 5
```

HAMT should maintain constant memory unlike standard transformers.

---

## Advanced Topics

### Custom Datasets

Modify `experiments/train.py` to load your dataset:

```python
from datasets import load_dataset

# Load custom dataset
dataset = load_dataset("your_dataset", split="train")

# Or load local files
dataset = load_dataset("text", data_files={"train": "your_file.txt"})
```

### Model Architecture Modifications

#### Add Hierarchical Memory

In `src/hamt/layers.py`, implement fast/slow memory:

```python
class HierarchicalHAMTLayer(HAMTLayer):
    def __init__(self, config):
        super().__init__(config)
        self.fast_memory = HolographicMemory(...)
        self.slow_memory = HolographicMemory(...)
        self.consolidation_gate = nn.Linear(...)
```

#### Custom Retrieval Heads

Modify `RetrievalHead` in `layers.py`:

```python
class CustomRetrievalHead(nn.Module):
    def __init__(self, ...):
        # Add attention over slots
        self.slot_attention = MultiheadAttention(...)
```

### Performance Optimization

#### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(input_ids, labels=input_ids)
    loss = outputs['loss']

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Gradient Checkpointing

In `model.py`:

```python
from torch.utils.checkpoint import checkpoint

# In HAMTBlock.forward()
hidden_states = checkpoint(self.hamt_layer, hidden_states, ...)
```

### Distributed Training

Use Accelerate for multi-GPU:

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

---

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce `batch_size`
- Reduce `max_length`
- Use gradient checkpointing

**2. High Aux Loss**
- Increase `hcm_dim`
- Increase `num_slots`
- Adjust `aux_loss_weight`

**3. Poor Generation**
- Train longer (more steps/epochs)
- Increase model size
- Use larger dataset
- Adjust temperature

**4. Memory Saturation**
- Check `decay_rate` (should be > 0)
- Verify gating is working
- Reduce superposition rate

### Debug Mode

Add verbose logging:

```python
# In experiments/train.py
import logging
logging.basicConfig(level=logging.DEBUG)

# In model forward pass
print(f"HCM norms: {torch.norm(hcm_state, dim=-1)}")
```

---

## Benchmarking

### FLOPs Comparison

```python
from hamt.utils import compute_flops_per_token

config = HAMTConfig(...)
flops = compute_flops_per_token(config, seq_len=1024)

print(f"Standard Transformer: {flops['standard_gflops']:.2f} GFLOPs")
print(f"HAMT: {flops['hamt_gflops']:.2f} GFLOPs")
print(f"Speedup: {flops['reduction_ratio']:.2f}x")
```

### Memory Usage

```python
import torch

# Before forward
torch.cuda.reset_peak_memory_stats()

# Run model
outputs = model(input_ids)

# After forward
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak GPU memory: {peak_memory:.2f} GB")
```

---

## Next Steps

1. **Train on Real Data**: Use WikiText-2, BooksCorpus, or your domain data
2. **Evaluate Downstream**: Fine-tune on GLUE/SuperGLUE tasks
3. **Long Context**: Test on >4096 token sequences
4. **Publish Results**: Compare with baseline transformers

## Resources

- **Paper**: `holographic_associative_memory_transformers_...md`
- **Tests**: `tests/test_hamt.py`
- **Examples**: `experiments/demo.py`, `experiments/quick_train.py`
- **Analysis**: `experiments/evaluate.py`

---

## Citation

```bibtex
@article{hamt2025,
  title={Holographic Associative Memory Transformers for Energy-Efficient LLMs},
  author={NeuroLab AI Syndicate},
  year={2025}
}
```

---

**Happy experimenting with HAMT!** 🚀

For questions or issues, check the tests and demo scripts for working examples.
