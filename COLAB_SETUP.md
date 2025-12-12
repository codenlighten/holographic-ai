# HAMT A100 Training - Google Colab Setup

This notebook sets up the HAMT training environment on Google Colab with A100 GPU.

## Quick Start

```python
# 1. Check GPU
!nvidia-smi
```

## Installation

```python
# 2. Install dependencies
!pip install torch transformers datasets wandb tqdm

# 3. Clone repository (or upload files)
# Option A: From GitHub
!git clone https://github.com/YOUR_USERNAME/holographic-ai-training.git
%cd holographic-ai-training

# Option B: Upload files directly
# - Use Colab's file upload
# - Or mount Google Drive
```

## Setup

```python
# 4. Import libraries
import sys
sys.path.append('/content/holographic-ai-training')

import torch
from train_gpu import main
import argparse

# 5. Verify GPU
print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"   CUDA: {torch.version.cuda}")
```

## Weights & Biases (Optional but Recommended)

```python
# 6. Setup W&B for experiment tracking
import wandb
wandb.login()  # Enter your API key when prompted
```

## Training Configurations

### Configuration 1: Quick Test (50M parameters)

```python
# Test run to validate setup (~30 minutes)
!python train_gpu.py \
    --hidden_dim 384 \
    --hcm_dim 1536 \
    --num_layers 6 \
    --num_slots 16 \
    --dataset wikitext-2 \
    --max_seq_length 512 \
    --batch_size 32 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --mixed_precision fp16 \
    --compile_model \
    --use_wandb \
    --output_dir outputs/test_50m \
    --run_name hamt-test-50m
```

### Configuration 2: Competitive 100M Model

```python
# Main training run (~4-6 hours)
!python train_gpu.py \
    --hidden_dim 512 \
    --hcm_dim 2048 \
    --num_layers 8 \
    --num_slots 16 \
    --dataset wikitext-103 \
    --max_seq_length 1024 \
    --batch_size 128 \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --mixed_precision bf16 \
    --compile_model \
    --use_wandb \
    --output_dir outputs/hamt_100m \
    --run_name hamt-100m-wikitext103
```

### Configuration 3: Ambitious 200M Model

```python
# Large-scale training (~8-12 hours)
!python train_gpu.py \
    --hidden_dim 768 \
    --hcm_dim 3072 \
    --num_layers 12 \
    --num_slots 16 \
    --dataset wikitext-103 \
    --max_seq_length 1024 \
    --batch_size 64 \
    --num_epochs 10 \
    --learning_rate 2e-4 \
    --mixed_precision bf16 \
    --compile_model \
    --gradient_checkpointing \
    --use_wandb \
    --output_dir outputs/hamt_200m \
    --run_name hamt-200m-wikitext103
```

## Baseline Comparison

```python
# After training, compare with baselines
!python compare_baselines.py \
    --hamt_checkpoint outputs/hamt_100m/checkpoints/best.pt \
    --baselines distilgpt2 gpt2 \
    --dataset wikitext-2 \
    --device cuda \
    --output_dir comparison_results
```

## Download Results

```python
# Package results for download
!zip -r results.zip outputs/ comparison_results/ *.log

# Download via Colab
from google.colab import files
files.download('results.zip')
```

## Monitoring

### Check Training Progress

```python
# View latest loss
!tail -20 outputs/hamt_100m/training.log
```

### Monitor GPU Usage

```python
# Watch GPU utilization
!watch -n 1 nvidia-smi
```

### TensorBoard (if not using W&B)

```python
%load_ext tensorboard
%tensorboard --logdir outputs/
```

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
--batch_size 32  # instead of 128

# Enable gradient checkpointing
--gradient_checkpointing

# Reduce sequence length
--max_seq_length 512  # instead of 1024
```

### Slow Training

```python
# Ensure mixed precision is enabled
--mixed_precision bf16  # BF16 is faster on A100

# Compile model
--compile_model

# Increase batch size (if memory allows)
--batch_size 256
```

### Disconnection Issues

```python
# Save checkpoints frequently
--save_steps 500  # Save every 500 steps

# Resume from checkpoint (add this if training interrupted)
# TODO: Implement checkpoint resuming in train_gpu.py
```

## Expected Results

### 100M Model (6 hours training)

- **Perplexity**: ~25-28 on WikiText-103
- **Throughput**: 4000-6000 tokens/sec
- **Memory**: 6-8 GB VRAM
- **Quality**: Comparable to GPT-2 Small (124M)

### 200M Model (12 hours training)

- **Perplexity**: ~22-25 on WikiText-103
- **Throughput**: 3000-5000 tokens/sec
- **Memory**: 12-16 GB VRAM
- **Quality**: Approaching GPT-2 Medium (355M)

## Next Steps

1. ✅ Run quick test (Config 1) to validate setup
2. ✅ Run main training (Config 2) for competitive model
3. ✅ Compare with baselines
4. ✅ Download results
5. ✅ Update whitepaper with results
6. ✅ (Optional) Train larger model (Config 3)

## Tips for A100

- **BF16 is better than FP16** on A100 (native support)
- **Large batch sizes** (128-256) utilize GPU better
- **Compile model** for 10-20% speedup (PyTorch 2.0+)
- **Save frequently** in case of disconnection
- **Use W&B** for easy experiment tracking
- **Monitor GPU** - should stay at 80-100% utilization

## Estimated Costs

Google Colab A100 (as of Dec 2025):
- **Compute Units**: ~100-150 units for full run
- **Time**: 6-12 hours for competitive model
- **Pro subscription**: Recommended for long runs

## Success Criteria

✅ Training completes without OOM  
✅ Loss decreases steadily  
✅ Perplexity < 30 on test set  
✅ Throughput > 3000 tok/s  
✅ Competitive with or better than GPT-2 baselines  

Good luck! 🚀
