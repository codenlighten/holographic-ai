"""
Training script for HAMT model
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from hamt import HAMTConfig, HAMTModel
from hamt.utils import count_parameters, get_lr_schedule, format_metrics, AverageMeter


def get_dataloaders(tokenizer, batch_size=8, max_length=512):
    """Load and prepare dataset"""
    # For initial testing, use a smaller dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    tokenized.set_format("torch")
    
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, config):
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            labels=input_ids,  # Next-token prediction
            return_aux_loss=config.use_auxiliary_loss
        )
        
        loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm
        )
        
        optimizer.step()
        scheduler.step()
        
        # Update meters
        loss_meter.update(loss.item())
        if "aux_loss" in outputs:
            aux_loss_meter.update(outputs["aux_loss"].item())
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "aux_loss": f"{aux_loss_meter.avg:.4f}" if aux_loss_meter.count > 0 else "N/A",
            "lr": f"{scheduler.get_last_lr()[0]:.6f}"
        })
    
    return {
        "loss": loss_meter.avg,
        "aux_loss": aux_loss_meter.avg if aux_loss_meter.count > 0 else None
    }


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                labels=input_ids,
                return_aux_loss=False
            )
            
            loss_meter.update(outputs["loss"].item())
    
    return {"loss": loss_meter.avg}


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create config
    config = HAMTConfig(
        hidden_dim=args.hidden_dim,
        hcm_dim=args.hcm_dim,
        num_layers=args.num_layers,
        num_slots=args.num_slots,
        binding_type=args.binding_type,
        use_auxiliary_loss=args.use_aux_loss,
        aux_loss_weight=args.aux_loss_weight,
        max_position_embeddings=args.max_length,
        bptt_horizon=args.bptt_horizon,
        dropout=args.dropout
    )
    
    print("Configuration:")
    print(config)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
    
    # Create model
    print("\nCreating model...")
    model = HAMTModel(config).to(device)
    
    # Print parameter count
    param_counts = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Total: {param_counts['total_millions']:.2f}M")
    print(f"  Trainable: {param_counts['trainable_millions']:.2f}M")
    
    # Create dataloaders
    print("\nLoading data...")
    train_dataloader = get_dataloaders(
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    num_training_steps = len(train_dataloader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_lr_schedule(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}\n")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, epoch, config
        )
        
        print(f"\nEpoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
        if train_metrics['aux_loss'] is not None:
            print(f"          Aux Loss: {train_metrics['aux_loss']:.4f}")
        
        # Save checkpoint
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            checkpoint_dir = Path(args.output_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f"hamt_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAMT model")
    
    # Model config
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--hcm_dim", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument("--binding_type", type=str, default="elementwise", choices=["elementwise", "circular_conv"])
    
    # Training config
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bptt_horizon", type=int, default=256)
    
    # Auxiliary loss
    parser.add_argument("--use_aux_loss", action="store_true", default=True)
    parser.add_argument("--aux_loss_weight", type=float, default=0.05)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()
    
    main(args)
