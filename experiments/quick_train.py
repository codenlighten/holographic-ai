"""
Quick training script for HAMT on a small dataset
This is a minimal training loop for fast experimentation
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hamt import HAMTConfig, HAMTModel
from hamt.utils import count_parameters, AverageMeter


class SimpleTextDataset(Dataset):
    """Simple dataset that tokenizes text on the fly"""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def get_sample_data():
    """Generate or load sample text data for training"""
    # Sample texts for demonstration
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Holographic memory systems offer constant time complexity.",
        "Vector symbolic architectures enable distributed representations.",
        "Neural networks learn patterns from data through backpropagation.",
        "Transformers revolutionized natural language processing tasks.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Language models generate coherent text by predicting next tokens.",
        "Deep learning requires large amounts of training data.",
        "Optimization algorithms minimize loss functions during training.",
    ] * 20  # Repeat for more training data
    
    return texts


def train_quick(args):
    """Quick training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create config
    config = HAMTConfig(
        hidden_dim=args.hidden_dim,
        hcm_dim=args.hcm_dim,
        num_layers=args.num_layers,
        num_slots=args.num_slots,
        num_attention_heads=args.num_attention_heads,
        binding_type=args.binding_type,
        use_auxiliary_loss=args.use_aux_loss,
        aux_loss_weight=args.aux_loss_weight,
        max_position_embeddings=args.max_length,
        dropout=args.dropout
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
    
    # Get sample data
    print("Loading data...")
    texts = get_sample_data()
    dataset = SimpleTextDataset(texts, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} samples\n")
    
    # Create model
    print("Creating model...")
    model = HAMTModel(config).to(device)
    
    param_counts = count_parameters(model)
    print(f"Model parameters: {param_counts['total_millions']:.2f}M")
    print(f"Trainable: {param_counts['trainable_millions']:.2f}M\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Training loop
    print(f"Training for {args.num_steps} steps...\n")
    model.train()
    
    loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    
    step = 0
    pbar = tqdm(total=args.num_steps, desc="Training")
    
    while step < args.num_steps:
        for batch in dataloader:
            if step >= args.num_steps:
                break
            
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                labels=input_ids,
                return_aux_loss=config.use_auxiliary_loss
            )
            
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update meters
            loss_meter.update(loss.item())
            if 'aux_loss' in outputs:
                aux_loss_meter.update(outputs['aux_loss'].item())
            
            # Update progress
            step += 1
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'aux': f'{aux_loss_meter.avg:.4f}' if aux_loss_meter.count > 0 else 'N/A'
            })
            
            # Log periodically
            if step % args.log_interval == 0:
                print(f"\nStep {step}/{args.num_steps}")
                print(f"  Loss: {loss_meter.avg:.4f}")
                if aux_loss_meter.count > 0:
                    print(f"  Aux Loss: {aux_loss_meter.avg:.4f}")
    
    pbar.close()
    
    # Save model
    if args.save_path:
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'step': step,
            'loss': loss_meter.avg
        }
        
        save_file = save_dir / f'hamt_quick_{step}steps.pt'
        torch.save(checkpoint, save_file)
        print(f"\n✅ Model saved to {save_file}")
    
    # Test generation
    print("\n" + "="*60)
    print("Testing generation...")
    print("="*60)
    
    model.eval()
    test_prompt = "The quick brown fox"
    input_ids = tokenizer(test_prompt, return_tensors='pt')['input_ids'].to(device)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=30,
            temperature=0.8,
            top_k=50
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nPrompt: {test_prompt}")
    print(f"Generated: {generated_text}\n")
    
    # Save training info
    if args.save_path:
        info = {
            'config': config.__dict__,
            'final_loss': loss_meter.avg,
            'final_aux_loss': aux_loss_meter.avg if aux_loss_meter.count > 0 else None,
            'steps': step,
            'parameters_M': param_counts['total_millions']
        }
        info_file = save_dir / 'training_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Training info saved to {info_file}")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick HAMT training")
    
    # Model config
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hcm_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--binding_type", type=str, default="elementwise")
    
    # Training config
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Aux loss
    parser.add_argument("--use_aux_loss", action="store_true", default=True)
    parser.add_argument("--aux_loss_weight", type=float, default=0.05)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="./checkpoints/quick_train")
    
    args = parser.parse_args()
    
    train_quick(args)
