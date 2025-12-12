"""
Train Your Own LLM - Simple Script
Train a 10M parameter HAMT language model on WikiText-2
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import time
from pathlib import Path
import json
import sys

sys.path.insert(0, 'src')

from hamt import HAMTConfig, HAMTModel


class TextDataset(torch.utils.data.Dataset):
    """Simple dataset for language modeling"""
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            if len(text.strip()) > 0:
                tokens = tokenizer.encode(text, add_special_tokens=True)
                if len(tokens) > 10:  # Skip very short texts
                    self.examples.append(tokens[:max_length])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        return torch.tensor(tokens[:self.max_length])


def train_llm(
    model_size="small",
    num_epochs=5,
    batch_size=4,
    learning_rate=3e-4,
    output_dir="checkpoints/my_llm"
):
    """
    Train your own LLM!
    
    Args:
        model_size: "tiny", "small", "medium"
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Where to save model
    """
    
    print("="*80)
    print("TRAINING YOUR OWN LLM WITH HAMT")
    print("="*80)
    
    # Model configurations
    configs = {
        "tiny": {
            "hidden_dim": 256,
            "hcm_dim": 768,
            "num_layers": 4,
            "num_slots": 8,
            "num_attention_heads": 8,
            "params": "~8M"
        },
        "small": {
            "hidden_dim": 384,
            "hcm_dim": 1024,
            "num_layers": 6,
            "num_slots": 12,
            "num_attention_heads": 12,
            "params": "~15M"
        },
        "medium": {
            "hidden_dim": 512,
            "hcm_dim": 1536,
            "num_layers": 8,
            "num_slots": 16,
            "num_attention_heads": 8,
            "params": "~30M"
        }
    }
    
    cfg = configs[model_size]
    print(f"\n📊 Model Configuration: {model_size.upper()}")
    print(f"   Hidden dim: {cfg['hidden_dim']}")
    print(f"   HCM dim: {cfg['hcm_dim']}")
    print(f"   Layers: {cfg['num_layers']}")
    print(f"   Slots: {cfg['num_slots']}")
    print(f"   Est. parameters: {cfg['params']}")
    
    # Create config
    config = HAMTConfig(
        hidden_dim=cfg['hidden_dim'],
        hcm_dim=cfg['hcm_dim'],
        num_layers=cfg['num_layers'],
        num_slots=cfg['num_slots'],
        num_attention_heads=cfg['num_attention_heads'],
        intermediate_dim=cfg['hidden_dim'] * 4,
        vocab_size=50257,
        max_position_embeddings=512,
        dropout=0.1,
        use_gating=True,
        use_auxiliary_loss=True,
        auxiliary_loss_weight=0.1
    )
    
    # Create model
    print("\n🏗️  Creating model...")
    model = HAMTModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params/1e6:.2f}M")
    
    # Load dataset
    print("\n📚 Loading WikiText-2 dataset...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        print(f"   Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        print("   Using dummy data for demonstration...")
        # Create dummy dataset
        dataset = [{"text": f"This is example text number {i}. " * 10} 
                   for i in range(1000)]
    
    # Tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    texts = [item['text'] for item in dataset] if isinstance(dataset, list) else dataset['text']
    train_dataset = TextDataset(texts[:10000], tokenizer, max_length=128)  # Use subset for speed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"   Training examples: {len(train_dataset)}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training
    print("\n" + "="*80)
    print("TRAINING STARTED")
    print("="*80)
    
    model.train()
    total_steps = 0
    start_time = time.time()
    
    training_history = {
        "model_size": model_size,
        "config": config.__dict__,
        "total_params": total_params,
        "losses": [],
        "aux_losses": []
    }
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_aux_loss = 0
        num_batches = 0
        
        print(f"\n📖 Epoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        for batch_idx, input_ids in enumerate(train_loader):
            # Forward pass
            outputs = model(input_ids, labels=input_ids, return_aux_loss=True)
            loss = outputs['loss']
            aux_loss = outputs.get('aux_loss', torch.tensor(0.0))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            if aux_loss is not None:
                epoch_aux_loss += aux_loss.item()
            num_batches += 1
            total_steps += 1
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / num_batches
                avg_aux = epoch_aux_loss / num_batches
                elapsed = time.time() - start_time
                print(f"   Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | Aux: {avg_aux:.4f} | "
                      f"Time: {elapsed/60:.1f}min")
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_aux = epoch_aux_loss / num_batches
        
        training_history["losses"].append(avg_epoch_loss)
        training_history["aux_losses"].append(avg_epoch_aux)
        
        print(f"\n✅ Epoch {epoch+1} Complete:")
        print(f"   Average Loss: {avg_epoch_loss:.4f}")
        print(f"   Average Aux Loss: {avg_epoch_aux:.4f}")
        
        # Save checkpoint
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = output_path / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'loss': avg_epoch_loss,
            'aux_loss': avg_epoch_aux
        }, checkpoint_path)
        print(f"   Saved: {checkpoint_path}")
    
    # Final save
    total_time = time.time() - start_time
    
    final_path = output_path / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_history': training_history,
        'total_time': total_time
    }, final_path)
    
    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            **training_history,
            "total_time_minutes": total_time / 60,
            "total_steps": total_steps
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE! 🎉")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Model: {model_size} ({total_params/1e6:.2f}M params)")
    print(f"   Epochs: {num_epochs}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Final loss: {training_history['losses'][-1]:.4f}")
    print(f"   Final aux loss: {training_history['aux_losses'][-1]:.4f}")
    print(f"\n💾 Saved to: {output_dir}/")
    print(f"   - final_model.pt")
    print(f"   - checkpoint_epoch_*.pt")
    print(f"   - training_history.json")
    
    print(f"\n🚀 Test your model:")
    print(f"   python experiments/evaluate.py {final_path}")
    
    return model, config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train your own LLM")
    parser.add_argument("--size", type=str, default="tiny", 
                        choices=["tiny", "small", "medium"],
                        help="Model size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--output", type=str, default="checkpoints/my_llm",
                        help="Output directory")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CREATE YOUR OWN LLM")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Size: {args.size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {args.output}")
    print()
    
    train_llm(
        model_size=args.size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output
    )
