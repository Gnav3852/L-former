"""
Training script for L-Former
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path

from config import ModelConfig
from modeling.lformer import LFormer
from data.toy_planning_dataset import create_dataloader


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_reasoning_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        tool_labels = batch["tool_labels"].to(device) if batch["tool_labels"] is not None else None
        values = batch["values"].to(device) if batch["values"] is not None else None
        plan_labels = batch["plan_labels"].to(device) if batch["plan_labels"] is not None else None
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            tool_labels=tool_labels,
            values=values,
            plan_labels=plan_labels
        )
        
        # Get losses
        losses = outputs["losses"]
        total_loss += losses["total"].item()
        total_lm_loss += losses["lm"].item()
        total_reasoning_loss += losses["reasoning"].item()
        
        # Backward pass
        losses["total"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Log every 10 batches
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Total Loss: {losses['total']:.4f}, "
                  f"LM Loss: {losses['lm']:.4f}, Reasoning Loss: {losses['reasoning']:.4f}")
    
    # Return average losses
    num_batches = len(dataloader)
    return {
        "total": total_loss / num_batches,
        "lm": total_lm_loss / num_batches,
        "reasoning": total_reasoning_loss / num_batches
    }


def evaluate(model, dataloader, device, config):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_lm_loss = 0
    total_reasoning_loss = 0
    
    tool_correct = 0
    tool_total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            tool_labels = batch["tool_labels"].to(device) if batch["tool_labels"] is not None else None
            values = batch["values"].to(device) if batch["values"] is not None else None
            plan_labels = batch["plan_labels"].to(device) if batch["plan_labels"] is not None else None
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                tool_labels=tool_labels,
                values=values,
                plan_labels=plan_labels
            )
            
            # Get losses
            losses = outputs["losses"]
            total_loss += losses["total"].item()
            total_lm_loss += losses["lm"].item()
            total_reasoning_loss += losses["reasoning"].item()
            
            # Tool accuracy
            if tool_labels is not None and "logits_tools" in outputs:
                tool_preds = torch.argmax(outputs["logits_tools"], dim=-1)
                tool_correct += (tool_preds == tool_labels).sum().item()
                tool_total += tool_labels.size(0)
    
    # Return metrics
    num_batches = len(dataloader)
    metrics = {
        "total_loss": total_loss / num_batches,
        "lm_loss": total_lm_loss / num_batches,
        "reasoning_loss": total_reasoning_loss / num_batches,
    }
    
    if tool_total > 0:
        metrics["tool_accuracy"] = tool_correct / tool_total
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train L-Former")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--tiny", action="store_true", help="Use tiny config for testing")
    parser.add_argument("--aggregator", type=str, default="ema", choices=["ema", "gru", "attn"], help="Aggregator type")
    parser.add_argument("--sequence_wise", action="store_true", default=True, help="Use sequence-wise aggregation")
    parser.add_argument("--detach_taps", action="store_true", default=True, help="Detach taps during training")
    parser.add_argument("--blend_into_lm", action="store_true", help="Blend reasoning state into LM head")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--phase_a_steps", type=int, default=100, help="Steps for Phase A")
    parser.add_argument("--last_k_unfreeze", type=int, default=0, help="Last k layers to unfreeze in Phase B")
    
    args = parser.parse_args()
    
    # Create config
    if args.tiny:
        config = ModelConfig.tiny()
    elif args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
    else:
        config = ModelConfig()
    
    # Override config with command line args
    config.aggregator = {"type": args.aggregator}
    config.sequence_wise = args.sequence_wise
    config.detach_taps = args.detach_taps
    config.blend_into_lm = args.blend_into_lm
    config.last_k_unfreeze = args.last_k_unfreeze
    
    print(f"Config: {config}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = LFormer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(config, "train", args.batch_size)
    val_dataloader = create_dataloader(config, "val", args.batch_size)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Phase A: Freeze base transformer (first few steps)
        if epoch == 0 and args.phase_a_steps > 0:
            print("Phase A: Freezing base transformer, training side-path + heads")
            model.freeze_base_transformer()
        
        # Phase B: Unfreeze some layers
        if epoch > 0 and args.last_k_unfreeze > 0:
            print(f"Phase B: Unfreezing last {args.last_k_unfreeze} layers")
            model.unfreeze_last_k_layers(args.last_k_unfreeze)
        
        # Train
        start_time = time.time()
        train_losses = train_epoch(model, train_dataloader, optimizer, device, config)
        train_time = time.time() - start_time
        
        # Evaluate
        val_metrics = evaluate(model, val_dataloader, device, config)
        
        # Log metrics
        print(f"Train Losses: {train_losses}")
        print(f"Val Metrics: {val_metrics}")
        print(f"Train time: {train_time:.2f}s")
        
        # Get alpha gates if using EMA
        if hasattr(model, 'get_alpha_gates'):
            alpha_gates = model.get_alpha_gates()
            if alpha_gates is not None:
                print(f"Alpha gates: {alpha_gates.detach().cpu().numpy()}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'train_losses': train_losses,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    print("Training completed!")
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_model_path)
    print(f"Saved final model: {final_model_path}")


if __name__ == "__main__":
    main() 