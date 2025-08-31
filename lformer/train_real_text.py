#!/usr/bin/env python3
"""
Training script for L-Former with REAL TEXT dataset - FIXED VERSION
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

# Import the new dataset
from data.real_text_dataset import create_dataloader
from modeling.lformer import LFormer
from config import ModelConfig


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch with gradient monitoring"""
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_tool_loss = 0
    total_value_loss = 0
    
    # Add gradient monitoring
    grad_norms = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
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
        
        if "tool" in losses:
            total_tool_loss += losses["tool"].item()
        if "value" in losses:
            total_value_loss += losses["value"].item()
        
        # Backward pass
        optimizer.zero_grad()
        losses["total"].backward()
        
        # Monitor and clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(total_norm.item())
        
        optimizer.step()
        
        # Print detailed loss breakdown every 50 batches
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}: Tool Loss: {losses['tool'].item():.4f}, "
                  f"LM Loss: {losses['lm'].item():.4f}, "
                  f"Grad Norm: {total_norm.item():.4f}")
    
    # Print gradient statistics
    if grad_norms:
        print(f"Gradient Norms - Mean: {sum(grad_norms)/len(grad_norms):.4f}, "
              f"Max: {max(grad_norms):.4f}, Min: {min(grad_norms):.4f}")
    
    # Return average losses
    num_batches = len(dataloader)
    return {
        "total": total_loss / num_batches,
        "lm": total_lm_loss / num_batches,
        "tool": total_tool_loss / num_batches if total_tool_loss > 0 else 0,
        "value": total_value_loss / num_batches if total_value_loss > 0 else 0
    }


def validate_epoch(model, dataloader, device, config):
    """Validate for one epoch - NO GRADIENTS"""
    model.eval()
    total_loss = 0
    total_lm_loss = 0
    total_tool_loss = 0
    total_value_loss = 0
    
    with torch.no_grad():  # No gradients during validation
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            tool_labels = batch["tool_labels"].to(device) if batch["tool_labels"] is not None else None
            values = batch["values"].to(device) if batch["values"] is not None else None
            plan_labels = batch["plan_labels"].to(device) if batch["plan_labels"] is not None else None
            
            # Forward pass only
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                tool_labels=tool_labels,
                values=values,
                plan_labels=plan_labels
            )
            
            # Get losses (no backward pass)
            losses = outputs["losses"]
            total_loss += losses["total"].item()
            total_lm_loss += losses["lm"].item()
            
            if "tool" in losses:
                total_tool_loss += losses["tool"].item()
            if "value" in losses:
                total_value_loss += losses["value"].item()
    
    # Return average losses
    num_batches = len(dataloader)
    return {
        "total": total_loss / num_batches,
        "lm": total_lm_loss / num_batches,
        "tool": total_tool_loss / num_batches if total_tool_loss > 0 else 0,
        "value": total_value_loss / num_batches if total_value_loss > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Train L-Former with real text")
    parser.add_argument("--tiny", action="store_true", help="Use tiny config")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--phase_a_steps", type=int, default=0, help="Phase A steps (0 = skip)")
    parser.add_argument("--last_k_unfreeze", type=int, default=0, help="Last k layers to unfreeze")
    
    args = parser.parse_args()
    
    # Create config with GPT-2 vocabulary size
    if args.tiny:
        config = ModelConfig.tiny()
        config.vocab_size = 50257  # GPT-2 vocabulary size
    else:
        config = ModelConfig()
        config.vocab_size = 50257  # GPT-2 vocabulary size
    
    print(f"Config: {config}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # If using CPU, reduce batch size for memory
    if device.type == "cpu":
        args.batch_size = min(args.batch_size, 8)
        print(f"CPU detected, reducing batch size to {args.batch_size}")
    
    # Create model
    model = LFormer(config)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(config, "train", args.batch_size)
    val_dataloader = create_dataloader(config, "val", args.batch_size)
    
    print("Starting training...")
    
    # Phase A: Train side-path + heads (freeze base transformer)
    if args.phase_a_steps > 0:
        print(f"\nEpoch 1/{args.epochs}")
        print("Phase A: Freezing base transformer, training side-path + heads")
        
        # Freeze base transformer
        for param in model.transformer.parameters():
            param.requires_grad = False
        
        # Train only side-path and heads
        optimizer = optim.AdamW([
            {'params': model.lpath.parameters(), 'lr': args.lr},
            {'params': model.lm_head.parameters(), 'lr': args.lr},
            {'params': model.tool_head.parameters(), 'lr': args.lr},
            {'params': model.value_head.parameters(), 'lr': args.lr}
        ], lr=args.lr)
        
        # Train for specified steps
        for step in range(args.phase_a_steps):
            batch = next(iter(train_dataloader))
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            tool_labels = batch["tool_labels"].to(device) if batch["tool_labels"] is not None else None
            values = batch["values"].to(device) if batch["values"] is not None else None
            
            outputs = model(input_ids=input_ids, labels=labels, tool_labels=tool_labels, values=values)
            loss = outputs["losses"]["total"]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")
    
    # Phase B: Unfreeze last k layers
    if args.last_k_unfreeze > 0:
        print(f"\nPhase B: Unfreezing last {args.last_k_unfreeze} layers")
        
        # Unfreeze last k layers
        for i, block in enumerate(model.transformer.blocks):
            if i >= len(model.transformer.blocks) - args.last_k_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
    
    # Unfreeze all parameters for final training
    for param in model.parameters():
        param.requires_grad = True
    
    # Final training
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_losses = train_epoch(model, train_dataloader, optimizer, device, config)
        
        # Validation
        model.eval()
        val_losses = validate_epoch(model, val_dataloader, device, config)
        
        # Print losses
        print(f"Train Losses: Total={train_losses['total']:.4f}, LM={train_losses['lm']:.4f}, Tool={train_losses['tool']:.4f}, Value={train_losses['value']:.4f}")
        print(f"Val Losses: Total={val_losses['total']:.4f}, LM={val_losses['lm']:.4f}, Tool={val_losses['tool']:.4f}, Value={val_losses['value']:.4f}")
        
        # Update learning rate based on validation loss
        scheduler.step(val_losses['total'])
        
        # Save checkpoint
        if epoch % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "train_losses": train_losses,
                "val_losses": val_losses
            }
            torch.save(checkpoint, f"checkpoints/checkpoint_epoch_{epoch}.pt")
            print(f"Saved checkpoint: checkpoints/checkpoint_epoch_{epoch}.pt")
    
    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    final_checkpoint = {
        "model": model,
        "config": config,
        "final_epoch": args.epochs
    }
    torch.save(final_checkpoint, "checkpoints/final_model_real_text.pt")
    print("Training completed!")
    print("Saved final model: checkpoints/final_model_real_text.pt")


if __name__ == "__main__":
    main()