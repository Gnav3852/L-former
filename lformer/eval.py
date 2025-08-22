"""
Evaluation script for L-Former
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import json

from config import ModelConfig
from modeling.lformer import LFormer
from data.toy_planning_dataset import create_dataloader


def compute_perplexity(model, dataloader, device):
    """Compute perplexity for language modeling"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            
            # Compute loss
            logits = outputs["logits_lm"]
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += (labels[:, 1:] != -100).sum().item()
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item(), avg_loss


def compute_tool_accuracy(model, dataloader, device):
    """Compute tool selection accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            tool_labels = batch["tool_labels"]
            
            if tool_labels is None:
                continue
            
            tool_labels = tool_labels.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            
            if "logits_tools" not in outputs:
                continue
            
            # Get predictions
            logits = outputs["logits_tools"]
            predictions = torch.argmax(logits, dim=-1)
            
            # Compute accuracy
            correct += (predictions == tool_labels).sum().item()
            total += tool_labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def compute_value_metrics(model, dataloader, device):
    """Compute value prediction metrics"""
    model.eval()
    total_mse = 0
    total_mae = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            values = batch["values"]
            
            if values is None:
                continue
            
            values = values.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            
            if "values" not in outputs:
                continue
            
            # Get predictions
            predicted_values = outputs["values"]
            
            # Compute metrics
            mse = F.mse_loss(predicted_values, values, reduction='sum')
            mae = F.l1_loss(predicted_values, values, reduction='sum')
            
            total_mse += mse.item()
            total_mae += mae.item()
            total_samples += values.size(0)
    
    avg_mse = total_mse / total_samples if total_samples > 0 else 0.0
    avg_mae = total_mae / total_samples if total_samples > 0 else 0.0
    
    return avg_mse, avg_mae, total_samples


def compute_plan_metrics(model, dataloader, device):
    """Compute plan generation metrics (simple BLEU-like)"""
    model.eval()
    total_exact_matches = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            plan_labels = batch["plan_labels"]
            
            if plan_labels is None:
                continue
            
            plan_labels = plan_labels.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            
            if "logits_plan" not in outputs:
                continue
            
            # Get predictions
            logits = outputs["logits_plan"]
            predictions = torch.argmax(logits, dim=-1)
            
            # Compute exact match accuracy
            for i in range(predictions.size(0)):
                pred_seq = predictions[i]
                true_seq = plan_labels[i]
                
                # Remove padding tokens
                pred_seq = pred_seq[pred_seq != -100]
                true_seq = true_seq[true_seq != -100]
                
                if pred_seq.size(0) == true_seq.size(0) and torch.all(pred_seq == true_seq):
                    total_exact_matches += 1
                
                total_samples += 1
    
    exact_match_accuracy = total_exact_matches / total_samples if total_samples > 0 else 0.0
    return exact_match_accuracy, total_exact_matches, total_samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate L-Former")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_file", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
    else:
        config = checkpoint.get('config', ModelConfig())
    
    print(f"Config: {config}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = LFormer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create dataloader
    val_dataloader = create_dataloader(config, "val", args.batch_size)
    
    # Evaluate
    print("Evaluating model...")
    
    results = {}
    
    # Language modeling perplexity
    print("Computing perplexity...")
    perplexity, avg_loss = compute_perplexity(model, val_dataloader, device)
    results["perplexity"] = perplexity
    results["lm_loss"] = avg_loss
    print(f"Perplexity: {perplexity:.4f}")
    print(f"LM Loss: {avg_loss:.4f}")
    
    # Tool accuracy
    if config.use_tool_head:
        print("Computing tool accuracy...")
        tool_accuracy, correct, total = compute_tool_accuracy(model, val_dataloader, device)
        results["tool_accuracy"] = tool_accuracy
        results["tool_correct"] = correct
        results["tool_total"] = total
        print(f"Tool Accuracy: {tool_accuracy:.4f} ({correct}/{total})")
    
    # Value metrics
    if config.use_value_head:
        print("Computing value metrics...")
        mse, mae, samples = compute_value_metrics(model, val_dataloader, device)
        results["value_mse"] = mse
        results["value_mae"] = mae
        results["value_samples"] = samples
        print(f"Value MSE: {mse:.4f}")
        print(f"Value MAE: {mae:.4f}")
    
    # Plan metrics
    if config.plan_decoder:
        print("Computing plan metrics...")
        exact_match, matches, samples = compute_plan_metrics(model, val_dataloader, device)
        results["plan_exact_match"] = exact_match
        results["plan_matches"] = matches
        results["plan_samples"] = samples
        print(f"Plan Exact Match: {exact_match:.4f} ({matches}/{samples})")
    
    # Get alpha gates if using EMA
    if hasattr(model, 'get_alpha_gates'):
        alpha_gates = model.get_alpha_gates()
        if alpha_gates is not None:
            alpha_values = alpha_gates.detach().cpu().numpy()
            results["alpha_gates"] = alpha_values.tolist()
            print(f"Alpha gates: {alpha_values}")
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Perplexity: {perplexity:.4f}")
    if config.use_tool_head:
        print(f"Tool Accuracy: {results['tool_accuracy']:.4f}")
    if config.use_value_head:
        print(f"Value MSE: {results['value_mse']:.4f}")
    if config.plan_decoder:
        print(f"Plan Exact Match: {results['plan_exact_match']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main() 