"""
Test forward pass and losses for L-Former
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ModelConfig
from modeling.lformer import LFormer


def test_forward_pass():
    """Test basic forward pass"""
    config = ModelConfig.tiny()
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass without labels
    outputs = model(input_ids)
    
    # Check outputs
    assert "logits_lm" in outputs
    assert "losses" in outputs
    
    # Check LM logits shape
    assert outputs["logits_lm"].shape == (batch_size, seq_len, config.vocab_size)
    
    # Check losses
    losses = outputs["losses"]
    assert "lm" in losses
    assert "reasoning" in losses
    assert "total" in losses
    
    # Without labels, losses should be 0
    assert losses["lm"].item() == 0.0
    assert losses["reasoning"].item() == 0.0
    assert losses["total"].item() == 0.0


def test_losses_with_labels():
    """Test losses computation with labels"""
    config = ModelConfig.tiny()
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward pass with labels
    outputs = model(input_ids, labels=labels)
    
    # Check losses
    losses = outputs["losses"]
    
    # With labels, losses should be positive
    assert losses["lm"].item() > 0.0
    assert losses["total"].item() > 0.0
    
    # LM loss should dominate initially
    assert losses["lm"].item() > losses["reasoning"].item()


def test_tool_loss():
    """Test tool selection loss"""
    config = ModelConfig.tiny()
    config.use_tool_head = True
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    tool_labels = torch.randint(0, config.n_tools, (batch_size,))
    
    # Forward pass
    outputs = model(input_ids, tool_labels=tool_labels)
    
    # Check tool logits
    assert "logits_tools" in outputs
    assert outputs["logits_tools"].shape == (batch_size, config.n_tools)
    
    # Check losses
    losses = outputs["losses"]
    assert losses["tool"].item() > 0.0
    assert losses["reasoning"].item() > 0.0


def test_value_loss():
    """Test value prediction loss"""
    config = ModelConfig.tiny()
    config.use_value_head = True
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    values = torch.randn(batch_size, 1)
    
    # Forward pass
    outputs = model(input_ids, values=values)
    
    # Check value predictions
    assert "values" in outputs
    assert outputs["values"].shape == (batch_size, 1)
    
    # Check losses
    losses = outputs["losses"]
    assert losses["value"].item() > 0.0
    assert losses["reasoning"].item() > 0.0


def test_plan_loss():
    """Test plan generation loss"""
    config = ModelConfig.tiny()
    config.plan_decoder = True
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    plan_labels = torch.randint(0, config.vocab_size, (batch_size, 5))
    
    # Forward pass
    outputs = model(input_ids, plan_labels=plan_labels)
    
    # Check plan logits
    assert "logits_plan" in outputs
    assert outputs["logits_plan"].shape == (batch_size, 5, config.vocab_size)
    
    # Check losses
    losses = outputs["losses"]
    assert losses["plan"].item() > 0.0
    assert losses["reasoning"].item() > 0.0


def test_loss_decrease():
    """Test that losses decrease with training steps"""
    config = ModelConfig.tiny()
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Initial loss
    outputs = model(input_ids, labels=labels)
    initial_loss = outputs["losses"]["total"].item()
    
    # Training step
    optimizer.zero_grad()
    outputs = model(input_ids, labels=labels)
    loss = outputs["losses"]["total"]
    loss.backward()
    optimizer.step()
    
    # Loss after training step
    outputs = model(input_ids, labels=labels)
    final_loss = outputs["losses"]["total"].item()
    
    # Loss should decrease (or stay the same)
    assert final_loss <= initial_loss + 1e-6  # Allow small numerical differences


def test_alpha_gates():
    """Test alpha gates for EMA aggregator"""
    config = ModelConfig.tiny()
    config.aggregator = {"type": "ema"}
    model = LFormer(config)
    
    # Get alpha gates
    alpha_gates = model.get_alpha_gates()
    assert alpha_gates is not None
    
    # Check shape
    assert alpha_gates.shape == (config.n_layers,)
    
    # Check values are between 0 and 1 (sigmoid output)
    assert torch.all(alpha_gates >= 0) and torch.all(alpha_gates <= 1)
    
    # Initial values should be around 0.12
    alpha_values = alpha_gates.detach().cpu().numpy()
    for alpha in alpha_values:
        assert 0.1 <= alpha <= 0.15  # Allow small variation


if __name__ == "__main__":
    # Run tests
    test_forward_pass()
    test_losses_with_labels()
    test_tool_loss()
    test_value_loss()
    test_plan_loss()
    test_loss_decrease()
    test_alpha_gates()
    print("All forward and loss tests passed!") 