"""
Test tensor shapes for L-Former
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ModelConfig
from modeling.lformer import LFormer


def test_sequence_wise_shapes():
    """Test shapes with sequence-wise aggregation"""
    config = ModelConfig.tiny()
    config.sequence_wise = True
    
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids, return_intermediates=True)
    
    # Check LM head output
    assert outputs["logits_lm"].shape == (batch_size, seq_len, config.vocab_size)
    
    # Check reasoning states
    reasoning_states = outputs["internals"]["reasoning_states"]
    assert len(reasoning_states) == config.n_layers
    
    # Each reasoning state should be [batch_size, d_side] for sequence-wise
    for s in reasoning_states:
        assert s.shape == (batch_size, config.d_side)
    
    # Check tool head output
    if config.use_tool_head:
        assert outputs["logits_tools"].shape == (batch_size, config.n_tools)
    
    # Check value head output
    if config.use_value_head:
        assert outputs["values"].shape == (batch_size, 1)


def test_token_wise_shapes():
    """Test shapes with token-wise aggregation"""
    config = ModelConfig.tiny()
    config.sequence_wise = False
    
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids, return_intermediates=True)
    
    # Check LM head output
    assert outputs["logits_lm"].shape == (batch_size, seq_len, config.vocab_size)
    
    # Check reasoning states
    reasoning_states = outputs["internals"]["reasoning_states"]
    assert len(reasoning_states) == config.n_layers
    
    # Each reasoning state should be [batch_size, seq_len, d_side] for token-wise
    for s in reasoning_states:
        assert s.shape == (batch_size, seq_len, config.d_side)
    
    # Check tool head output (should still be [batch_size, n_tools] - uses final reasoning state)
    if config.use_tool_head:
        assert outputs["logits_tools"].shape == (batch_size, config.n_tools)
    
    # Check value head output
    if config.use_value_head:
        assert outputs["values"].shape == (batch_size, 1)


def test_different_aggregators():
    """Test different aggregator types"""
    config = ModelConfig.tiny()
    
    # Test EMA aggregator
    config.aggregator = {"type": "ema"}
    model_ema = LFormer(config)
    
    # Test GRU aggregator
    config.aggregator = {"type": "gru"}
    model_gru = LFormer(config)
    
    # Test attention aggregator
    config.aggregator = {"type": "attn"}
    model_attn = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # All should produce same output shapes
    outputs_ema = model_ema(input_ids)
    outputs_gru = model_gru(input_ids)
    outputs_attn = model_attn(input_ids)
    
    assert outputs_ema["logits_lm"].shape == outputs_gru["logits_lm"].shape
    assert outputs_ema["logits_lm"].shape == outputs_attn["logits_lm"].shape


def test_blend_into_lm():
    """Test blending reasoning state into LM head"""
    config = ModelConfig.tiny()
    config.blend_into_lm = True
    config.sequence_wise = True  # Required for blending
    
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids)
    
    # Check LM head output
    assert outputs["logits_lm"].shape == (batch_size, seq_len, config.vocab_size)
    
    # Check that blending gate exists
    assert hasattr(model, 'blend_gate')
    assert model.blend_gate.requires_grad


if __name__ == "__main__":
    # Run tests
    test_sequence_wise_shapes()
    test_token_wise_shapes()
    test_different_aggregators()
    test_blend_into_lm()
    print("All shape tests passed!") 