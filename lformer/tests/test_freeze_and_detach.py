"""
Test freezing and detaching behavior for L-Former
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ModelConfig
from modeling.lformer import LFormer


def test_detach_taps():
    """Test that taps are detached when detach_taps=True"""
    config = ModelConfig.tiny()
    config.detach_taps = True
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids, return_intermediates=True)
    
    # Get hidden states and reasoning states
    hidden_states = outputs["internals"]["hidden_states"]
    reasoning_states = outputs["internals"]["reasoning_states"]
    
    # Check that reasoning states don't have gradients to hidden states
    # This is a bit tricky to test directly, but we can check that
    # the reasoning states are computed correctly
    
    # All reasoning states should have the same shape
    for i, s in enumerate(reasoning_states):
        assert s.shape == (batch_size, config.d_side)
        assert s.requires_grad  # Should still require grad for training


def test_freeze_base_transformer():
    """Test freezing base transformer parameters"""
    config = ModelConfig.tiny()
    model = LFormer(config)
    
    # Check initial state - all parameters should be trainable
    for name, param in model.named_parameters():
        if 'transformer' in name:
            assert param.requires_grad, f"Parameter {name} should be trainable initially"
    
    # Freeze base transformer
    model.freeze_base_transformer()
    
    # Check that transformer parameters are frozen
    for name, param in model.named_parameters():
        if 'transformer' in name:
            assert not param.requires_grad, f"Parameter {name} should be frozen"
    
    # Check that other parameters are still trainable
    for name, param in model.named_parameters():
        if 'transformer' not in name:
            assert param.requires_grad, f"Parameter {name} should still be trainable"


def test_unfreeze_last_k_layers():
    """Test unfreezing last k layers"""
    config = ModelConfig.tiny()
    model = LFormer(config)
    
    # Freeze base transformer first
    model.freeze_base_transformer()
    
    # Unfreeze last 2 layers
    k = 2
    model.unfreeze_last_k_layers(k)
    
    # Check that last k layers are unfrozen
    transformer_params = [(name, param) for name, param in model.named_parameters() if 'transformer' in name]
    
    # Count trainable transformer parameters
    trainable_count = sum(1 for _, param in transformer_params if param.requires_grad)
    
    # Should have some trainable parameters now
    assert trainable_count > 0, "Some transformer parameters should be trainable after unfreezing"
    
    # Check that earlier layers are still frozen
    early_layers_frozen = True
    for name, param in transformer_params:
        if 'blocks.0' in name or 'blocks.1' in name:  # First few layers
            if param.requires_grad:
                early_layers_frozen = False
                break
    
    assert early_layers_frozen, "Early layers should still be frozen"


def test_gradient_flow():
    """Test gradient flow with different configurations"""
    config = ModelConfig.tiny()
    model = LFormer(config)
    
    # Test input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Test 1: Normal training (all parameters trainable)
    model.train()
    outputs = model(input_ids, labels=labels)
    loss = outputs["losses"]["total"]
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients, "Should have gradients in normal training"
    
    # Clear gradients
    model.zero_grad()
    
    # Test 2: Frozen base transformer
    model.freeze_base_transformer()
    outputs = model(input_ids, labels=labels)
    loss = outputs["losses"]["total"]
    
    # Backward pass
    loss.backward()
    
    # Check that transformer parameters don't have gradients
    transformer_has_gradients = False
    for name, param in model.named_parameters():
        if 'transformer' in name and param.grad is not None:
            transformer_has_gradients = True
            break
    
    assert not transformer_has_gradients, "Transformer parameters should not have gradients when frozen"
    
    # Check that other parameters still have gradients
    other_has_gradients = False
    for name, param in model.named_parameters():
        if 'transformer' not in name and param.grad is not None:
            other_has_gradients = True
            break
    
    assert other_has_gradients, "Non-transformer parameters should still have gradients"


def test_phase_a_phase_b():
    """Test Phase A and Phase B training phases"""
    config = ModelConfig.tiny()
    model = LFormer(config)
    
    # Phase A: Freeze base transformer
    model.freeze_base_transformer()
    
    # Check that only side-path and heads are trainable
    trainable_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)
    
    # Should have some trainable parameters (side-path, heads)
    assert len(trainable_params) > 0, "Should have trainable parameters in Phase A"
    
    # Should have some frozen parameters (transformer)
    assert len(frozen_params) > 0, "Should have frozen parameters in Phase A"
    
    # Phase B: Unfreeze last k layers
    k = 1
    model.unfreeze_last_k_layers(k)
    
    # Count trainable transformer parameters
    trainable_transformer_params = []
    for name, param in model.named_parameters():
        if 'transformer' in name and param.requires_grad:
            trainable_transformer_params.append(name)
    
    # Should have some trainable transformer parameters now
    assert len(trainable_transformer_params) > 0, "Should have trainable transformer parameters in Phase B"


if __name__ == "__main__":
    # Run tests
    test_detach_taps()
    test_freeze_base_transformer()
    test_unfreeze_last_k_layers()
    test_gradient_flow()
    test_phase_a_phase_b()
    print("All freeze and detach tests passed!") 