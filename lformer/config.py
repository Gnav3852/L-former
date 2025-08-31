"""
Configuration for L-Former model
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union


@dataclass
class ModelConfig:
    """Configuration for L-Former model"""
    
    # Base Transformer parameters
    vocab_size: int = 5000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    
    # L-Path parameters
    d_side: int = 128
    tap_every: int = 1  # Tap every N layers (1 = every layer)
    sequence_wise: bool = True  # True: [B, d_side], False: [B, T, d_side]
    
    # Aggregator configuration
    aggregator: Dict[str, Any] = None  # {"type": "ema"|"gru"|"attn"}
    
    # Training parameters
    detach_taps: bool = True  # Detach gradients from base transformer during Phase A
    lambda_lm: float = 0.1      # Reduce LM weight further
    lambda_plan: float = 2.0    # Increase tool selection weight
    
    # Reasoning head parameters
    n_tools: int = 5
    use_value_head: bool = False  # CHANGE TO FALSE
    use_tool_head: bool = True
    plan_decoder: bool = False
    
    # Fine-tuning parameters
    last_k_unfreeze: int = 0  # Number of last layers to unfreeze in Phase B
    
    # Advanced L-Path features
    tree_checkpoint: int = 0  # Tree checkpoint every N layers (0 = off)
    blend_into_lm: bool = False  # Blend S^L into LM head via gated concat
    
    def __post_init__(self):
        """Set default aggregator if not specified"""
        if self.aggregator is None:
            self.aggregator = {"type": "ema"}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def tiny(cls) -> 'ModelConfig':
        """Create ultra-tiny config for testing"""
        config = cls(
            vocab_size=1000,
            d_model=32,         # Reduce from 128
            n_layers=1,         # Reduce from 4
            n_heads=2,          # Reduce from 4
            d_ff=128,          # Reduce from 512
            d_side=8,          # Reduce from 32
            n_tools=3,
            use_value_head=False,
            use_tool_head=True
        )
        # Set loss weights after creation
        config.lambda_lm = 0.1      # Reduce LM weight
        config.lambda_plan = 5.0    # Increase tool weight dramatically
        return config 