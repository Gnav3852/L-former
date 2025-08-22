"""
Base Transformer module for L-Former
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(x.device)
        
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with Pre-LN and residual connection"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN
        normalized = self.layer_norm(x)
        
        # FFN
        output = self.w_1(normalized)
        output = F.gelu(output)
        output = self.dropout(output)
        output = self.w_2(output)
        output = self.dropout(output)
        
        # Residual connection
        return x + output


class TransformerBlock(nn.Module):
    """Single Transformer block with MHA + FFN"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # MHA with residual
        x = self.attention(x, mask)
        
        # FFN with residual
        x = self.feed_forward(x)
        
        return x


class Transformer(nn.Module):
    """Base Transformer that returns all per-layer hidden states"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(1024, config.d_model)  # Max seq len 1024
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, 
                config.n_heads, 
                config.d_ff, 
                config.dropout
            ) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through transformer
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)
            
        Returns:
            final_hidden: [batch_size, seq_len, d_model]
            hidden_states: List of [batch_size, seq_len, d_model] for each layer
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Store all hidden states
        hidden_states = []
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
            hidden_states.append(x)  # Post-block hidden state
        
        # Final layer norm
        final_hidden = self.final_layer_norm(x)
        
        return final_hidden, hidden_states 