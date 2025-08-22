"""
Heads for L-Former model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class LMHead(nn.Module):
    """Language modeling head"""
    
    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Projection to vocabulary
        self.projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Project to vocabulary
        logits = self.projection(hidden_states)
        
        return logits


class ToolHead(nn.Module):
    """Tool selection head"""
    
    def __init__(self, d_side: int, n_tools: int, dropout: float = 0.1):
        super().__init__()
        self.d_side = d_side
        self.n_tools = n_tools
        
        # Projection to tool space
        self.projection = nn.Linear(d_side, n_tools)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, reasoning_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reasoning_state: [batch_size, d_side]
            
        Returns:
            logits: [batch_size, n_tools]
        """
        # Apply dropout
        reasoning_state = self.dropout(reasoning_state)
        
        # Project to tool space
        logits = self.projection(reasoning_state)
        
        return logits


class ValueHead(nn.Module):
    """Value prediction head"""
    
    def __init__(self, d_side: int, dropout: float = 0.1):
        super().__init__()
        self.d_side = d_side
        
        # Projection to scalar value
        self.projection = nn.Linear(d_side, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, reasoning_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reasoning_state: [batch_size, d_side]
            
        Returns:
            values: [batch_size, 1]
        """
        # Apply dropout
        reasoning_state = self.dropout(reasoning_state)
        
        # Project to scalar value
        values = self.projection(reasoning_state)
        
        return values


class PlanDecoder(nn.Module):
    """Tiny causal decoder for plan generation"""
    
    def __init__(self, d_side: int, d_model: int, vocab_size: int, max_plan_len: int = 20, dropout: float = 0.1):
        super().__init__()
        self.d_side = d_side
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_plan_len = max_plan_len
        
        # Project reasoning state to decoder dimension
        self.reasoning_proj = nn.Linear(d_side, d_model)
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_plan_len, d_model)
        
        # Cross-attention to reasoning state
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Self-attention for plan generation
        self.self_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, reasoning_state: torch.Tensor, plan_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            reasoning_state: [batch_size, d_side]
            plan_ids: [batch_size, plan_len] (optional, for teacher forcing)
            
        Returns:
            logits: [batch_size, plan_len, vocab_size]
        """
        batch_size = reasoning_state.shape[0]
        device = reasoning_state.device
        
        # Project reasoning state to decoder dimension
        reasoning_proj = self.reasoning_proj(reasoning_state)  # [B, d_model]
        reasoning_proj = reasoning_proj.unsqueeze(1)  # [B, 1, d_model]
        
        if plan_ids is None:
            # Generate plan autoregressively
            return self._generate_plan(reasoning_proj)
        
        # Teacher forcing mode
        plan_len = plan_ids.shape[1]
        
        # Embeddings
        token_embeds = self.token_embedding(plan_ids)
        position_ids = torch.arange(plan_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Cross-attention to reasoning state
        x = self.norm1(x + self._cross_attention_block(x, reasoning_proj))
        
        # Self-attention (causal)
        x = self.norm2(x + self._self_attention_block(x))
        
        # Feed-forward
        x = self.norm3(x + self.feed_forward(x))
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits
    
    def _cross_attention_block(self, x: torch.Tensor, reasoning: torch.Tensor) -> torch.Tensor:
        """Cross-attention block"""
        attended, _ = self.cross_attention(x, reasoning, reasoning)
        return attended
    
    def _self_attention_block(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention block with causal masking"""
        seq_len = x.shape[1]
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)
        
        attended, _ = self.self_attention(x, x, x, attn_mask=mask)
        return attended
    
    def _generate_plan(self, reasoning_proj: torch.Tensor) -> torch.Tensor:
        """Generate plan autoregressively"""
        batch_size = reasoning_proj.shape[0]
        device = reasoning_proj.device
        
        # Start with BOS token
        current_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)  # Assuming 0 is BOS
        
        for step in range(self.max_plan_len - 1):
            # Forward pass
            logits = self.forward(reasoning_proj.squeeze(1), current_ids)
            
            # Get next token (greedy decoding)
            next_logits = logits[:, -1, :]  # [B, vocab_size]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [B, 1]
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Stop if EOS token is generated
            if (next_token == 1).any():  # Assuming 1 is EOS
                break
        
        # Return final logits
        return self.forward(reasoning_proj.squeeze(1), current_ids) 