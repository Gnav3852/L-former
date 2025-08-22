"""
L-Path: Progressive aggregation side-path for L-Former
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any


class TapProjector(nn.Module):
    """Projects tapped hidden states to side dimension"""
    
    def __init__(self, d_model: int, d_side: int, dropout: float = 0.1, 
                 sequence_wise: bool = True, pool: str = "mean"):
        super().__init__()
        self.d_model = d_model
        self.d_side = d_side
        self.sequence_wise = sequence_wise
        self.pool = pool
        
        # Projection: Linear + LayerNorm + Dropout
        self.projection = nn.Linear(d_model, d_side)
        self.layer_norm = nn.LayerNorm(d_side)
        self.dropout = nn.Dropout(dropout)
        
        # Optional attention pooling
        if pool == "attn":
            self.attention = nn.MultiheadAttention(d_side, num_heads=4, batch_first=True)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            
        Returns:
            projected: [batch_size, d_side] if sequence_wise else [batch_size, seq_len, d_side]
        """
        # Project to side dimension
        projected = self.projection(hidden_states)
        projected = self.layer_norm(projected)
        projected = self.dropout(projected)
        
        if self.sequence_wise:
            # Pool over sequence dimension
            if self.pool == "mean":
                projected = projected.mean(dim=1)  # [B, d_side]
            elif self.pool == "cls":
                projected = projected[:, 0]  # [B, d_side]
            elif self.pool == "attn":
                # Use attention pooling
                query = projected.mean(dim=1, keepdim=True)  # [B, 1, d_side]
                projected, _ = self.attention(query, projected, projected)
                projected = projected.squeeze(1)  # [B, d_side]
        
        return projected


class EmaAggregator(nn.Module):
    """EMA-style aggregator with learnable alpha gates"""
    
    def __init__(self, d_side: int, n_layers: int, tree_checkpoint: int = 0):
        super().__init__()
        self.d_side = d_side
        self.n_layers = n_layers
        self.tree_checkpoint = tree_checkpoint
        
        # Learnable alpha parameters (initialized to ~0.12)
        self.alpha_params = nn.Parameter(torch.full((n_layers,), -2.0))
        
        # MLP phi: d_side -> 4*d_side -> d_side
        self.phi = nn.Sequential(
            nn.Linear(d_side, 4 * d_side),
            nn.GELU(),
            nn.LayerNorm(4 * d_side),
            nn.Linear(4 * d_side, d_side),
            nn.LayerNorm(d_side)
        )
        
        # Tree checkpoint MLP if enabled
        if tree_checkpoint > 0:
            self.tree_mlp = nn.Sequential(
                nn.Linear(2 * d_side, d_side),
                nn.GELU(),
                nn.LayerNorm(d_side)
            )
    
    def forward(self, z_list: List[torch.Tensor], s_init: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Args:
            z_list: List of [batch_size, d_side] projected taps
            s_init: Initial reasoning state [batch_size, d_side]
            
        Returns:
            s_list: List of reasoning states [batch_size, d_side] for each layer
        """
        batch_size = z_list[0].shape[0]
        device = z_list[0].device
        
        # Initialize reasoning state
        if s_init is None:
            s_init = torch.zeros(batch_size, self.d_side, device=device)
        
        s_list = []
        s_current = s_init
        
        for l, z_l in enumerate(z_list):
            # Compute alpha gate: σ(a^l) ≈ 0.12 initially
            alpha_l = torch.sigmoid(self.alpha_params[l])
            
            # Apply MLP phi to current tap
            phi_z = self.phi(z_l)
            
            # EMA update: S^l = (1-α^l) * S^{l-1} + α^l * φ(z^l)
            s_current = (1 - alpha_l) * s_current + alpha_l * phi_z
            
            # Tree checkpoint: if enabled and at checkpoint layer
            if self.tree_checkpoint > 0 and (l + 1) % self.tree_checkpoint == 0:
                # Concatenate current S with checkpoint S and process
                s_checkpoint = s_list[-(self.tree_checkpoint // 2)] if len(s_list) >= self.tree_checkpoint // 2 else s_init
                s_combined = torch.cat([s_current, s_checkpoint], dim=-1)
                s_current = self.tree_mlp(s_combined)
            
            s_list.append(s_current)
        
        return s_list


class GruAggregator(nn.Module):
    """GRU-based aggregator treating depth as time dimension"""
    
    def __init__(self, d_side: int, n_layers: int):
        super().__init__()
        self.d_side = d_side
        self.n_layers = n_layers
        
        # GRU cell for depth-wise aggregation
        self.gru = nn.GRUCell(d_side, d_side)
        
        # MLP phi: d_side -> 4*d_side -> d_side
        self.phi = nn.Sequential(
            nn.Linear(d_side, 4 * d_side),
            nn.GELU(),
            nn.LayerNorm(4 * d_side),
            nn.Linear(4 * d_side, d_side),
            nn.LayerNorm(d_side)
        )
    
    def forward(self, z_list: List[torch.Tensor], s_init: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Args:
            z_list: List of [batch_size, d_side] projected taps
            s_init: Initial reasoning state [batch_size, d_side]
            
        Returns:
            s_list: List of reasoning states [batch_size, d_side] for each layer
        """
        batch_size = z_list[0].shape[0]
        device = z_list[0].device
        
        # Initialize reasoning state
        if s_init is None:
            s_init = torch.zeros(batch_size, self.d_side, device=device)
        
        s_list = []
        s_current = s_init
        
        for z_l in z_list:
            # Apply MLP phi to current tap
            phi_z = self.phi(z_l)
            
            # GRU update: treat depth as time
            s_current = self.gru(phi_z, s_current)
            
            s_list.append(s_current)
        
        return s_list


class DepthAttentionAggregator(nn.Module):
    """Depth attention aggregator with memory of previous taps"""
    
    def __init__(self, d_side: int, n_layers: int, max_memory: int = 10):
        super().__init__()
        self.d_side = d_side
        self.n_layers = n_layers
        self.max_memory = max_memory
        
        # MLP phi: d_side -> 4*d_side -> d_side
        self.phi = nn.Sequential(
            nn.Linear(d_side, 4 * d_side),
            nn.GELU(),
            nn.LayerNorm(4 * d_side),
            nn.Linear(4 * d_side, d_side),
            nn.LayerNorm(d_side)
        )
        
        # Attention mechanism for depth memory
        self.attention = nn.MultiheadAttention(d_side, num_heads=4, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(d_side, d_side)
    
    def forward(self, z_list: List[torch.Tensor], s_init: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Args:
            z_list: List of [batch_size, d_side] projected taps
            s_init: Initial reasoning state [batch_size, d_side]
            
        Returns:
            s_list: List of reasoning states [batch_size, d_side] for each layer
        """
        batch_size = z_list[0].shape[0]
        device = z_list[0].device
        
        # Initialize reasoning state
        if s_init is None:
            s_init = torch.zeros(batch_size, self.d_side, device=device)
        
        s_list = []
        s_current = s_init
        memory = []  # FIFO memory of previous taps
        
        for z_l in z_list:
            # Apply MLP phi to current tap
            phi_z = self.phi(z_l)
            
            # Add to memory (FIFO)
            memory.append(phi_z)
            if len(memory) > self.max_memory:
                memory.pop(0)
            
            # Attend from current S to memory
            if len(memory) > 1:
                memory_tensor = torch.stack(memory, dim=1)  # [B, mem_len, d_side]
                s_expanded = s_current.unsqueeze(1)  # [B, 1, d_side]
                
                # Cross-attention: S attends to memory
                attended, _ = self.attention(s_expanded, memory_tensor, memory_tensor)
                attended = attended.squeeze(1)  # [B, d_side]
                
                # Combine current S with attended memory
                s_current = self.output_proj(s_current + attended)
            else:
                # No memory yet, just use current tap
                s_current = phi_z
            
            s_list.append(s_current)
        
        return s_list


class LPath(nn.Module):
    """L-shaped progressive aggregation side-path"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Tap projector
        self.tap_projector = TapProjector(
            config.d_model, 
            config.d_side, 
            config.dropout,
            config.sequence_wise
        )
        
        # Aggregator
        aggregator_type = config.aggregator.get("type", "ema")
        if aggregator_type == "ema":
            self.aggregator = EmaAggregator(
                config.d_side, 
                config.n_layers,
                config.tree_checkpoint
            )
        elif aggregator_type == "gru":
            self.aggregator = GruAggregator(config.d_side, config.n_layers)
        elif aggregator_type == "attn":
            self.aggregator = DepthAttentionAggregator(config.d_side, config.n_layers)
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")
    
    def forward(self, hidden_states: List[torch.Tensor], detach_taps: bool = True) -> List[torch.Tensor]:
        """
        Forward pass through L-Path
        
        Args:
            hidden_states: List of [batch_size, seq_len, d_model] from transformer
            detach_taps: Whether to detach gradients from taps
            
        Returns:
            s_list: List of reasoning states [batch_size, d_side] for each layer
        """
        # Filter taps based on tap_every
        tapped_layers = hidden_states[::self.config.tap_every]
        
        # Project taps to side dimension
        z_list = []
        for h_l in tapped_layers:
            if detach_taps:
                h_l = h_l.detach()
            
            z_l = self.tap_projector(h_l)
            z_list.append(z_l)
        
        # Aggregate progressively
        s_list = self.aggregator(z_list)
        
        return s_list 