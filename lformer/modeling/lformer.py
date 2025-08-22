"""
Main L-Former model combining transformer, L-path, and reasoning heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

from .transformer import Transformer
from .lpath import LPath
from .heads import LMHead, ToolHead, ValueHead, PlanDecoder


class LFormer(nn.Module):
    """L-Former: Transformer with L-shaped progressive aggregation side-path"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base transformer
        self.transformer = Transformer(config)
        
        # L-Path for progressive aggregation
        self.lpath = LPath(config)
        
        # Heads
        self.lm_head = LMHead(config.d_model, config.vocab_size, config.dropout)
        
        if config.use_tool_head:
            self.tool_head = ToolHead(config.d_side, config.n_tools, config.dropout)
        
        if config.use_value_head:
            self.value_head = ValueHead(config.d_side, config.dropout)
        
        if config.plan_decoder:
            self.plan_decoder = PlanDecoder(
                config.d_side, 
                config.d_model, 
                config.vocab_size,
                dropout=config.dropout
            )
        
        # Optional blending gate for LM head
        if config.blend_into_lm:
            self.blend_gate = nn.Parameter(torch.tensor(0.0))  # Initialize to 0
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tool_labels: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        plan_labels: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through L-Former
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (optional)
            labels: [batch_size, seq_len] for language modeling
            tool_labels: [batch_size] for tool selection
            values: [batch_size, 1] for value prediction
            plan_labels: [batch_size, plan_len] for plan generation
            return_intermediates: Whether to return intermediate states
            
        Returns:
            Dictionary containing:
                - logits_lm: [batch_size, seq_len, vocab_size]
                - logits_tools: [batch_size, n_tools] (if enabled)
                - values: [batch_size, 1] (if enabled)
                - logits_plan: [batch_size, plan_len, vocab_size] (if enabled)
                - losses: Dictionary of losses
                - internals: Intermediate states (if requested)
        """
        batch_size, seq_len = input_ids.shape
        
        # Forward through transformer
        final_hidden, hidden_states = self.transformer(input_ids, attention_mask)
        
        # Forward through L-Path
        reasoning_states = self.lpath(hidden_states, self.config.detach_taps)
        final_reasoning = reasoning_states[-1]  # S^L
        
        # Language modeling head
        if self.config.blend_into_lm and final_reasoning.shape[1] == 1:
            # Blend S^L into LM head via gated concat
            # Expand reasoning state to match sequence length
            reasoning_expanded = final_reasoning.expand(-1, seq_len, -1)
            
            # Gated concatenation: H^L || (β * S^L)
            blended_hidden = torch.cat([
                final_hidden,
                self.blend_gate * reasoning_expanded
            ], dim=-1)
            
            # Project back to d_model
            blend_proj = nn.Linear(
                self.config.d_model + self.config.d_side, 
                self.config.d_model
            ).to(final_hidden.device)
            blended_hidden = blend_proj(blended_hidden)
            
            logits_lm = self.lm_head(blended_hidden)
        else:
            logits_lm = self.lm_head(final_hidden)
        
        # Initialize outputs
        outputs = {
            "logits_lm": logits_lm,
            "losses": {}
        }
        
        # Tool head
        if self.config.use_tool_head:
            logits_tools = self.tool_head(final_reasoning)
            outputs["logits_tools"] = logits_tools
        
        # Value head
        if self.config.use_value_head:
            predicted_values = self.value_head(final_reasoning)
            outputs["values"] = predicted_values
        
        # Plan decoder
        if self.config.plan_decoder:
            logits_plan = self.plan_decoder(final_reasoning, plan_labels)
            outputs["logits_plan"] = logits_plan
        
        # Compute losses
        losses = self._compute_losses(
            logits_lm, labels,
            logits_tools if self.config.use_tool_head else None, tool_labels,
            predicted_values if self.config.use_value_head else None, values,
            logits_plan if self.config.plan_decoder else None, plan_labels
        )
        
        outputs["losses"] = losses
        
        # Return intermediate states if requested
        if return_intermediates:
            outputs["internals"] = {
                "hidden_states": hidden_states,
                "reasoning_states": reasoning_states
            }
        
        return outputs
    
    def _compute_losses(
        self,
        logits_lm: torch.Tensor,
        labels: Optional[torch.Tensor],
        logits_tools: Optional[torch.Tensor],
        tool_labels: Optional[torch.Tensor],
        predicted_values: Optional[torch.Tensor],
        values: Optional[torch.Tensor],
        logits_plan: Optional[torch.Tensor],
        plan_labels: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses"""
        losses = {}
        
        # Language modeling loss (causal)
        if labels is not None:
            # Shift for causal LM: predict next token
            lm_loss = F.cross_entropy(
                logits_lm[:, :-1].contiguous().view(-1, logits_lm.size(-1)),
                labels[:, 1:].contiguous().view(-1)
            )
            losses["lm"] = lm_loss
        else:
            losses["lm"] = torch.tensor(0.0, device=logits_lm.device)
        
        # Tool selection loss
        if logits_tools is not None and tool_labels is not None:
            tool_loss = F.cross_entropy(logits_tools, tool_labels)
            losses["tool"] = tool_loss
        else:
            losses["tool"] = torch.tensor(0.0, device=logits_lm.device)
        
        # Value prediction loss
        if predicted_values is not None and values is not None:
            value_loss = F.mse_loss(predicted_values, values)
            losses["value"] = value_loss
        else:
            losses["value"] = torch.tensor(0.0, device=logits_lm.device)
        
        # Plan generation loss
        if logits_plan is not None and plan_labels is not None:
            plan_loss = F.cross_entropy(
                logits_plan.view(-1, logits_plan.size(-1)),
                plan_labels.view(-1)
            )
            losses["plan"] = plan_loss
        else:
            losses["plan"] = torch.tensor(0.0, device=logits_lm.device)
        
        # Combine reasoning losses
        reasoning_loss = (
            losses["tool"] + 
            losses["value"] + 
            losses["plan"]
        )
        losses["reasoning"] = reasoning_loss
        
        # Total loss
        total_loss = (
            self.config.lambda_lm * losses["lm"] + 
            self.config.lambda_plan * reasoning_loss
        )
        losses["total"] = total_loss
        
        return losses
    
    def get_alpha_gates(self) -> Optional[torch.Tensor]:
        """Get alpha gate values from EMA aggregator (for interpretability)"""
        if hasattr(self.lpath.aggregator, 'alpha_params'):
            return torch.sigmoid(self.lpath.aggregator.alpha_params)
        return None
    
    def freeze_base_transformer(self):
        """Freeze base transformer parameters"""
        for param in self.transformer.parameters():
            param.requires_grad = False
    
    def unfreeze_last_k_layers(self, k: int):
        """Unfreeze last k layers of transformer"""
        if k > 0:
            for i, block in enumerate(self.transformer.blocks):
                if i >= len(self.transformer.blocks) - k:
                    for param in block.parameters():
                        param.requires_grad = True 