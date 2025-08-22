# L-Former: Transformer with L-shaped Progressive Aggregation Side-Path

L-Former is a novel Transformer architecture that introduces an **L-shaped progressive aggregation side-path** that taps every layer, builds a compact reasoning state $S$, and exposes a **ReasoningHead** alongside the normal **LMHead**.

## 🎯 Concept

L-Former combines the power of Transformers with **progressive layer aggregation** inspired by Deep Layer Aggregation (DLA). The key insight is to:

1. **Tap every layer** of the base Transformer to extract intermediate representations
2. **Progressively aggregate** these representations using learnable gates (EMA-style) or GRU cells
3. **Build a compact reasoning state** $S^L$ that captures hierarchical information
4. **Supervise reasoning** through specialized heads for tool selection, value prediction, and plan generation

### Mathematical Formulation

The core aggregation mechanism uses **EMA-style updates**:

$$\alpha^l = \sigma(a^l) \quad \text{(learnable gates, initialized to ~0.12)}$$

$$S^l = (1-\alpha^l) \cdot S^{l-1} + \alpha^l \cdot \phi(z^l)$$

Where:
- $z^l$ is the projected tap from layer $l$
- $\phi$ is an MLP: $d_{side} \rightarrow 4d_{side} \rightarrow d_{side}$
- $S^l$ is the reasoning state at layer $l$

## 🏗️ Architecture

```
Input → [Transformer Layers] → LMHead
   ↓           ↓
   ↓      [Tap Projector] → [EMA/GRU Aggregator] → ReasoningHead
   ↓           ↓                    ↓
   ↓      [LayerNorm]           [S^1, S^2, ..., S^L]
   ↓           ↓
   ↓      [Dropout]
   ↓
   ↓      [Pooling: mean/cls/attn]
   ↓
   ↓      [z^1, z^2, ..., z^L]
```

### Key Components

- **Base Transformer**: GPT-style decoder with causal masking
- **L-Path**: Progressive aggregation side-path with multiple aggregator options
- **Reasoning Heads**: Tool selection, value prediction, and plan generation
- **Dual Training**: Language modeling + reasoning supervision

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Gnav3852/L-former.git
cd L-former
pip install torch torchvision torchaudio
```

### Basic Usage

```python
from lformer import ModelConfig, LFormer

# Create config
config = ModelConfig.tiny()  # Small model for testing

# Create model
model = LFormer(config)

# Forward pass
input_ids = torch.randint(0, 1000, (2, 10))  # [batch_size, seq_len]
outputs = model(input_ids)

# Access outputs
logits_lm = outputs["logits_lm"]           # Language modeling
logits_tools = outputs["logits_tools"]     # Tool selection
values = outputs["values"]                  # Value prediction
losses = outputs["losses"]                 # All losses
```

### Training

```bash
# Phase A: Train side-path + heads (freeze base transformer)
python train.py --tiny --phase_a_steps 100

# Phase B: Unfreeze last k layers
python train.py --tiny --last_k_unfreeze 2

# Use different aggregator
python train.py --tiny --aggregator gru

# Enable blending into LM head
python train.py --tiny --blend_into_lm
```

### Evaluation

```bash
python eval.py --checkpoint ./checkpoints/final_model.pt
```

## ⚙️ Configuration

### ModelConfig Options

```python
@dataclass
class ModelConfig:
    # Base Transformer
    vocab_size: int = 5000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    
    # L-Path
    d_side: int = 128
    tap_every: int = 1
    sequence_wise: bool = True
    aggregator: Dict = {"type": "ema"}  # "ema", "gru", "attn"
    
    # Training
    detach_taps: bool = True
    lambda_lm: float = 1.0
    lambda_plan: float = 0.2
    
    # Heads
    n_tools: int = 5
    use_value_head: bool = True
    use_tool_head: bool = True
    plan_decoder: bool = False
    
    # Advanced
    tree_checkpoint: int = 0
    blend_into_lm: bool = False
    last_k_unfreeze: int = 0
```

### Aggregator Types

1. **EMA** (default): Exponential moving average with learnable gates
2. **GRU**: Gated recurrent unit treating depth as time
3. **Attention**: Cross-attention to memory of previous taps

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python tests/test_shapes.py
python tests/test_forward_losses.py
python tests/test_freeze_and_detach.py
```

## 📊 Training Phases

### Phase A: Side-Path Training
- Freeze base Transformer parameters
- Train L-Path aggregator and reasoning heads
- Use `detach_taps=True` to prevent gradient flow to base

### Phase B: Fine-tuning
- Optionally unfreeze last k layers
- Train entire model end-to-end
- Blend reasoning state into LM head if desired

## 🔬 Research Features

- **Tree Checkpoints**: Periodic aggregation checkpoints for hierarchical reasoning
- **Depth Attention**: Memory-based attention to previous layer representations
- **Gated Blending**: Learnable gate to blend $S^L$ into LM head
- **Sparsity Control**: L1/L2 regularization on alpha gates

## 📈 Expected Behavior

During training, you should observe:

1. **Alpha gates** start near 0.12 and learn upward
2. **Tool accuracy** improves above random baseline
3. **Reasoning loss** decreases alongside LM loss
4. **Gradient sparsity** in early layers during Phase A

## 🎨 ASCII Architecture Diagram

```
Input: "What is 5 + 3?"
     ↓
┌─────────────────────────────────────────────────────────────┐
│                    Base Transformer                        │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  │
│  │ L1  │→│ L2  │→│ L3  │→│ L4  │→│ L5  │→│ L6  │  │
│  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  │
│     ↓        ↓        ↓        ↓        ↓        ↓       │
│   H¹       H²       H³       H⁴       H⁵       H⁶      │
└─────────────────────────────────────────────────────────────┘
     ↓        ↓        ↓        ↓        ↓        ↓
┌─────────────────────────────────────────────────────────────┐
│                    L-Path (Side)                          │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  │
│  │Tap  │→│Tap  │→│Tap  │→│Tap  │→│Tap  │→│Tap  │  │
│  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  │
│     ↓        ↓        ↓        ↓        ↓        ↓       │
│   z¹       z²       z³       z⁴       z⁵       z⁶      │
│     ↓        ↓        ↓        ↓        ↓        ↓       │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           Progressive Aggregator                    │  │
│  │  S⁰ → S¹ → S² → S³ → S⁴ → S⁵ → S⁶                │  │
│  │  (EMA: S^l = (1-α^l)S^{l-1} + α^l φ(z^l))        │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
     ↓                    ↓                    ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  LMHead     │    │ ToolHead    │    │ ValueHead   │
│ (H⁶)       │    │ (S⁶)        │    │ (S⁶)        │
└─────────────┘    └─────────────┘    └─────────────┘
     ↓                    ↓                    ↓
  Next token         Tool selection      Value prediction
```

## 🙏 Acknowledgments

- Inspired by Deep Layer Aggregation (DLA)
- Built on PyTorch and modern Transformer architectures
- Thanks to the open-source AI community
