<div align="center">

```
██╗   ██╗██████╗ ██████╗ ██████╗  █████╗ ███████╗████████╗    ██╗     ███╗   ███╗
██║   ██║██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝    ██║     ████╗ ████║
██║   ██║██████╔╝██║  ██║██████╔╝███████║█████╗     ██║       ██║     ██╔████╔██║
██║   ██║██╔═══╝ ██║  ██║██╔══██╗██╔══██║██╔══╝     ██║       ██║     ██║╚██╔╝██║
╚██████╔╝██║     ██████╔╝██║  ██║██║  ██║██║        ██║       ███████╗██║ ╚═╝ ██║
 ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝       ╚══════╝╚═╝     ╚═╝
```

### A GPT-1 Level Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test_model.py)

**117M parameters · Built from scratch · Mathematical rigor**

</div>

---

---

## Installation

```bash
git clone https://github.com/yourusername/updraft-lm.git
cd updraft-lm
pip install -r requirements.txt
```

<div align="center">

### Quick Start

</div>

Run the demo with pretrained GPT-2 weights:

```bash
./quickstart.sh
```

<br>

<div align="center">

## Usage

</div>

### Generate Text

```bash
python main.py generate \
    --checkpoint checkpoints/pretrained_gpt2.pt \
    --prompt "Once upon a time" \
    --max-length 100 \
    --temperature 0.8 \
    --top-k 50
```

Or in Python:

```python
from generator import load_model_for_inference

generator = load_model_for_inference('checkpoints/pretrained_gpt2.pt')
outputs = generator.generate(
    "The future of artificial intelligence",
    max_length=100,
    temperature=0.8,
    top_k=50
)
print(outputs[0])
```

### Train from Scratch

```bash
python main.py train \
    --dataset wikitext \
    --epochs 5 \
    --batch-size 64 \
    --learning-rate 2.5e-4
```

### Interactive Mode

```bash
python main.py interactive \
    --checkpoint checkpoints/pretrained_gpt2.pt \
    --temperature 0.8
```

---

<div align="center">

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT TOKEN IDS                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Token Embedding │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────────┐
              │ Positional Encoding │
              └────────┬───────────┘
                       │
        ╔══════════════╧═══════════════╗
        ║   Transformer Block × 12     ║
        ║  ┌─────────────────────────┐ ║
        ║  │  Multi-Head Attention   │ ║
        ║  │    (12 heads × 64 dim)  │ ║
        ║  └──────────┬──────────────┘ ║
        ║             │                 ║
        ║             ▼                 ║
        ║  ┌─────────────────────────┐ ║
        ║  │   Feed-Forward Network  │ ║
        ║  │      (768 → 3072)       │ ║
        ║  └─────────────────────────┘ ║
        ╚══════════════╤═══════════════╝
                       │
                       ▼
              ┌────────────────┐
              │  Layer Norm     │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │   LM Head       │
              └────────┬───────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │         LOGITS               │
        │    (vocabulary: 50,257)      │
        └──────────────────────────────┘
```

</div>

### Model Specifications

| Component              | Value  | Details                    |
| ---------------------- | ------ | -------------------------- |
| Layers                 | 12     | Transformer decoder blocks |
| Attention Heads        | 12     | 64 dimensions each         |
| Embedding Dimension    | 768    | d_model                    |
| Feed-Forward Dimension | 3072   | 4x embedding dimension     |
| Max Sequence Length    | 512    | Maximum context window     |
| Vocabulary Size        | 50,257 | GPT-2 tokenizer            |
| Total Parameters       | ~117M  | Comparable to GPT-1        |

### Mathematical Implementation

<table>
<tr>
<td width="50%">

**Scaled Dot-Product Attention**

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

</td>
<td width="50%">

**Multi-Head Attention**

```
MultiHead(Q, K, V) = Concat(head₁, ..., head₁₂)W^O
where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

</td>
</tr>
<tr>
<td colspan="2">

**Positional Encoding**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

</td>
</tr>
</table>

Complete mathematical derivations available in [MATH.md](MATH.md).

---

<div align="center">

## Project Structure

</div>

```
updraft-lm/
│
├── model/
│   ├── transformer.py      # Attention, feed-forward, transformer blocks
│   └── gpt1.py            # Complete GPT-1 model
│
├── data/
│   ├── tokenizer.py       # TikToken wrapper
│   └── dataset.py         # Data loading and preprocessing
│
├── config.py              # Model configuration
├── trainer.py             # Training loop and optimization
├── generator.py           # Text generation
├── utils.py               # Utilities and metrics
├── main.py                # CLI interface
├── load_pretrained.py     # Pretrained weight loading
├── test_model.py          # Test suite
└── MATH.md                # Mathematical documentation
```

---

<div align="center">

## Testing

</div>

```bash
python test_model.py
```

All core functionality tested:

<div align="center">

| Test | Status |
|------|--------|
| Model creation and initialization | ✓ |
| Tokenizer encoding/decoding | ✓ |
| Attention mechanism | ✓ |
| Forward pass | ✓ |
| Text generation | ✓ |

**All tests passing**

</div>

---

<div align="center">

## Features

</div>

<table>
<tr>
<td width="50%" valign="top">

**Core Architecture**
- Multi-head self-attention
- Sinusoidal positional encoding
- Layer normalization
- Residual connections
- GELU activations

**Training**
- AdamW optimizer
- Weight decay
- Learning rate warmup
- Cosine annealing
- Gradient clipping

</td>
<td width="50%" valign="top">

**Generation**
- Greedy sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Beam search
- Temperature control

**Utilities**
- Pretrained GPT-2 weight loading
- Checkpoint management
- HuggingFace dataset support
- TikToken tokenizer

</td>
</tr>
</table>

---

<div align="center">

## Performance

</div>

<table>
<tr>
<td width="50%">

**Computational Complexity**

```
Self-Attention:  O(n² · d)
Feed-Forward:    O(n · d · d_ff)
Full Model:      O(L · (n² · d + n · d · d_ff))
```

**Total:** ~143 billion FLOPs per forward pass

</td>
<td width="50%">

**Memory Requirements**

```
Model Parameters:         ~468 MB (FP32)
Optimizer States:         ~936 MB
Training Batch (64×512):  ~96 MB
```

**Total Training:** ~1.5 GB

</td>
</tr>
</table>

---

<div align="center">

**MIT License**

Built with mathematical precision from first principles

</div>
