# Updraft-LM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test_model.py)

A GPT-1 level transformer language model built from scratch. 117M parameters, complete with mathematical implementations and pretrained weight support.

Every component—from scaled dot-product attention to positional encoding—is implemented from first principles with no high-level abstractions.

## Installation

```bash
git clone https://github.com/yourusername/updraft-lm.git
cd updraft-lm
pip install -r requirements.txt
```

## Quick Start

Run the demo with pretrained GPT-2 weights:

```bash
./quickstart.sh
```

## Usage

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

## Architecture

### Model Specifications

| Component | Value | Details |
|-----------|-------|---------|
| Layers | 12 | Transformer decoder blocks |
| Attention Heads | 12 | 64 dimensions each |
| Embedding Dimension | 768 | d_model |
| Feed-Forward Dimension | 3072 | 4x embedding dimension |
| Max Sequence Length | 512 | Maximum context window |
| Vocabulary Size | 50,257 | GPT-2 tokenizer |
| Total Parameters | ~117M | Comparable to GPT-1 |

### Mathematical Implementation

**Scaled Dot-Product Attention**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**Multi-Head Attention**
```
MultiHead(Q, K, V) = Concat(head₁, ..., head₁₂)W^O
where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

**Positional Encoding**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Complete mathematical derivations available in [MATH.md](MATH.md).

## Project Structure

```
updraft-lm/
├── model/
│   ├── transformer.py      # Attention, feed-forward, transformer blocks
│   └── gpt1.py            # Complete GPT-1 model
├── data/
│   ├── tokenizer.py       # TikToken wrapper
│   └── dataset.py         # Data loading and preprocessing
├── config.py              # Model configuration
├── trainer.py             # Training loop and optimization
├── generator.py           # Text generation
├── utils.py               # Utilities and metrics
├── main.py                # CLI interface
├── load_pretrained.py     # Pretrained weight loading
├── test_model.py          # Test suite
└── MATH.md                # Mathematical documentation
```

## Testing

```bash
python test_model.py
```

All core functionality tested:
- Model creation and initialization
- Tokenizer encoding/decoding
- Attention mechanism
- Forward pass
- Text generation

## Features

- Complete transformer implementation from scratch
- Multi-head self-attention with scaled dot-product
- Sinusoidal positional encoding
- Layer normalization with residual connections
- AdamW optimizer with weight decay
- Learning rate warmup and cosine annealing
- Gradient clipping
- Multiple sampling strategies (greedy, top-k, top-p, beam search)
- Temperature-controlled generation
- Pretrained GPT-2 weight loading
- Checkpoint saving and loading
- HuggingFace dataset support

## Performance

**Computational Complexity**
- Self-Attention: O(n² · d)
- Feed-Forward: O(n · d · d_ff)
- Full Model: O(L · (n² · d + n · d · d_ff))
- Total: ~143 billion FLOPs per forward pass

**Memory Requirements**
- Model Parameters: ~468 MB (FP32)
- Training (batch 64×512): ~1.5 GB

## License

MIT
