<div align="center">

```
██╗   ██╗██████╗ ██████╗ ██████╗  █████╗ ███████╗████████╗    ██╗     ███╗   ███╗
██║   ██║██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝    ██║     ████╗ ████║
██║   ██║██████╔╝██║  ██║██████╔╝███████║█████╗     ██║       ██║     ██╔████╔██║
██║   ██║██╔═══╝ ██║  ██║██╔══██╗██╔══██║██╔══╝     ██║       ██║     ██║╚██╔╝██║
╚██████╔╝██║     ██████╔╝██║  ██║██║  ██║██║        ██║       ███████╗██║ ╚═╝ ██║
 ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝       ╚══════╝╚═╝     ╚═╝
```

### Advanced LLaMA-Style Language Model Architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/) [![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test_model.py)

**Version 2.0.0 · Built from scratch · Production-Ready Framework**

</div>

### Overview

Updraft-LM is a robust, clean, and advanced implementation of a causal language model framework inspired by the LLaMA architecture. Designed for research and production workflows, it utilizes state-of-the-art transformer components including Rotary Positional Embeddings (RoPE), SwiGLU activation, RMSNorm, and Grouped-Query Attention (GQA).

### Installation

```bash
git clone https://github.com/yourusername/updraft-lm.git
cd updraft-lm
pip install -r requirements.txt
```

### Quick Start

Run the demo using the built-in minimal model instantiation:

```bash
./quickstart.sh
```

## Usage

Updraft-LM uses a rich terminal user interface for an enhanced monitoring experience during training and generation.

### Generate Text

```bash
python main.py generate \
    --checkpoint checkpoints/pretrained_model.pt \
    --prompt "Once upon a time" \
    --max-length 100 \
    --temperature 0.8 \
    --top-k 50
```

Or programmatically in Python:

```python
from generator import load_model_for_inference

generator = load_model_for_inference('checkpoints/pretrained_model.pt')
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
    --learning-rate 2.5e-4 \
    --max-seq-len 512
```

### Interactive Mode

```bash
python main.py interactive \
    --checkpoint checkpoints/pretrained_model.pt \
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
        ╔══════════════╧═══════════════╗
        ║   Transformer Block × 12     ║
        ║  ┌─────────────────────────┐ ║
        ║  │        RMSNorm          │ ║
        ║  └──────────┬──────────────┘ ║
        ║             ▼                ║
        ║  ┌─────────────────────────┐ ║
        ║  │ Grouped Query Attention │ ║
        ║  │ (12 q_heads, 4 kv_heads)│ ║
        ║  │    + Rotary Bias (RoPE) │ ║
        ║  └──────────┬──────────────┘ ║
        ║             │                ║
        ║             ▼                ║
        ║  ┌─────────────────────────┐ ║
        ║  │        RMSNorm          │ ║
        ║  └──────────┬──────────────┘ ║
        ║             ▼                ║
        ║  ┌─────────────────────────┐ ║
        ║  │   Feed-Forward Network  │ ║
        ║  │   (SwiGLU Activation)   │ ║
        ║  └─────────────────────────┘ ║
        ╚══════════════╤═══════════════╝
                       │
                       ▼
              ┌────────────────┐
              │     RMSNorm    │
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

| Component              | Value  | Details                                        |
| ---------------------- | ------ | ---------------------------------------------- |
| Layers                 | 12     | LLaMA-style decoder blocks                     |
| Attention Heads        | 12     | Query heads                                    |
| KV Heads               | 4      | Grouped Query Attention (GQA)                   |
| Embedding Dimension    | 768    | d_model                                        |
| Feed-Forward Dimension | 3072   | SwiGLU hidden projection                       |
| Supported Context      | 512    | Configurable up to larger windows via RoPE     |
| Vocabulary Size        | 50,257 | Configurable context tokens                    |
| Normalization          | 1e-5   | RMSNorm epsilon parameter                      |

For detailed architectural logic, please view the mathematical derivations in `MATH.md`.
