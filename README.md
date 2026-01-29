<div align="center">

```
██╗   ██╗██████╗ ██████╗ ██████╗  █████╗ ███████╗████████╗    ██╗     ███╗   ███╗
██║   ██║██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝    ██║     ████╗ ████║
██║   ██║██████╔╝██║  ██║██████╔╝███████║█████╗     ██║       ██║     ██╔████╔██║
██║   ██║██╔═══╝ ██║  ██║██╔══██╗██╔══██║██╔══╝     ██║       ██║     ██║╚██╔╝██║
╚██████╔╝██║     ██████╔╝██║  ██║██║  ██║██║        ██║       ███████╗██║ ╚═╝ ██║
 ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝       ╚══════╝╚═╝     ╚═╝
```

### A Simple Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/) [![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test_model.py)

**117M parameters · Built from scratch · With Some Math**

</div>

### Installations

```bash
git clone https://github.com/yourusername/updraft-lm.git
cd updraft-lm
pip install -r requirements.txt
```

### Quick Start

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
