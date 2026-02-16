<div align="center">

```
██╗   ██╗██████╗ ██████╗ ██████╗  █████╗ ███████╗████████╗    ██╗     ███╗   ███╗
██║   ██║██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝╚══██╔══╝    ██║     ████╗ ████║
██║   ██║██████╔╝██║  ██║██████╔╝███████║█████╗     ██║       ██║     ██╔████╔██║
██║   ██║██╔═══╝ ██║  ██║██╔══██╗██╔══██║██╔══╝     ██║       ██║     ██║╚██╔╝██║
╚██████╔╝██║     ██████╔╝██║  ██║██║  ██║██║        ██║       ███████╗██║ ╚═╝ ██║
 ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝       ╚══════╝╚═╝     ╚═╝
```

### A Powerful Transformer with Modern TUI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/) [![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test_model.py) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**117M parameters · Built from scratch · With Advanced Features · Powerful TUI**

</div>

## ✨ Features

- 🚀 **Powerful Text-Based UI (TUI)**: Beautiful, interactive terminal interface powered by Textual
- 🎯 **Advanced Generation**: Beam search, nucleus sampling, top-k/top-p sampling
- 📦 **Model Export**: ONNX, TorchScript, and quantization support
- 🔧 **Flexible Training**: Mixed precision, gradient accumulation, multiple optimizers
- 📊 **Comprehensive Monitoring**: TensorBoard, Weights & Biases integration
- 🎨 **Code Quality**: Full type hints, formatted with Black, linted with Flake8
- 🧪 **Well Tested**: Comprehensive test suite with pytest

## 📦 Installation

```bash
git clone https://github.com/desenyon/updraft-lm.git
cd updraft-lm
pip install -r requirements.txt
```

## 🎮 Quick Start

### Launch the TUI (Recommended!)

Experience the powerful Text User Interface:

```bash
python main.py tui
```

The TUI provides:
- Interactive text generation with live preview
- Real-time training monitoring
- Model configuration editor
- Checkpoint management
- Beautiful terminal visualizations

### Traditional CLI

Run the demo with pretrained GPT-2 weights:

```bash
./quickstart.sh
```

## 📖 Usage

### 1. Text Generation (TUI Mode)

```bash
python main.py tui
```

Navigate with keyboard shortcuts:
- `g`: Switch to Generate tab
- `t`: Switch to Train tab  
- `m`: Switch to Model Info tab
- `q`: Quit application

### 2. Generate Text (CLI)

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

### 3. Train from Scratch

```bash
python main.py train \
    --dataset wikitext \
    --epochs 5 \
    --batch-size 64 \
    --learning-rate 2.5e-4
```

### 4. Interactive Mode

```bash
python main.py interactive \
    --checkpoint checkpoints/pretrained_gpt2.pt \
    --temperature 0.8
```

### 5. Advanced Generation with Python

```python
from advanced_generation import GenerationConfig, generate_with_constraints
from model.gpt1 import GPT1Model
from config import GPT1Config
import torch

# Load model
config = GPT1Config()
model = GPT1Model(config)

# Configure generation
gen_config = GenerationConfig(
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    num_beams=4  # Use beam search
)

# Generate with constraints
input_ids = torch.tensor([[1, 2, 3]])  # Your input tokens
output = generate_with_constraints(
    model, 
    input_ids, 
    gen_config,
    forbidden_tokens=[123, 456],  # Tokens to avoid
    required_tokens=[789, 012]    # Tokens that must appear
)
```

### 6. Export Models

```python
from export import export_to_onnx, export_to_torchscript, quantize_model

# Export to ONNX
sample_input = torch.randint(0, 50257, (1, 10))
export_to_onnx(model, "model.onnx", sample_input)

# Export to TorchScript
export_to_torchscript(model, "model.pt", sample_input)

# Quantize for faster inference
quantized_model = quantize_model(model, quantization_type="dynamic")
```

## 🏗️ Architecture

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

## 🎯 Advanced Features

### Training Enhancements
- **Mixed Precision Training**: Automatic FP16/BF16 support for faster training
- **Gradient Accumulation**: Train with larger effective batch sizes
- **Learning Rate Schedulers**: Cosine, linear, polynomial, and constant schedules
- **Multiple Optimizers**: AdamW, Adam, SGD with configurable parameters
- **Gradient Clipping**: Prevent exploding gradients
- **Checkpoint Management**: Auto-save with configurable intervals

### Generation Strategies
- **Greedy Decoding**: Deterministic generation
- **Temperature Sampling**: Control randomness
- **Top-K Sampling**: Sample from top K tokens
- **Top-P (Nucleus) Sampling**: Dynamic vocabulary truncation
- **Beam Search**: Find high-probability sequences
- **Repetition Penalty**: Reduce repetitive outputs
- **Constrained Generation**: Force/forbid specific tokens

### Model Export & Optimization
- **ONNX Export**: Cross-platform deployment
- **TorchScript**: JIT compilation for production
- **Dynamic Quantization**: Reduce model size
- **Torch Compile**: PyTorch 2.0+ optimization

## 🛠️ Development

### Code Quality Tools

Format code:
```bash
black .
isort .
```

Lint code:
```bash
flake8 .
pylint *.py
```

Type checking:
```bash
mypy .
```

Run tests:
```bash
pytest
```

### Project Structure

```
updraft-lm/
├── tui/                    # Text User Interface
│   ├── __init__.py
│   └── app.py             # Main TUI application
├── model/                  # Model architecture
│   ├── gpt1.py            # GPT-1 model implementation
│   └── transformer.py     # Transformer blocks
├── data/                   # Data processing
│   ├── dataset.py         # Dataset loaders
│   └── tokenizer.py       # Tokenization
├── config.py              # Configuration classes
├── main.py                # CLI entry point
├── trainer.py             # Training loop
├── generator.py           # Text generation
├── advanced_generation.py # Advanced sampling strategies
├── export.py              # Model export utilities
├── utils.py               # Helper functions
├── requirements.txt       # Dependencies
├── pyproject.toml         # Tool configurations
└── README.md              # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with PyTorch and Transformers
- TUI powered by Textual and Rich
- Inspired by GPT-1 architecture
- Thanks to the open-source community

---

<div align="center">

Made with ❤️ by the Updraft-LM team

</div>
