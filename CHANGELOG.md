# Changelog

All notable changes to Updraft-LM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-16

### Added
- **Powerful TUI (Text User Interface)**: Interactive terminal interface powered by Textual
  - Generate tab with live text generation
  - Train tab with training monitoring
  - Model info tab with architecture details
  - Keyboard shortcuts for navigation (g/t/m/q)
  - Real-time model statistics display
  - Beautiful terminal visualizations

- **Advanced Generation Strategies**:
  - Beam search implementation
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Repetition penalty
  - Constrained generation with forbidden/required tokens
  - Length penalty support
  - Early stopping

- **Model Export & Optimization**:
  - ONNX export functionality
  - TorchScript export
  - Dynamic quantization support
  - Model size estimation utilities
  - Torch compile optimization

- **Enhanced Training Features**:
  - Mixed precision training support
  - Gradient accumulation configuration
  - Multiple optimizer support (AdamW, Adam, SGD)
  - Learning rate schedulers (cosine, linear, polynomial, constant)
  - Configurable warmup steps
  - Checkpoint management with rotation

- **Code Quality Tools**:
  - Black code formatting configuration
  - isort import sorting
  - Flake8 linting rules
  - Pylint configuration
  - Mypy type checking setup
  - pytest testing framework
  - Pre-commit hooks configuration

- **Developer Experience**:
  - Comprehensive type hints
  - Enhanced logging framework
  - Configuration save/load utilities
  - Time formatting helpers
  - Model size estimation
  - Setup.py for package installation
  - CLI tool script for common operations
  - GitHub Actions CI workflow

- **Documentation**:
  - Updated README with comprehensive examples
  - DEVELOPERS.md guide for contributors
  - Code quality badge
  - TUI usage instructions
  - Advanced feature documentation
  - Export and optimization examples

### Changed
- Upgraded PyTorch to 2.1.0+
- Upgraded Transformers to 4.35.0+
- Upgraded all dependencies to latest stable versions
- Enhanced GPT1Config with 40+ configuration parameters
- Improved utils.py with type hints and better error handling
- Updated README with modern badges and feature list

### Fixed
- Added proper error handling in model loading
- Improved device selection logic (CUDA, MPS, CPU)
- Better checkpoint validation

## [0.1.0] - Initial Release

### Added
- Basic GPT-1 architecture implementation
- Training from scratch capability
- Text generation with basic sampling
- GPT-2 tokenizer integration
- Command-line interface
- Model checkpointing
- TensorBoard logging support
- Weights & Biases integration
- Basic testing suite

---

## Upgrade Guide

### From 0.1.0 to 1.0.0

#### New TUI Mode
Launch the new powerful TUI:
```bash
python main.py tui
```

#### Configuration Updates
If you have custom configs, update them with new parameters:
```python
from config import GPT1Config

config = GPT1Config()
# New parameters available:
config.mixed_precision = True
config.gradient_accumulation_steps = 4
config.lr_scheduler = "cosine"
```

#### Generation Updates
Use advanced generation strategies:
```python
from advanced_generation import GenerationConfig, generate_with_constraints

gen_config = GenerationConfig(
    max_length=100,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    num_beams=4
)
```

#### Export Functionality
Export your models to different formats:
```python
from export import export_to_onnx, quantize_model

export_to_onnx(model, "model.onnx", sample_input)
quantized = quantize_model(model)
```
