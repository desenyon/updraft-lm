# Developer Guide

## Getting Started

### Development Installation

```bash
# Clone the repository
git clone https://github.com/desenyon/updraft-lm.git
cd updraft-lm

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Project Structure

```
updraft-lm/
├── tui/                        # Text User Interface
│   ├── __init__.py
│   └── app.py                 # Main TUI application
├── model/                      # Model architecture
│   ├── __init__.py
│   ├── gpt1.py                # GPT-1 model
│   └── transformer.py         # Transformer blocks
├── data/                       # Data processing
│   ├── __init__.py
│   ├── dataset.py             # Dataset loaders
│   └── tokenizer.py           # Tokenization
├── config.py                   # Configuration
├── main.py                     # CLI entry point
├── trainer.py                  # Training loop
├── generator.py                # Text generation
├── advanced_generation.py      # Advanced sampling
├── export.py                   # Model export utilities
├── utils.py                    # Helper functions
├── test_model.py              # Tests
├── setup.py                    # Package setup
├── requirements.txt            # Dependencies
├── pyproject.toml             # Tool configurations
└── .pre-commit-config.yaml    # Pre-commit hooks
```

## Development Workflow

### 1. Code Style

We use several tools to maintain code quality:

#### Black (Code Formatting)
```bash
# Format all files
black .

# Check without modifying
black --check .
```

#### isort (Import Sorting)
```bash
# Sort imports
isort .

# Check without modifying
isort --check-only .
```

#### Flake8 (Linting)
```bash
# Lint all files
flake8 .

# Lint specific file
flake8 main.py
```

#### Mypy (Type Checking)
```bash
# Type check all files
mypy .

# Type check specific module
mypy model/
```

### 2. Running Tests

```bash
# Run all tests
python test_model.py

# With pytest (if installed)
pytest

# With coverage
pytest --cov=. --cov-report=html
```

### 3. Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

## Adding New Features

### Adding a New Generation Strategy

1. Add the strategy to `advanced_generation.py`:

```python
def my_new_sampling(logits: torch.Tensor, **kwargs) -> torch.Tensor:
    """Your new sampling method"""
    # Implementation
    return sampled_tokens
```

2. Update `GenerationConfig` if needed:

```python
@dataclass
class GenerationConfig:
    # ... existing fields
    my_new_param: float = 1.0
```

3. Add tests in `test_model.py` or create a new test file

4. Update documentation in README.md

### Adding a New TUI Feature

1. Add widget to `tui/app.py`:

```python
class MyNewWidget(Static):
    """Your new widget"""
    
    def compose(self) -> ComposeResult:
        yield Label("Hello from new widget")
```

2. Add to main app layout in `UpdraftTUI.compose()`

3. Add event handlers if needed

4. Test the TUI:
```bash
python main.py tui
```

### Adding Model Export Format

1. Add export function to `export.py`:

```python
def export_to_myformat(
    model: nn.Module,
    output_path: Path,
    **kwargs
) -> None:
    """Export to your format"""
    # Implementation
```

2. Add command-line support in `main.py`

3. Update README with usage example

## Architecture Details

### Model Architecture

The model follows GPT-1 architecture:
- Transformer decoder-only
- Multi-head self-attention
- Position-wise feed-forward networks
- Layer normalization
- Learned positional embeddings

Key files:
- `model/gpt1.py`: Main model class
- `model/transformer.py`: Transformer blocks
- `config.py`: Model configuration

### Training Pipeline

1. Data loading: `data/dataset.py`
2. Training loop: `trainer.py`
3. Optimization: Configured in `config.py`
4. Checkpointing: Handled by trainer
5. Logging: TensorBoard/Weights & Biases

### Generation Pipeline

1. Input tokenization
2. Model forward pass
3. Sampling strategy application
4. Output detokenization

Strategies in `advanced_generation.py`:
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Beam search
- Constrained generation

## Performance Optimization

### Mixed Precision Training

```python
config = GPT1Config()
config.mixed_precision = True
```

### Gradient Accumulation

```python
config = GPT1Config()
config.gradient_accumulation_steps = 4
```

### Model Optimization

```python
from export import optimize_for_inference

optimized_model = optimize_for_inference(model, sample_input, device)
```

## Debugging

### Debug Mode

```python
import logging
from utils import setup_logging

logger = setup_logging(log_file="debug.log", level=logging.DEBUG)
```

### Profiling

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    # Your code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Contributing

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests and linters
5. Commit with descriptive messages
6. Push to your fork
7. Create a pull request

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests

Example:
```
Add beam search generation strategy

- Implement BeamSearchGenerator class
- Add configuration parameters
- Update documentation
- Add tests

Closes #123
```

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`
5. Create GitHub release
6. Build and publish to PyPI (if applicable)

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Reinstall in development mode
pip install -e .
```

#### CUDA Out of Memory
```python
# Reduce batch size
config.batch_size = 32

# Enable gradient accumulation
config.gradient_accumulation_steps = 2
```

#### Slow Training
```python
# Enable mixed precision
config.mixed_precision = True

# Increase number of workers
config.num_workers = 8
```

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Textual Documentation](https://textual.textualize.io/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [GPT-1 Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

## Getting Help

- Open an issue on GitHub
- Check existing issues and documentation
- Review code examples in README.md

## License

MIT License - See LICENSE file for details
