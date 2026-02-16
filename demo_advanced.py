"""
Updraft-LM Advanced Features Demo
This script demonstrates the powerful new features added to Updraft-LM
"""

import torch
from pathlib import Path

print("=" * 80)
print("Updraft-LM Advanced Features Demo")
print("=" * 80)

# ============================================================================
# 1. Enhanced Configuration
# ============================================================================
print("\n1. Enhanced Configuration")
print("-" * 40)

from config import GPT1Config

config = GPT1Config()
print(f"✓ Created config with {len(config.to_dict())} parameters")
print(f"  - Mixed precision: {config.mixed_precision}")
print(f"  - Gradient accumulation: {config.gradient_accumulation_steps} steps")
print(f"  - LR scheduler: {config.lr_scheduler}")
print(f"  - Optimizer: {config.optimizer}")

# You can customize the config
config.mixed_precision = True
config.gradient_accumulation_steps = 4
config.lr_scheduler = "cosine"
config.learning_rate = 3e-4
print("\n✓ Customized training configuration")

# ============================================================================
# 2. Advanced Generation Strategies
# ============================================================================
print("\n2. Advanced Generation Strategies")
print("-" * 40)

from advanced_generation import GenerationConfig

gen_config = GenerationConfig(
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    length_penalty=1.0,
    num_beams=4,
    early_stopping=True
)

print("✓ Created advanced generation config")
print(f"  - Max length: {gen_config.max_length}")
print(f"  - Temperature: {gen_config.temperature}")
print(f"  - Top-k: {gen_config.top_k}")
print(f"  - Top-p: {gen_config.top_p}")
print(f"  - Repetition penalty: {gen_config.repetition_penalty}")
print(f"  - Num beams: {gen_config.num_beams}")

# ============================================================================
# 3. Model Creation and Utilities
# ============================================================================
print("\n3. Model Creation and Utilities")
print("-" * 40)

from model.gpt1 import GPT1Model
from utils import count_parameters, estimate_model_size, format_time

# Create a smaller model for demo
demo_config = GPT1Config()
demo_config.n_layers = 4
demo_config.n_heads = 8
demo_config.d_model = 512
demo_config.d_ff = 2048

model = GPT1Model(demo_config)
print(f"✓ Created demo model")

# Count parameters
param_count = count_parameters(model, trainable_only=True)
print(f"  - Trainable parameters: {param_count:,}")

# Estimate model size
size_info = estimate_model_size(model)
print(f"  - Model size: {size_info['total_mb']:.2f} MB")
print(f"  - Parameters size: {size_info['parameters_mb']:.2f} MB")

# Format time example
training_time = 3661  # seconds
print(f"  - Example training time: {format_time(training_time)}")

# ============================================================================
# 4. Model Export Capabilities
# ============================================================================
print("\n4. Model Export Capabilities")
print("-" * 40)

from export import get_model_info

model_info = get_model_info(model)
print("✓ Model information:")
print(f"  - Total parameters: {model_info['total_parameters']:,}")
print(f"  - Trainable parameters: {model_info['trainable_parameters']:,}")
print(f"  - Model size: {model_info['size_mb']:.2f} MB")

# Note: Actual export would require sample inputs
print("\n✓ Export capabilities available:")
print("  - ONNX export: export_to_onnx(model, 'model.onnx', sample_input)")
print("  - TorchScript: export_to_torchscript(model, 'model.pt', sample_input)")
print("  - Quantization: quantize_model(model, quantization_type='dynamic')")

# ============================================================================
# 5. Advanced Sampling Techniques
# ============================================================================
print("\n5. Advanced Sampling Techniques")
print("-" * 40)

from advanced_generation import (
    apply_top_k_filtering,
    apply_top_p_filtering,
    apply_repetition_penalty,
    sample_with_temperature
)

# Create dummy logits
logits = torch.randn(1, 100)  # batch_size=1, vocab_size=100

# Apply top-k filtering
logits_k = apply_top_k_filtering(logits.clone(), top_k=10)
print(f"✓ Top-k filtering: keeps top {10} tokens")

# Apply top-p filtering
logits_p = apply_top_p_filtering(logits.clone(), top_p=0.9)
print(f"✓ Top-p filtering: keeps tokens with cumulative prob < {0.9}")

# Temperature sampling
sample = sample_with_temperature(logits.clone(), temperature=0.8)
print(f"✓ Temperature sampling: sampled token ID {sample.item()}")

# ============================================================================
# 6. Configuration Management
# ============================================================================
print("\n6. Configuration Management")
print("-" * 40)

from utils import save_config, format_time

# Create temporary directory for demo
import tempfile
temp_dir = tempfile.mkdtemp()
config_path = Path(temp_dir) / "config.json"

# Save configuration
save_config(config, config_path, format="json")
print(f"✓ Saved configuration to {config_path}")

# Config can be loaded back with load_config
print(f"✓ Configuration can be loaded with load_config()")

# ============================================================================
# 7. Logging and Monitoring
# ============================================================================
print("\n7. Logging and Monitoring")
print("-" * 40)

from utils import setup_logging

logger = setup_logging(level=20)  # INFO level
logger.info("Logging system initialized")
print("✓ Logging framework configured")
print("  - Console logging: enabled")
print("  - File logging: optional (pass log_file parameter)")
print("  - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL")

# ============================================================================
# 8. TUI Features
# ============================================================================
print("\n8. Text User Interface (TUI)")
print("-" * 40)

print("✓ Launch TUI with: python main.py tui")
print("\nTUI Features:")
print("  - Interactive text generation with live preview")
print("  - Real-time training monitoring")
print("  - Model configuration editor")
print("  - Checkpoint management")
print("  - Beautiful terminal visualizations")
print("\nKeyboard Shortcuts:")
print("  - g: Switch to Generate tab")
print("  - t: Switch to Train tab")
print("  - m: Switch to Model Info tab")
print("  - q: Quit application")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("✅ All advanced features demonstrated successfully!")
print("=" * 80)

print("\nKey Features:")
features = [
    "Enhanced configuration with 40+ parameters",
    "Advanced generation strategies (beam search, nucleus sampling, etc.)",
    "Model export to ONNX, TorchScript, and quantized formats",
    "Comprehensive utilities for training and evaluation",
    "Type hints and improved code quality",
    "Powerful Text User Interface (TUI)",
    "Logging and monitoring capabilities",
    "Pre-commit hooks and CI/CD setup"
]

for i, feature in enumerate(features, 1):
    print(f"  {i}. {feature}")

print("\nNext Steps:")
print("  1. Launch the TUI: python main.py tui")
print("  2. Train a model: python main.py train --dataset wikitext")
print("  3. Generate text: python main.py generate --checkpoint model.pt --prompt 'Hello'")
print("  4. Export a model: see export.py for examples")
print("  5. Read DEVELOPERS.md for contribution guidelines")

print("\n" + "=" * 80)
