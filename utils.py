"""
Utility functions for Updraft-LM
Includes training helpers, evaluation metrics, and model management
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import logging
import json
import yaml


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("updraft-lm")
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_perplexity(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """Calculate perplexity of model on dataloader"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits, loss = model(input_ids, labels)
            
            mask = (labels != -100)
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BLEU and other metrics"""
    from collections import defaultdict
    
    def tokenize(text: str) -> List[str]:
        return text.lower().split()
    
    def calculate_bleu_n(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
        
        if len(pred_ngrams) == 0:
            return 0.0
        
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        matches = sum((pred_counter & ref_counter).values())
        total = len(pred_ngrams)
        
        return matches / total if total > 0 else 0.0
    
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize(pred)
        ref_tokens = tokenize(ref)
        
        bleu_1 = calculate_bleu_n(pred_tokens, ref_tokens, 1)
        bleu_2 = calculate_bleu_n(pred_tokens, ref_tokens, 2)
        bleu_3 = calculate_bleu_n(pred_tokens, ref_tokens, 3)
        bleu_4 = calculate_bleu_n(pred_tokens, ref_tokens, 4)
        
        bleu = (bleu_1 * bleu_2 * bleu_3 * bleu_4) ** 0.25
        bleu_scores.append(bleu)
    
    return {
        'bleu': np.mean(bleu_scores),
        'bleu_1': np.mean([calculate_bleu_n(tokenize(p), tokenize(r), 1) for p, r in zip(predictions, references)]),
        'bleu_2': np.mean([calculate_bleu_n(tokenize(p), tokenize(r), 2) for p, r in zip(predictions, references)]),
    }


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def save_model(model: nn.Module, path: Union[str, Path], metadata: Optional[Dict] = None) -> None:
    """Save model with optional metadata"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(
    model: nn.Module,
    path: Union[str, Path],
    device: Union[str, torch.device] = 'cpu',
    strict: bool = True
) -> Tuple[nn.Module, Dict]:
    """Load model from checkpoint"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        metadata = checkpoint.get('metadata', {})
    else:
        model.load_state_dict(checkpoint, strict=strict)
        metadata = {}
    
    print(f"Model loaded from {path}")
    return model, metadata


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    warmup_steps: int,
    min_lr: float = 0.0
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler"""
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=min_lr
        )
    elif scheduler_type == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / optimizer.defaults['lr'],
            total_iters=num_training_steps - warmup_steps
        )
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    elif scheduler_type == "polynomial":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=num_training_steps - warmup_steps,
            power=1.0
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_config(config: object, path: Union[str, Path], format: str = "json") -> None:
    """Save configuration to file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
    
    if format == "json":
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    elif format == "yaml":
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_config(path: Union[str, Path], config_class: type) -> object:
    """Load configuration from file"""
    path = Path(path)
    
    if path.suffix == ".json":
        with open(path, 'r') as f:
            config_dict = json.load(f)
    elif path.suffix in [".yaml", ".yml"]:
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")
    
    if hasattr(config_class, 'from_dict'):
        return config_class.from_dict(config_dict)
    else:
        return config_class(**config_dict)


def estimate_model_size(model: nn.Module) -> Dict[str, float]:
    """Estimate model size in memory"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'parameters_mb': param_size / 1024**2,
        'buffers_mb': buffer_size / 1024**2,
        'total_mb': total_size / 1024**2,
        'total_gb': total_size / 1024**3,
    }
