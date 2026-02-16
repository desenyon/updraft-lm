from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GPT1Config:
    # Model Architecture
    vocab_size: int = 50257
    max_seq_len: int = 512
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    activation: str = "gelu"
    
    # Training Hyperparameters
    learning_rate: float = 2.5e-4
    batch_size: int = 64
    num_epochs: int = 5
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Advanced Training Features
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    max_grad_norm: float = 1.0
    
    # Learning Rate Scheduling
    lr_scheduler: str = "cosine"  # Options: "cosine", "linear", "constant", "polynomial"
    lr_scheduler_warmup_ratio: float = 0.1
    min_learning_rate: float = 1e-6
    
    # Optimization
    optimizer: str = "adamw"  # Options: "adamw", "adam", "sgd"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Checkpointing & Logging
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    keep_checkpoints: int = 3  # Keep only last N checkpoints
    
    # Model Export
    export_onnx: bool = False
    export_torchscript: bool = False
    
    # Generation Settings
    generation_max_length: int = 100
    generation_temperature: float = 1.0
    generation_top_k: Optional[int] = 50
    generation_top_p: Optional[float] = None
    generation_repetition_penalty: float = 1.0
    
    # System
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    
    # Monitoring
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: str = "updraft-lm"
    wandb_run_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.dropout >= 0.0 and self.dropout <= 1.0, "dropout must be in [0, 1]"
        assert self.learning_rate > 0, "learning_rate must be positive"
        
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'GPT1Config':
        """Create config from dictionary"""
        return cls(**config_dict)
