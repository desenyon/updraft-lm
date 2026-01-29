from dataclasses import dataclass

@dataclass
class GPT1Config:
    vocab_size: int = 50257
    max_seq_len: int = 512
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    activation: str = "gelu"
    
    learning_rate: float = 2.5e-4
    batch_size: int = 64
    num_epochs: int = 5
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    
    device: str = "cuda"
    seed: int = 42
