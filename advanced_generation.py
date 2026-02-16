"""
Advanced text generation strategies
Includes beam search, nucleus sampling, and more
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_length: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0


class BeamSearchGenerator:
    """Beam search text generation"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: int = 4,
        max_length: int = 100,
        length_penalty: float = 1.0,
        early_stopping: bool = True
    ) -> torch.Tensor:
        """
        Generate text using beam search
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            num_beams: Number of beams
            max_length: Maximum generation length
            length_penalty: Length penalty factor
            early_stopping: Whether to stop when all beams finish
        
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        
        # Expand input for beam search
        input_ids = input_ids.unsqueeze(1).expand(-1, num_beams, -1)
        input_ids = input_ids.reshape(batch_size * num_beams, -1)
        
        # Initialize beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        beam_scores = beam_scores.view(-1)
        
        # Track finished beams
        done = [False] * batch_size
        
        with torch.no_grad():
            for step in range(max_length):
                # Get logits from model
                logits, _ = self.model(input_ids)
                next_token_logits = logits[:, -1, :]
                
                # Apply length penalty
                if length_penalty != 1.0:
                    next_token_logits = next_token_logits / (step + 1) ** length_penalty
                
                # Calculate scores
                vocab_size = next_token_logits.shape[-1]
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)
                next_token_scores = next_token_scores + beam_scores[:, None]
                
                # Reshape for beam selection
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
                
                # Select top beams
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
                )
                
                # Process each batch
                next_batch_beam = []
                for batch_idx in range(batch_size):
                    if done[batch_idx]:
                        continue
                    
                    # Get beam tokens and indices
                    beam_idx = 0
                    for beam_token_rank, (token_score, token_id) in enumerate(
                        zip(next_token_scores[batch_idx], next_tokens[batch_idx])
                    ):
                        beam_id = token_id // vocab_size
                        token_id = token_id % vocab_size
                        
                        if beam_idx >= num_beams:
                            break
                        
                        next_batch_beam.append((token_score, token_id, batch_idx * num_beams + beam_id))
                        beam_idx += 1
                
                # Update beam scores and sequences
                beam_scores = torch.tensor([x[0] for x in next_batch_beam], device=self.device)
                beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=self.device)
                beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=self.device)
                
                # Update input_ids
                input_ids = input_ids[beam_idx]
                input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(-1)], dim=-1)
                
                # Check for EOS tokens
                if early_stopping:
                    # Simplified early stopping
                    if input_ids.shape[1] >= max_length:
                        break
        
        return input_ids[:num_beams]  # Return top beams


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float = 1.0
) -> torch.Tensor:
    """
    Apply repetition penalty to logits
    
    Args:
        logits: Logits from model [batch_size, vocab_size]
        input_ids: Previously generated tokens [batch_size, seq_len]
        penalty: Repetition penalty factor (> 1.0 penalizes repetition)
    
    Returns:
        Modified logits
    """
    if penalty == 1.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    
    for i in range(batch_size):
        for token_id in set(input_ids[i].tolist()):
            if logits[i, token_id] < 0:
                logits[i, token_id] *= penalty
            else:
                logits[i, token_id] /= penalty
    
    return logits


def apply_top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Filter logits to keep only top-k tokens
    
    Args:
        logits: Logits tensor [batch_size, vocab_size]
        top_k: Number of top tokens to keep
    
    Returns:
        Filtered logits
    """
    if top_k <= 0:
        return logits
    
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits


def apply_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus sampling: filter logits to keep tokens with cumulative probability > top_p
    
    Args:
        logits: Logits tensor [batch_size, vocab_size]
        top_p: Cumulative probability threshold
    
    Returns:
        Filtered logits
    """
    if top_p >= 1.0:
        return logits
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    return logits


def sample_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from logits with temperature scaling
    
    Args:
        logits: Logits tensor [batch_size, vocab_size]
        temperature: Temperature for scaling (lower = more deterministic)
    
    Returns:
        Sampled token IDs
    """
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.squeeze(-1)


def generate_with_constraints(
    model,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    forbidden_tokens: Optional[List[int]] = None,
    required_tokens: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Generate text with constraints
    
    Args:
        model: Language model
        input_ids: Input token IDs
        config: Generation configuration
        forbidden_tokens: List of token IDs that should not be generated
        required_tokens: List of token IDs that must appear in generation
    
    Returns:
        Generated token IDs
    """
    model.eval()
    device = input_ids.device
    
    generated = input_ids
    required_set = set(required_tokens) if required_tokens else set()
    forbidden_set = set(forbidden_tokens) if forbidden_tokens else set()
    
    with torch.no_grad():
        for _ in range(config.max_length):
            # Get logits
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = apply_repetition_penalty(
                    next_token_logits, generated, config.repetition_penalty
                )
            
            # Forbid certain tokens
            if forbidden_set:
                for token_id in forbidden_set:
                    next_token_logits[:, token_id] = float('-inf')
            
            # Apply top-k filtering
            if config.top_k:
                next_token_logits = apply_top_k_filtering(next_token_logits, config.top_k)
            
            # Apply top-p filtering
            if config.top_p:
                next_token_logits = apply_top_p_filtering(next_token_logits, config.top_p)
            
            # Sample next token
            next_token = sample_with_temperature(next_token_logits, 1.0)
            
            # Update generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            
            # Check if required tokens are met
            if required_set:
                generated_set = set(generated[0].tolist())
                required_set -= generated_set
    
    return generated
