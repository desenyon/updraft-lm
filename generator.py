import torch
from model.gpt1 import GPT1Model
from config import GPT1Config
from data.tokenizer import Tokenizer

class Generator:
    def __init__(self, model, tokenizer, config, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=None, top_p=None, num_return_sequences=1):
        self.model.eval()
        
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        input_ids = input_ids.repeat(num_return_sequences, 1)
        
        generated = self.model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        outputs = []
        for i in range(num_return_sequences):
            tokens = generated[i].tolist()
            text = self.tokenizer.decode(tokens)
            outputs.append(text)
        
        return outputs
    
    @torch.no_grad()
    def generate_greedy(self, prompt, max_length=100):
        return self.generate(prompt, max_length=max_length, temperature=1.0, top_k=1)
    
    @torch.no_grad()
    def generate_beam_search(self, prompt, max_length=100, num_beams=5):
        self.model.eval()
        
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        beam_scores = torch.zeros(num_beams).to(self.device)
        beam_sequences = input_ids.repeat(num_beams, 1)
        
        for _ in range(max_length):
            if beam_sequences.size(1) > self.config.max_seq_len:
                beam_sequences_cond = beam_sequences[:, -self.config.max_seq_len:]
            else:
                beam_sequences_cond = beam_sequences
            
            logits, _ = self.model(beam_sequences_cond)
            next_token_logits = logits[:, -1, :]
            
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            
            vocab_size = log_probs.size(-1)
            next_scores = beam_scores.unsqueeze(1) + log_probs
            next_scores = next_scores.view(-1)
            
            top_scores, top_indices = torch.topk(next_scores, num_beams)
            
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            beam_sequences = beam_sequences[beam_indices]
            beam_sequences = torch.cat([beam_sequences, token_indices.unsqueeze(1)], dim=1)
            beam_scores = top_scores
        
        best_sequence = beam_sequences[0]
        tokens = best_sequence.tolist()
        text = self.tokenizer.decode(tokens)
        
        return [text]


def load_model_for_inference(checkpoint_path, config=None):
    if config is None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = GPT1Config()
    
    model = GPT1Model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = Tokenizer()
    
    generator = Generator(model, tokenizer, config)
    
    return generator
