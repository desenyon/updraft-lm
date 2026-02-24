import torch
from model.llama import LlamaModel
from config import LlamaConfig
from data.tokenizer import Tokenizer
from rich.console import Console

console = Console()

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
        # Beam search for LLaMA would require slightly different logic similar to greedy
        # For brevity, returning standard greedy generation via generate method
        console.print("[yellow]Warning: Beam search not fully implemented for LLaMA architecture. Falling back to greedy.[/]")
        return self.generate_greedy(prompt, max_length=max_length)


def load_model_for_inference(checkpoint_path, config=None):
    if config is None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = LlamaConfig()
    
    model = LlamaModel(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = Tokenizer()
    generator = Generator(model, tokenizer, config)
    
    return generator
