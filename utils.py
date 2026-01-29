import torch
import numpy as np
from collections import Counter

def calculate_perplexity(model, dataloader, device):
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


def calculate_metrics(predictions, references):
    from collections import defaultdict
    
    def tokenize(text):
        return text.lower().split()
    
    def calculate_bleu_n(pred_tokens, ref_tokens, n):
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    print(f"Model loaded from {path}")
    return model
