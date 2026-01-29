import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, labels


def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_labels = []
    
    for ids, lbls in zip(input_ids, labels):
        padding_len = max_len - len(ids)
        padded_ids = torch.cat([ids, torch.zeros(padding_len, dtype=torch.long)])
        padded_lbls = torch.cat([lbls, torch.full((padding_len,), -100, dtype=torch.long)])
        
        padded_input_ids.append(padded_ids)
        padded_labels.append(padded_lbls)
    
    return torch.stack(padded_input_ids), torch.stack(padded_labels)


def get_dataloader(dataset_name, tokenizer, max_seq_len, batch_size, split='train', num_workers=4):
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    elif dataset_name == 'openwebtext':
        dataset = load_dataset('openwebtext', split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
    
    text_dataset = TextDataset(dataset, tokenizer, max_seq_len)
    
    dataloader = DataLoader(
        text_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def create_simple_dataset(text_file, tokenizer, max_seq_len, batch_size):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    
    sequences = []
    for i in range(0, len(tokens) - max_seq_len, max_seq_len // 2):
        seq = tokens[i:i + max_seq_len]
        if len(seq) == max_seq_len:
            sequences.append({'text': tokenizer.decode(seq)})
    
    from torch.utils.data import Dataset as TorchDataset
    
    class SimpleDataset(TorchDataset):
        def __init__(self, sequences):
            self.data = sequences
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = SimpleDataset(sequences)
    text_dataset = TextDataset(dataset, tokenizer, max_seq_len)
    
    dataloader = DataLoader(
        text_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader
