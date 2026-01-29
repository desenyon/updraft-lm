import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import os
import math
from tqdm import tqdm
from config import GPT1Config

class Trainer:
    def __init__(self, model, config: GPT1Config, train_loader, val_loader=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()
        
        self.global_step = 0
        self.epoch = 0
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def configure_optimizer(self):
        decay = set()
        no_decay = set()
        
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (nn.Linear, nn.Embedding)):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, nn.LayerNorm):
                    no_decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        optimizer = AdamW(optim_groups, lr=self.config.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        return optimizer
    
    def configure_scheduler(self):
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            else:
                progress = float(step - self.config.warmup_steps) / float(max(1, len(self.train_loader) * self.config.num_epochs - self.config.warmup_steps))
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(self.optimizer, lr_lambda)
        return scheduler
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        
        for batch_idx, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, loss = self.model(input_ids, labels)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.6f}'})
            
            if self.global_step % self.config.eval_interval == 0 and self.val_loader is not None:
                val_loss = self.evaluate()
                print(f"\nStep {self.global_step} - Val Loss: {val_loss:.4f}")
                self.model.train()
            
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        
        for input_ids, labels in self.val_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            logits, loss = self.model(input_ids, labels)
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Model has {self.model.get_num_params():,} parameters")
        print(f"Training on {self.device}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_loss:.4f}")
            
            if self.val_loader is not None:
                val_loss = self.evaluate()
                print(f"Epoch {epoch+1}/{self.config.num_epochs} - Val Loss: {val_loss:.4f}")
            
            self.save_checkpoint(f'epoch_{epoch+1}.pt')
        
        print("Training complete!")
    
    def save_checkpoint(self, filename=None):
        if filename is None:
            filename = f'checkpoint_step_{self.global_step}.pt'
        
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
