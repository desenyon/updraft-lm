import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import os
import math
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from config import LlamaConfig

console = Console()

class Trainer:
    def __init__(self, model, config: LlamaConfig, train_loader, val_loader=None):
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
                elif pn.endswith('weight') and 'norm' in mn.lower():
                    no_decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
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
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("Loss: {task.fields[loss]:.4f}"),
            TextColumn("LR: {task.fields[lr]:.6f}")
        ) as progress:
            task = progress.add_task(f"[cyan]Epoch {self.epoch+1}", total=len(self.train_loader), loss=0.0, lr=self.optimizer.param_groups[0]['lr'])
            
            for batch_idx, (input_ids, labels) in enumerate(self.train_loader):
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
                    progress.update(task, advance=0, loss=avg_loss, lr=lr)
                
                progress.advance(task)
                
                if self.global_step % self.config.eval_interval == 0 and self.val_loader is not None:
                    val_loss = self.evaluate()
                    console.print(f"\n[cyan]Step {self.global_step}[/] - [yellow]Val Loss:[/] {val_loss:.4f}")
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
        console.print(f"[bold cyan]Starting training for {self.config.num_epochs} epochs[/]")
        console.print(f"[cyan]Model has {self.model.get_num_params():,} parameters[/]")
        console.print(f"[cyan]Training on {self.device}[/]")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            
            console.print(f"\n[bold green]Epoch {epoch+1}/{self.config.num_epochs}[/] - [yellow]Train Loss:[/] {train_loss:.4f}")
            
            if self.val_loader is not None:
                val_loss = self.evaluate()
                console.print(f"[bold green]Epoch {epoch+1}/{self.config.num_epochs}[/] - [yellow]Val Loss:[/] {val_loss:.4f}")
            
            self.save_checkpoint(f'epoch_{epoch+1}.pt')
        
        console.print("[bold green]Training complete![/]")
    
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
        console.print(f"[dim]Checkpoint saved: {filepath}[/]")
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        console.print(f"[bold green]Checkpoint loaded:[/] {filepath}")
        console.print(f"[green]Resumed from epoch {self.epoch}, step {self.global_step}[/]")
