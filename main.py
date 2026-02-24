import argparse
import os
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.traceback import install

install(show_locals=True)
console = Console()

from model.llama import LlamaModel
from config import LlamaConfig
from data.tokenizer import Tokenizer
from data.dataset import get_dataloader
from trainer import Trainer
from generator import Generator, load_model_for_inference
from utils import set_seed, get_device

def train(args):
    set_seed(args.seed)
    console.print(Panel(f"[bold blue]Updraft-LM Training | Version {LlamaConfig.version}[/]", expand=False))
    
    config = LlamaConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.max_seq_len = args.max_seq_len
    config.device = str(get_device())
    config.seed = args.seed
    
    console.print("[green]Initializing tokenizer...[/]")
    tokenizer = Tokenizer()
    
    console.print(f"[green]Loading dataset:[/] {args.dataset}")
    try:
        train_loader = get_dataloader(
            args.dataset,
            tokenizer,
            config.max_seq_len,
            config.batch_size,
            split='train',
            num_workers=args.num_workers
        )
        
        val_loader = None
        if args.validate:
            val_loader = get_dataloader(
                args.dataset,
                tokenizer,
                config.max_seq_len,
                config.batch_size,
                split='validation',
                num_workers=args.num_workers
            )
    except Exception as e:
        console.print(f"[red]Error loading dataset:[/] {e}")
        console.print("Please ensure the dataset is available or use a local text file.")
        return
    
    console.print("[green]Initializing model...[/]")
    model = LlamaModel(config)
    console.print(f"[bold cyan]Model has {model.get_num_params():,} parameters[/]")
    
    console.print("[green]Initializing trainer...[/]")
    trainer = Trainer(model, config, train_loader, val_loader)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    console.print("[bold green]Starting training...[/]")
    trainer.train()
    
    console.print(f"[bold green]Training complete! Checkpoints saved to {config.checkpoint_dir}[/]")

def generate(args):
    console.print(f"[green]Loading model from:[/] {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        console.print(f"[red]Error: Checkpoint not found at {args.checkpoint}[/]")
        return
    
    generator = load_model_for_inference(args.checkpoint)
    
    console.print(Panel(f"[bold blue]Generating text...[/]\n[cyan]Prompt:[/] {args.prompt}", expand=False))
    
    outputs = generator.generate(
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_sequences
    )
    
    for i, output in enumerate(outputs):
        console.print(Panel(output, title=f"Generation {i+1}", border_style="green"))

def interactive(args):
    console.print(f"[green]Loading model from:[/] {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        console.print(f"[red]Error: Checkpoint not found at {args.checkpoint}[/]")
        return
    
    generator = load_model_for_inference(args.checkpoint)
    
    console.print(Panel("[bold blue]Interactive mode[/]\nType 'quit' or 'exit' to stop.", expand=False))
    
    while True:
        prompt = console.input("[bold cyan]Enter prompt:[/] ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt.strip():
            continue
        
        console.print("[dim]Generating...[/]")
        outputs = generator.generate(
            prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=1
        )
        
        console.print(Panel(outputs[0], border_style="green"))

def demo():
    console.print(Panel(f"[bold blue]Running Updraft-LM Demo | Version {LlamaConfig.version}[/]", expand=False))
    
    config = LlamaConfig()
    config.n_layers = 4
    config.n_heads = 8
    config.n_kv_heads = 2
    config.d_model = 512
    config.d_ff = 2048
    config.max_seq_len = 256
    
    console.print("[green]Initializing small model for demo...[/]")
    model = LlamaModel(config)
    console.print(f"[bold cyan]Model has {model.get_num_params():,} parameters[/]")
    
    tokenizer = Tokenizer()
    
    device = get_device()
    console.print(f"[green]Using device:[/] {device}")
    
    generator = Generator(model, tokenizer, config, device)
    
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a galaxy far, far away"
    ]
    
    console.print("[bold green]\nGenerating sample outputs (untrained model):[/]")
    
    for prompt in prompts:
        console.print(f"\n[cyan]Prompt:[/] {prompt}")
        
        outputs = generator.generate(
            prompt,
            max_length=50,
            temperature=1.0,
            top_k=50,
            num_return_sequences=1
        )
        
        console.print(Panel(outputs[0][:200] + "...", border_style="yellow"))
    
    console.print("[bold green]\nDemo complete![/]")

def main():
    parser = argparse.ArgumentParser(description='Updraft-LM: Advanced Language Model Framework')
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset', type=str, default='wikitext', help='Dataset name')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, default=2.5e-4, help='Learning rate')
    train_parser.add_argument('--max-seq-len', type=int, default=512, help='Maximum sequence length')
    train_parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    train_parser.add_argument('--validate', action='store_true', help='Run validation')
    train_parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    gen_parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    gen_parser.add_argument('--max-length', type=int, default=100, help='Maximum generation length')
    gen_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    gen_parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling')
    gen_parser.add_argument('--top-p', type=float, default=None, help='Top-p (nucleus) sampling')
    gen_parser.add_argument('--num-sequences', type=int, default=1, help='Number of sequences to generate')
    
    interactive_parser = subparsers.add_parser('interactive', help='Interactive generation')
    interactive_parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    interactive_parser.add_argument('--max-length', type=int, default=100, help='Maximum generation length')
    interactive_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    interactive_parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    interactive_parser.add_argument('--top-p', type=float, default=None, help='Top-p (nucleus) sampling')
    
    demo_parser = subparsers.add_parser('demo', help='Run demo with untrained model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'generate':
        generate(args)
    elif args.mode == 'interactive':
        interactive(args)
    elif args.mode == 'demo':
        demo()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
