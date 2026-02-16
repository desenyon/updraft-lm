import argparse
import torch
from model.gpt1 import GPT1Model
from config import GPT1Config
from data.tokenizer import Tokenizer
from data.dataset import get_dataloader
from trainer import Trainer
from generator import Generator, load_model_for_inference
from utils import set_seed, get_device
import os

def train(args):
    set_seed(args.seed)
    
    config = GPT1Config()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.max_seq_len = args.max_seq_len
    config.device = str(get_device())
    config.seed = args.seed
    
    print("Initializing tokenizer...")
    tokenizer = Tokenizer()
    
    print(f"Loading dataset: {args.dataset}")
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
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is available or use a local text file.")
        return
    
    print("Initializing model...")
    model = GPT1Model(config)
    print(f"Model has {model.get_num_params():,} parameters")
    
    print("Initializing trainer...")
    trainer = Trainer(model, config, train_loader, val_loader)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    print("Starting training...")
    trainer.train()
    
    print(f"Training complete! Checkpoints saved to {config.checkpoint_dir}")


def generate(args):
    print(f"Loading model from {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    generator = load_model_for_inference(args.checkpoint)
    
    print("\nGenerating text...")
    print(f"Prompt: {args.prompt}")
    print("-" * 80)
    
    outputs = generator.generate(
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_return_sequences=args.num_sequences
    )
    
    for i, output in enumerate(outputs):
        print(f"\nGeneration {i+1}:")
        print(output)
        print("-" * 80)


def interactive(args):
    print(f"Loading model from {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    generator = load_model_for_inference(args.checkpoint)
    
    print("\nInteractive mode. Type 'quit' to exit.")
    print("=" * 80)
    
    while True:
        prompt = input("\nEnter prompt: ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt.strip():
            continue
        
        print("\nGenerating...")
        outputs = generator.generate(
            prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=1
        )
        
        print("-" * 80)
        print(outputs[0])
        print("-" * 80)


def demo():
    print("Running Updraft-LM Demo")
    print("=" * 80)
    
    config = GPT1Config()
    config.n_layers = 4
    config.n_heads = 8
    config.d_model = 512
    config.d_ff = 2048
    config.max_seq_len = 256
    
    print("Initializing small model for demo...")
    model = GPT1Model(config)
    print(f"Model has {model.get_num_params():,} parameters")
    
    tokenizer = Tokenizer()
    
    device = get_device()
    print(f"Using device: {device}")
    
    generator = Generator(model, tokenizer, config, device)
    
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a galaxy far, far away"
    ]
    
    print("\nGenerating sample outputs (untrained model):")
    print("=" * 80)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 80)
        
        outputs = generator.generate(
            prompt,
            max_length=50,
            temperature=1.0,
            top_k=50,
            num_return_sequences=1
        )
        
        print(outputs[0][:200] + "...")
        print("-" * 80)
    
    print("\nDemo complete!")


def tui():
    """Launch the TUI (Text User Interface)"""
    try:
        from tui.app import UpdraftTUI
        app = UpdraftTUI()
        app.run()
    except ImportError as e:
        print(f"Error: TUI dependencies not installed. Please run: pip install textual rich")
        print(f"Details: {e}")
        return
    except Exception as e:
        print(f"Error launching TUI: {e}")
        return


def main():
    parser = argparse.ArgumentParser(description='Updraft-LM: GPT-1 Level Language Model')
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
    
    tui_parser = subparsers.add_parser('tui', help='Launch powerful Text User Interface')
    
    demo_parser = subparsers.add_parser('demo', help='Run demo with untrained model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'generate':
        generate(args)
    elif args.mode == 'interactive':
        interactive(args)
    elif args.mode == 'tui':
        tui()
    elif args.mode == 'demo':
        demo()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
