import torch
from model.gpt1 import GPT1Model
from config import GPT1Config
from data.tokenizer import Tokenizer
from data.dataset import create_simple_dataset
from trainer import Trainer
import os

def create_sample_text():
    sample_text = """
The transformer architecture revolutionized natural language processing. It introduced the concept of self-attention, allowing models to weigh the importance of different words in a sequence. This mechanism enabled parallel processing of sequences, making training much faster than recurrent neural networks.

The attention mechanism computes three vectors for each input: query, key, and value. The model learns to focus on relevant parts of the input by computing attention scores between queries and keys. These scores are then used to create weighted combinations of values.

Multi-head attention extends this idea by using multiple attention heads in parallel. Each head can learn different aspects of the relationships between words. The outputs from all heads are concatenated and linearly transformed to produce the final result.

Position encoding is crucial in transformers because the attention mechanism itself has no notion of sequence order. By adding positional information to the input embeddings, the model can distinguish between words at different positions.

The feed-forward networks in each transformer layer consist of two linear transformations with an activation function in between. These networks process each position independently and identically, adding expressive power to the model.

Layer normalization and residual connections help stabilize training in deep networks. Residual connections allow gradients to flow directly through the network, preventing the vanishing gradient problem.

The GPT architecture uses a decoder-only transformer structure. It is trained with a language modeling objective, predicting the next token given all previous tokens. This autoregressive approach enables the model to generate coherent text.

Training large language models requires massive amounts of text data and computational resources. The models learn statistical patterns in language, including grammar, facts, and reasoning capabilities. As models scale up in size, they exhibit emergent abilities not seen in smaller models.

Fine-tuning allows pre-trained models to adapt to specific tasks. By continuing training on task-specific data, the model can specialize while retaining its general language understanding.

The future of language modeling involves scaling to even larger models, improving efficiency, and developing better training techniques. Research continues on understanding how these models work and making them more reliable and controllable.
"""
    return sample_text

def train_small_model():
    print("Creating sample training data...")
    sample_text = create_sample_text()
    
    os.makedirs('data/samples', exist_ok=True)
    with open('data/samples/train.txt', 'w') as f:
        f.write(sample_text * 10)
    
    config = GPT1Config()
    config.n_layers = 6
    config.n_heads = 8
    config.d_model = 512
    config.d_ff = 2048
    config.max_seq_len = 256
    config.batch_size = 4
    config.num_epochs = 10
    config.learning_rate = 5e-4
    config.warmup_steps = 100
    config.eval_interval = 50
    config.save_interval = 200
    config.log_interval = 10
    
    print("Initializing tokenizer...")
    tokenizer = Tokenizer()
    
    print("Creating dataset...")
    train_loader = create_simple_dataset(
        'data/samples/train.txt',
        tokenizer,
        config.max_seq_len,
        config.batch_size
    )
    
    print("Initializing model...")
    model = GPT1Model(config)
    print(f"Model has {model.get_num_params():,} parameters")
    
    print("Starting training...")
    trainer = Trainer(model, config, train_loader)
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Model saved to {config.checkpoint_dir}")
    
    print("\nTesting generation...")
    from generator import Generator
    generator = Generator(model, tokenizer, config)
    
    prompts = [
        "The transformer architecture",
        "Attention mechanism",
        "Language models"
    ]
    
    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}")
        outputs = generator.generate(prompt, max_length=50, temperature=0.8, top_k=50)
        print(outputs[0])

if __name__ == '__main__':
    train_small_model()
