# Updraft-LM: GPT-1 Level Language Model

A complete implementation of a GPT-1 level transformer language model from scratch with full mathematical details.

## Architecture

- 12-layer transformer decoder
- 768-dimensional embeddings
- 12 attention heads
- 3072-dimensional feed-forward network
- Maximum sequence length: 512 tokens

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python main.py --mode train --dataset wikitext
```

## Generation

```bash
python main.py --mode generate --prompt "Once upon a time" --max_length 100
```

## Features

- Full transformer implementation with multi-head self-attention
- Positional encoding
- Layer normalization
- Training with gradient clipping and warmup
- Various sampling strategies (greedy, top-k, top-p)
- Checkpoint saving and loading
