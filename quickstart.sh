#!/bin/bash

echo "========================================="
echo "Updraft-LM Quick Start"
echo "========================================="

echo ""
echo "Step 1: Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Step 2: Loading pretrained GPT-2 weights..."
python load_pretrained.py --model gpt2 --output checkpoints/pretrained_gpt2.pt

echo ""
echo "Step 3: Running demo generation..."
echo ""

python main.py generate \
    --checkpoint checkpoints/pretrained_gpt2.pt \
    --prompt "Once upon a time in a land far away" \
    --max-length 100 \
    --temperature 0.8 \
    --top-k 50

echo ""
echo "========================================="
echo "Quick start complete!"
echo ""
echo "Try these commands:"
echo "  python main.py interactive --checkpoint checkpoints/pretrained_gpt2.pt"
echo "  python main.py generate --checkpoint checkpoints/pretrained_gpt2.pt --prompt 'Your prompt here'"
echo "  python main.py train --dataset wikitext --epochs 1"
echo "========================================="
