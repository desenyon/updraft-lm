#!/bin/bash

# Quick Start Script for Updraft-LM
# Run this script to see the demo in action.

set -e

echo "========================================"
echo "      Updraft-LM Quick Start Demo       "
echo "========================================"

# Check if virtual environment exists, if so activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if not already installed
if ! python3 -c "import rich" &> /dev/null; then
    echo "Installing required dependencies..."
    pip install -r requirements.txt
fi

echo "Running the Updraft-LM demo mode..."
python main.py demo

echo "Demo complete."
