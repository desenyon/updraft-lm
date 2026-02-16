# Updraft-LM TUI Screenshots and Demo

## TUI Interface Overview

The Updraft-LM TUI provides a powerful, interactive terminal interface for working with language models.

### Main Interface Structure

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Updraft-LM - Language Model TUI                                           │
├────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────┬──────────┬─────────────┐                                       │
│ │Generate │  Train   │ Model Info  │  ← Tab Navigation                    │
│ └─────────┴──────────┴─────────────┘                                       │
│                                                                             │
│ ┌─────────────────────────┬───────────────────────────────────────────┐   │
│ │ Left Panel              │ Right Panel                               │   │
│ │                         │                                           │   │
│ │ Model Configuration     │ ╔═══════════════════════════════════════╗ │   │
│ │ ─────────────────────   │ ║     Generated Text Output             ║ │   │
│ │                         │ ║                                       ║ │   │
│ │ Checkpoint Path:        │ ║  Once upon a time, in a land far      ║ │   │
│ │ [________________]      │ ║  away, there lived a wise old         ║ │   │
│ │                         │ ║  wizard who possessed great           ║ │   │
│ │ Prompt:                 │ ║  magical powers...                    ║ │   │
│ │ ┌───────────────────┐   │ ║                                       ║ │   │
│ │ │Once upon a time   │   │ ╚═══════════════════════════════════════╝ │   │
│ │ │                   │   │                                           │   │
│ │ └───────────────────┘   │ ╔═══════════════════════════════════════╗ │   │
│ │                         │ ║     Model Information                 ║ │   │
│ │ Max Length: [100___]    │ ║  Parameters: 117,000,000              ║ │   │
│ │ Temperature: [0.8__]    │ ║  Layers: 12                          ║ │   │
│ │ Top-K: [50_____]        │ ║  Heads: 12                           ║ │   │
│ │                         │ ║  d_model: 768                        ║ │   │
│ │ [Load Model] [Generate] │ ║  Max Seq Len: 512                    ║ │   │
│ │                         │ ╚═══════════════════════════════════════╝ │   │
│ └─────────────────────────┴───────────────────────────────────────────┘   │
├────────────────────────────────────────────────────────────────────────────┤
│ g: Generate | t: Train | m: Model Info | q: Quit                          │
└────────────────────────────────────────────────────────────────────────────┘
```

### Generate Tab Features

- **Interactive Prompt Input**: Multi-line text area for entering generation prompts
- **Parameter Controls**: Adjust temperature, top-k, max length in real-time
- **Live Generation**: See generated text as it appears
- **Model Loading**: Load checkpoints from disk
- **Generation History**: Review previous generations

### Train Tab Features

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Train Tab                                                                  │
├────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┬───────────────────────────────────────────┐   │
│ │ Training Configuration  │ Training Monitor                          │   │
│ │                         │                                           │   │
│ │ Dataset: [wikitext___]  │ [19:20:45] Starting training epoch 1/5   │   │
│ │ Batch Size: [64______]  │ [19:20:46] Batch 1/1000 - Loss: 4.234   │   │
│ │ Learning Rate: [2.5e-4] │ [19:20:47] Batch 2/1000 - Loss: 4.189   │   │
│ │ Epochs: [5__________]   │ [19:20:48] Batch 3/1000 - Loss: 4.156   │   │
│ │                         │ ...                                       │   │
│ │ [Start Training]        │                                           │   │
│ │ [Stop Training]         │ ╔════════════════════════════════════╗    │   │
│ │                         │ ║ Progress: ████████░░░░░░░░░ 50%   ║    │   │
│ │                         │ ╚════════════════════════════════════╝    │   │
│ └─────────────────────────┴───────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Model Info Tab

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Model Info Tab                                                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Updraft-LM Architecture                                                   │
│  ════════════════════════                                                  │
│                                                                             │
│  A GPT-1 level transformer language model built from scratch.              │
│                                                                             │
│  Default Configuration:                                                    │
│  • Layers: 12 transformer blocks                                           │
│  • Attention Heads: 12 heads (64 dimensions each)                          │
│  • Embedding Dimension: 768                                                │
│  • Feed-Forward Dimension: 3072                                            │
│  • Max Sequence Length: 512                                                │
│  • Vocabulary Size: 50,257                                                 │
│  • Total Parameters: ~117M                                                 │
│                                                                             │
│  Features:                                                                 │
│  • Multi-head self-attention                                               │
│  • Positional encoding                                                     │
│  • Layer normalization                                                     │
│  • GELU activation                                                         │
│  • Dropout regularization                                                  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `g` | Switch to Generate tab |
| `t` | Switch to Train tab |
| `m` | Switch to Model Info tab |
| `q` | Quit application |
| `Tab` | Navigate between elements |
| `Enter` | Activate buttons/submit |
| `Ctrl+C` | Force quit |

## Usage Examples

### Launch TUI
```bash
python main.py tui
```

### Quick Text Generation
1. Launch TUI
2. Press `g` to go to Generate tab (if not already there)
3. Enter checkpoint path or leave empty for demo model
4. Click "Load Model"
5. Enter your prompt in the text area
6. Adjust parameters (temperature, top-k, etc.)
7. Click "Generate"
8. View output in the right panel

### Monitor Training (Coming Soon)
1. Launch TUI
2. Press `t` to go to Train tab
3. Configure training parameters
4. Click "Start Training"
5. Watch real-time progress and logs

## Color Scheme

The TUI uses a modern, professional color scheme:
- **Primary**: Cyan for main elements
- **Success**: Green for positive actions
- **Warning**: Yellow for warnings
- **Error**: Red for errors
- **Info**: Blue for information

## Responsive Design

The TUI automatically adapts to your terminal size:
- Minimum recommended: 80x24 characters
- Optimal: 120x40 characters
- Maximum: Full screen

## Features in Detail

### Real-time Generation
- See text appear as it's generated
- Stop generation at any time
- Save outputs to file

### Model Statistics
- Live parameter counts
- Memory usage estimation
- Device information (CPU/CUDA)

### Configuration Presets
- Save favorite configurations
- Load preset configurations
- Quick switching between models

## Technical Details

### Built With
- **Textual**: Modern TUI framework
- **Rich**: Beautiful terminal formatting
- **PyTorch**: Deep learning backend

### Performance
- Asynchronous operations
- Non-blocking UI updates
- Efficient rendering

### Accessibility
- Keyboard navigation
- Screen reader friendly
- High contrast mode support
