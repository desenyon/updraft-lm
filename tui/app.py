"""
Main TUI Application for Updraft-LM
Built with Textual for a powerful, modern terminal interface
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Button, Static, Input, TextArea,
    Tree, Label, ProgressBar, Tabs, Tab, DataTable, Log,
    TabbedContent, TabPane
)
from textual.binding import Binding
from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import torch
from pathlib import Path
from typing import Optional
import asyncio
from datetime import datetime

from config import GPT1Config
from model.gpt1 import GPT1Model
from generator import Generator, load_model_for_inference
from data.tokenizer import Tokenizer


class ModelStats(Static):
    """Widget to display model statistics"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_info = {}
    
    def update_stats(self, model_info: dict):
        self.model_info = model_info
        self.refresh_display()
    
    def refresh_display(self):
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in self.model_info.items():
            table.add_row(key, str(value))
        
        self.update(Panel(table, title="[bold blue]Model Information[/]", border_style="blue"))


class GenerationOutput(Static):
    """Widget to display generated text"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_text = ""
    
    def set_output(self, text: str):
        self.output_text = text
        syntax = Syntax(text, "text", theme="monokai", line_numbers=False, word_wrap=True)
        self.update(Panel(syntax, title="[bold green]Generated Text[/]", border_style="green"))


class TrainingMonitor(Static):
    """Widget to monitor training progress"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_logs = []
    
    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_logs.append(f"[{timestamp}] {message}")
        if len(self.training_logs) > 100:
            self.training_logs.pop(0)
        self.refresh_display()
    
    def refresh_display(self):
        log_text = "\n".join(self.training_logs[-20:])
        self.update(Panel(log_text, title="[bold yellow]Training Log[/]", border_style="yellow"))


class UpdraftTUI(App):
    """Main TUI Application for Updraft-LM"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-container {
        height: 100%;
    }
    
    #left-panel {
        width: 30%;
        border: solid $primary;
        padding: 1;
    }
    
    #right-panel {
        width: 70%;
        padding: 1;
    }
    
    .input-group {
        height: auto;
        margin: 1 0;
    }
    
    .button-row {
        height: auto;
        align: center middle;
        margin: 1 0;
    }
    
    Button {
        margin: 0 1;
    }
    
    #generation-output {
        height: 50%;
        border: solid $accent;
    }
    
    #training-monitor {
        height: 25%;
        border: solid $warning;
        margin-top: 1;
    }
    
    #model-stats {
        height: 25%;
        border: solid $success;
    }
    
    Input {
        margin: 0 0 1 0;
    }
    
    TextArea {
        height: 10;
        border: solid $primary;
    }
    
    DataTable {
        height: 100%;
    }
    """
    
    TITLE = "Updraft-LM - Language Model TUI"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("g", "focus_generate", "Generate"),
        Binding("t", "focus_train", "Train"),
        Binding("m", "focus_model", "Model"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]
    
    def __init__(self):
        super().__init__()
        self.model: Optional[GPT1Model] = None
        self.generator: Optional[Generator] = None
        self.config: Optional[GPT1Config] = None
        self.tokenizer: Optional[Tokenizer] = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        with TabbedContent():
            with TabPane("Generate", id="tab-generate"):
                with Container(id="main-container"):
                    with Horizontal():
                        with Vertical(id="left-panel"):
                            yield Static("[bold cyan]Model Configuration[/]", classes="section-title")
                            yield Label("Checkpoint Path:")
                            yield Input(
                                placeholder="checkpoints/pretrained_gpt2.pt",
                                id="checkpoint-path",
                                classes="input-group"
                            )
                            yield Label("Prompt:")
                            yield TextArea(id="prompt-input", classes="input-group")
                            yield Label("Max Length:")
                            yield Input(value="100", id="max-length", classes="input-group")
                            yield Label("Temperature:")
                            yield Input(value="0.8", id="temperature", classes="input-group")
                            yield Label("Top-K:")
                            yield Input(value="50", id="top-k", classes="input-group")
                            with Horizontal(classes="button-row"):
                                yield Button("Load Model", id="load-model", variant="primary")
                                yield Button("Generate", id="generate-btn", variant="success")
                        
                        with VerticalScroll(id="right-panel"):
                            yield GenerationOutput(id="generation-output")
                            yield ModelStats(id="model-stats")
            
            with TabPane("Train", id="tab-train"):
                with Container():
                    with Horizontal():
                        with Vertical(id="left-panel"):
                            yield Static("[bold yellow]Training Configuration[/]", classes="section-title")
                            yield Label("Dataset:")
                            yield Input(value="wikitext", id="dataset", classes="input-group")
                            yield Label("Batch Size:")
                            yield Input(value="64", id="batch-size", classes="input-group")
                            yield Label("Learning Rate:")
                            yield Input(value="2.5e-4", id="learning-rate", classes="input-group")
                            yield Label("Epochs:")
                            yield Input(value="5", id="epochs", classes="input-group")
                            with Horizontal(classes="button-row"):
                                yield Button("Start Training", id="start-train", variant="primary")
                                yield Button("Stop Training", id="stop-train", variant="error")
                        
                        with VerticalScroll(id="right-panel"):
                            yield TrainingMonitor(id="training-monitor")
                            yield ProgressBar(id="training-progress", total=100)
            
            with TabPane("Model Info", id="tab-model"):
                with VerticalScroll():
                    yield Static(self._get_architecture_info(), id="architecture-info")
        
        yield Footer()
    
    def _get_architecture_info(self) -> Panel:
        """Get model architecture information as a rich panel"""
        info_text = """
        [bold cyan]Updraft-LM Architecture[/]
        
        A GPT-1 level transformer language model built from scratch.
        
        [bold yellow]Default Configuration:[/]
        • Layers: 12 transformer blocks
        • Attention Heads: 12 heads (64 dimensions each)
        • Embedding Dimension: 768
        • Feed-Forward Dimension: 3072
        • Max Sequence Length: 512
        • Vocabulary Size: 50,257
        • Total Parameters: ~117M
        
        [bold green]Features:[/]
        • Multi-head self-attention
        • Positional encoding
        • Layer normalization
        • GELU activation
        • Dropout regularization
        • Weight initialization optimized for deep networks
        
        [bold blue]Capabilities:[/]
        • Text generation with various sampling strategies
        • Training from scratch or fine-tuning
        • Compatible with GPT-2 tokenizer
        • Mixed precision training support
        • Checkpoint save/load
        """
        return Panel(info_text, title="Model Architecture", border_style="cyan")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        button_id = event.button.id
        
        if button_id == "load-model":
            await self.load_model()
        elif button_id == "generate-btn":
            await self.generate_text()
        elif button_id == "start-train":
            await self.start_training()
        elif button_id == "stop-train":
            self.notify("Training stopped", severity="warning")
    
    async def load_model(self) -> None:
        """Load the model from checkpoint"""
        checkpoint_input = self.query_one("#checkpoint-path", Input)
        checkpoint_path = checkpoint_input.value
        
        if not checkpoint_path:
            self.notify("Please enter a checkpoint path", severity="error")
            return
        
        self.notify(f"Loading model from {checkpoint_path}...", severity="information")
        
        try:
            # Check if checkpoint exists
            if not Path(checkpoint_path).exists():
                # Create a demo model instead
                self.notify("Checkpoint not found. Creating demo model...", severity="warning")
                self.config = GPT1Config()
                self.config.n_layers = 4
                self.config.n_heads = 8
                self.config.d_model = 512
                self.config.d_ff = 2048
                self.config.max_seq_len = 256
                
                self.model = GPT1Model(self.config)
                try:
                    self.tokenizer = Tokenizer()
                except Exception:
                    self.notify("Warning: Could not load tokenizer", severity="warning")
                    self.tokenizer = None
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.generator = Generator(self.model, self.tokenizer, self.config, device) if self.tokenizer else None
            else:
                self.generator = load_model_for_inference(checkpoint_path)
                self.model = self.generator.model
                self.config = self.generator.config
            
            # Update model stats
            model_stats = self.query_one("#model-stats", ModelStats)
            model_stats.update_stats({
                "Parameters": f"{self.model.get_num_params():,}",
                "Layers": self.config.n_layers,
                "Heads": self.config.n_heads,
                "d_model": self.config.d_model,
                "d_ff": self.config.d_ff,
                "Max Seq Len": self.config.max_seq_len,
                "Device": str(next(self.model.parameters()).device),
            })
            
            self.notify("Model loaded successfully!", severity="success")
        except Exception as e:
            self.notify(f"Error loading model: {str(e)}", severity="error")
    
    async def generate_text(self) -> None:
        """Generate text based on the prompt"""
        if not self.generator:
            self.notify("Please load a model first", severity="error")
            return
        
        prompt_input = self.query_one("#prompt-input", TextArea)
        max_length_input = self.query_one("#max-length", Input)
        temperature_input = self.query_one("#temperature", Input)
        top_k_input = self.query_one("#top-k", Input)
        
        prompt = prompt_input.text
        if not prompt.strip():
            self.notify("Please enter a prompt", severity="error")
            return
        
        try:
            max_length = int(max_length_input.value)
            temperature = float(temperature_input.value)
            top_k = int(top_k_input.value) if top_k_input.value else None
        except ValueError:
            self.notify("Invalid parameter values", severity="error")
            return
        
        self.notify("Generating text...", severity="information")
        
        try:
            outputs = self.generator.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                num_return_sequences=1
            )
            
            output_widget = self.query_one("#generation-output", GenerationOutput)
            output_widget.set_output(outputs[0])
            
            self.notify("Generation complete!", severity="success")
        except Exception as e:
            self.notify(f"Generation error: {str(e)}", severity="error")
    
    async def start_training(self) -> None:
        """Start model training"""
        monitor = self.query_one("#training-monitor", TrainingMonitor)
        monitor.add_log("Training feature coming soon!")
        monitor.add_log("This will include real-time training monitoring")
        monitor.add_log("Loss plots, metrics, and progress tracking")
        self.notify("Training interface under development", severity="information")
    
    def action_focus_generate(self) -> None:
        """Switch to generate tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-generate"
    
    def action_focus_train(self) -> None:
        """Switch to train tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-train"
    
    def action_focus_model(self) -> None:
        """Switch to model info tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "tab-model"


def main():
    """Run the TUI application"""
    app = UpdraftTUI()
    app.run()


if __name__ == "__main__":
    main()
