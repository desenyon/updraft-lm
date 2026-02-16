"""
Model Export Utilities
Support for exporting models to ONNX, TorchScript, and other formats
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger("updraft-lm.export")


def export_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    sample_input: torch.Tensor,
    opset_version: int = 14,
    dynamic_axes: Optional[dict] = None,
    verbose: bool = False
) -> None:
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        sample_input: Example input tensor for tracing
        opset_version: ONNX opset version
        dynamic_axes: Dictionary of dynamic axes for inputs/outputs
        verbose: Whether to print export information
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    if dynamic_axes is None:
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'}
        }
    
    try:
        torch.onnx.export(
            model,
            sample_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
            verbose=verbose
        )
        logger.info(f"Model exported to ONNX: {output_path}")
        
        # Verify the exported model
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verified successfully")
        
    except Exception as e:
        logger.error(f"Failed to export to ONNX: {e}")
        raise


def export_to_torchscript(
    model: nn.Module,
    output_path: Union[str, Path],
    sample_input: torch.Tensor,
    method: str = "trace"
) -> None:
    """
    Export PyTorch model to TorchScript
    
    Args:
        model: PyTorch model to export
        output_path: Path to save TorchScript model
        sample_input: Example input for tracing
        method: Export method ('trace' or 'script')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    try:
        if method == "trace":
            traced_model = torch.jit.trace(model, sample_input)
        elif method == "script":
            traced_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'")
        
        traced_model.save(str(output_path))
        logger.info(f"Model exported to TorchScript: {output_path}")
        
        # Verify by loading
        loaded_model = torch.jit.load(str(output_path))
        logger.info("TorchScript model verified successfully")
        
    except Exception as e:
        logger.error(f"Failed to export to TorchScript: {e}")
        raise


def quantize_model(
    model: nn.Module,
    quantization_type: str = "dynamic",
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Quantize model for reduced size and faster inference
    
    Args:
        model: PyTorch model to quantize
        quantization_type: Type of quantization ('dynamic' or 'static')
        dtype: Quantization dtype
    
    Returns:
        Quantized model
    """
    model.eval()
    
    try:
        if quantization_type == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=dtype
            )
            logger.info("Dynamic quantization applied")
        elif quantization_type == "static":
            # Static quantization requires calibration data
            logger.warning("Static quantization requires calibration. Using dynamic instead.")
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=dtype
            )
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        return quantized_model
        
    except Exception as e:
        logger.error(f"Failed to quantize model: {e}")
        raise


def optimize_for_inference(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device
) -> nn.Module:
    """
    Optimize model for inference using torch.compile and other techniques
    
    Args:
        model: Model to optimize
        sample_input: Example input for optimization
        device: Device to optimize for
    
    Returns:
        Optimized model
    """
    model.eval()
    model = model.to(device)
    
    # Use torch.compile if available (PyTorch 2.0+)
    try:
        import torch._dynamo
        optimized_model = torch.compile(model, mode="reduce-overhead")
        logger.info("Model optimized with torch.compile")
        return optimized_model
    except (ImportError, AttributeError):
        logger.warning("torch.compile not available, skipping optimization")
        return model


def get_model_info(model: nn.Module) -> dict:
    """Get comprehensive model information"""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    
    return {
        'total_parameters': param_count,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': param_count - trainable_params,
        'size_mb': (param_size + buffer_size) / 1024**2,
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
    }
