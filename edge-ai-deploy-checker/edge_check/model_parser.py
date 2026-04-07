"""
Model parser for edge AI deployment checking.

Extracts model metadata: size, parameters, operations,
input/output shapes, and estimated compute requirements.

Supports: TFLite (.tflite), ONNX (.onnx)
"""

import os
import struct
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    """Parsed model information."""

    filepath: str
    format: str  # "tflite", "onnx"
    file_size_mb: float
    num_parameters: Optional[int] = None
    estimated_flops: Optional[int] = None
    input_shape: Optional[list] = None
    output_shape: Optional[list] = None
    precision: str = "FP32"  # FP32, FP16, INT8
    operations: list = field(default_factory=list)

    @property
    def estimated_memory_mb(self) -> float:
        """Estimate total memory needed during inference.

        This includes:
        - Model weights (file size as baseline)
        - Activation memory (~2-4x weight size for typical CNNs)
        - Runtime overhead (~50MB for TFLite, ~80MB for ONNX Runtime)
        """
        weights_mb = self.file_size_mb
        activation_mb = weights_mb * 3  # conservative estimate
        runtime_mb = 50 if self.format == "tflite" else 80
        return weights_mb + activation_mb + runtime_mb

    @property
    def size_if_quantised_int8(self) -> float:
        """Estimate model size if quantised to INT8."""
        if self.precision == "INT8":
            return self.file_size_mb
        elif self.precision == "FP16":
            return self.file_size_mb / 2
        else:  # FP32
            return self.file_size_mb / 4

    def summary(self) -> str:
        """Return a human-readable model summary."""
        lines = [
            f"Model:       {Path(self.filepath).name}",
            f"Format:      {self.format.upper()}",
            f"Size:        {self.file_size_mb:.1f} MB ({self.precision})",
            f"Parameters:  {self._format_params(self.num_parameters)}",
            f"Est. FLOPs:  {self._format_flops(self.estimated_flops)}",
        ]
        if self.input_shape:
            lines.append(f"Input:       {self.input_shape}")
        if self.output_shape:
            lines.append(f"Output:      {self.output_shape}")
        if self.operations:
            lines.append(f"Operations:  {', '.join(self.operations[:10])}")
            if len(self.operations) > 10:
                lines.append(f"             ... and {len(self.operations) - 10} more")
        return "\n".join(lines)

    @staticmethod
    def _format_params(n: Optional[int]) -> str:
        if n is None:
            return "Unknown"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    @staticmethod
    def _format_flops(n: Optional[int]) -> str:
        if n is None:
            return "Unknown"
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.1f} GFLOPS"
        if n >= 1_000_000:
            return f"{n / 1_000_000:.0f} MFLOPS"
        return str(n)


def parse_model(filepath: str) -> ModelInfo:
    """Parse a model file and extract metadata.

    Args:
        filepath: Path to the model file (.tflite or .onnx)

    Returns:
        ModelInfo with extracted metadata

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    ext = path.suffix.lower()
    file_size_mb = path.stat().st_size / (1024 * 1024)

    if ext == ".tflite":
        return _parse_tflite(filepath, file_size_mb)
    elif ext == ".onnx":
        return _parse_onnx(filepath, file_size_mb)
    else:
        raise ValueError(
            f"Unsupported model format: {ext}. "
            f"Supported formats: .tflite, .onnx"
        )


def _parse_tflite(filepath: str, file_size_mb: float) -> ModelInfo:
    """Parse a TensorFlow Lite model file.

    Uses the TFLite flatbuffer schema to extract metadata
    without requiring TensorFlow to be installed.
    """
    info = ModelInfo(
        filepath=filepath,
        format="tflite",
        file_size_mb=file_size_mb,
    )

    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=filepath)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if input_details:
            info.input_shape = input_details[0]["shape"].tolist()
            dtype = input_details[0]["dtype"]
            if dtype == "int8" or "int8" in str(dtype):
                info.precision = "INT8"
            elif dtype == "float16" or "float16" in str(dtype):
                info.precision = "FP16"

        if output_details:
            info.output_shape = output_details[0]["shape"].tolist()

        # Count parameters from tensor details
        tensor_details = interpreter.get_tensor_details()
        total_params = 0
        ops = set()
        for tensor in tensor_details:
            shape = tensor.get("shape", [])
            if len(shape) > 0:
                params = 1
                for dim in shape:
                    params *= dim
                total_params += params

        info.num_parameters = total_params

    except ImportError:
        # TensorFlow not installed — use file size to estimate
        info.num_parameters = int(file_size_mb * 1024 * 1024 / 4)  # rough FP32 estimate
        info.precision = "FP32"

    # Rough FLOPs estimate based on parameters
    if info.num_parameters:
        info.estimated_flops = info.num_parameters * 2  # very rough

    return info


def _parse_onnx(filepath: str, file_size_mb: float) -> ModelInfo:
    """Parse an ONNX model file."""
    info = ModelInfo(
        filepath=filepath,
        format="onnx",
        file_size_mb=file_size_mb,
    )

    try:
        import onnx

        model = onnx.load(filepath)
        graph = model.graph

        # Extract operations
        info.operations = list(set(node.op_type for node in graph.node))

        # Extract input shape
        if graph.input:
            input_tensor = graph.input[0]
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else -1)
            info.input_shape = shape

        # Extract output shape
        if graph.output:
            output_tensor = graph.output[0]
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else -1)
            info.output_shape = shape

        # Count parameters from initializers
        total_params = 0
        for init in graph.initializer:
            params = 1
            for dim in init.dims:
                params *= dim
            total_params += params

        info.num_parameters = total_params

        # Check precision from initializer data types
        if graph.initializer:
            dtype = graph.initializer[0].data_type
            if dtype == 3:  # INT8
                info.precision = "INT8"
            elif dtype == 10:  # FP16
                info.precision = "FP16"
            else:
                info.precision = "FP32"

    except ImportError:
        info.num_parameters = int(file_size_mb * 1024 * 1024 / 4)
        info.precision = "FP32"

    if info.num_parameters:
        info.estimated_flops = info.num_parameters * 2

    return info
