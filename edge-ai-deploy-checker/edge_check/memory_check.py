"""
Memory check module.

Compares model memory requirements against device capabilities,
accounting for OS overhead, runtime memory, and activation buffers.
"""

from dataclasses import dataclass
from .model_parser import ModelInfo
from .device_profiles import DeviceProfile


@dataclass
class MemoryCheckResult:
    """Result of a memory compatibility check."""

    passed: bool
    model_weights_mb: float
    activation_memory_mb: float
    runtime_overhead_mb: float
    total_required_mb: float
    available_mb: float
    utilisation_percent: float
    recommendation: str

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"  Model weights:          {self.model_weights_mb:.1f} MB",
            f"  Activation memory:      ~{self.activation_memory_mb:.1f} MB",
            f"  Runtime overhead:       ~{self.runtime_overhead_mb:.1f} MB",
            f"  Total required:         ~{self.total_required_mb:.1f} MB",
            f"  Available (est.):       {self.available_mb:,.0f} MB",
            f"  Utilisation:            {self.utilisation_percent:.1f}%",
            f"  Status:                 {'✓' if self.passed else '✗'} {status}",
        ]
        if not self.passed:
            lines.append(f"  Recommendation:         {self.recommendation}")
        return "\n".join(lines)


def check_memory(model: ModelInfo, device: DeviceProfile) -> MemoryCheckResult:
    """Check if model fits in device memory.

    Accounts for:
    - Model weight storage
    - Peak activation memory during inference
    - Runtime framework overhead
    - OS and system memory reservation
    """
    model_weights_mb = model.file_size_mb

    # Activation memory estimation
    # For CNNs, peak activation memory is roughly 2-4x weight size
    # For transformer models, it can be higher due to attention matrices
    activation_multiplier = 3.0  # conservative default
    if model.input_shape and len(model.input_shape) >= 3:
        # Larger input resolutions need more activation memory
        spatial = model.input_shape[1] if len(model.input_shape) == 4 else model.input_shape[0]
        if spatial > 416:
            activation_multiplier = 4.0
        elif spatial > 224:
            activation_multiplier = 3.5

    activation_memory_mb = model_weights_mb * activation_multiplier

    # Runtime overhead depends on the framework
    runtime_overhead = {
        "tflite": 50,
        "onnx": 80,
    }
    runtime_mb = runtime_overhead.get(model.format, 60)

    total_required = model_weights_mb + activation_memory_mb + runtime_mb
    available = device.available_ram_mb

    utilisation = (total_required / available) * 100 if available > 0 else 100
    passed = total_required < available * 0.85  # 85% threshold to leave headroom

    # Generate recommendation if failed
    recommendation = ""
    if not passed:
        if model.precision != "INT8":
            savings = model.size_if_quantised_int8
            recommendation = (
                f"Quantise to INT8 to reduce model size from "
                f"{model_weights_mb:.1f} MB to ~{savings:.1f} MB"
            )
        else:
            recommendation = (
                f"Model requires {total_required:.0f} MB but only "
                f"{available:.0f} MB available. Consider a smaller "
                f"architecture or reducing input resolution."
            )

    return MemoryCheckResult(
        passed=passed,
        model_weights_mb=model_weights_mb,
        activation_memory_mb=activation_memory_mb,
        runtime_overhead_mb=runtime_mb,
        total_required_mb=total_required,
        available_mb=available,
        utilisation_percent=utilisation,
        recommendation=recommendation,
    )
