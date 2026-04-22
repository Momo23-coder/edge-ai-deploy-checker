"""
Compute check module.

Estimates inference time for a model on a target edge device,
based on model FLOPs, device compute capability, and processor type.
Compares against a target framerate to determine if real-time
inference is feasible.
"""

from dataclasses import dataclass
from typing import Optional
from .model_parser import ModelInfo
from .device_profiles import DeviceProfile


# Efficiency factors for different compute configurations
# Real-world throughput is significantly lower than theoretical peak
# due to memory bandwidth, cache misses, pipeline stalls, and
# operations that don't map efficiently to hardware
EFFICIENCY_FACTORS = {
    "cpu_only": 0.15,        # CPU alone — memory-bound for most models
    "cpu_neon": 0.25,        # CPU with NEON/SIMD optimisation
    "gpu_delegate": 0.40,    # GPU delegation (e.g., TFLite GPU delegate)
    "npu": 0.55,             # Dedicated NPU/TPU (e.g., Edge TPU, NVDLA)
    "tensorrt": 0.50,        # TensorRT optimised (Jetson devices)
}

# INT8 speedup factor over FP32 on ARM processors
# NEON SIMD units process 4x more INT8 ops per cycle than FP32
QUANTISATION_SPEEDUP = {
    "FP32": 1.0,
    "FP16": 1.8,   # roughly 2x throughput on hardware with FP16 support
    "INT8": 3.0,   # 2-4x on ARM with NEON, conservative estimate
}

# Thermal throttling factor for sustained inference
# After ~30 seconds of continuous load, most edge devices
# reduce clock speed to manage heat
THERMAL_THROTTLE_FACTOR = 0.80  # 20% reduction under sustained load


@dataclass
class ComputeCheckResult:
    """Result of a compute performance check."""

    passed: bool
    estimated_inference_ms: float
    sustained_inference_ms: float  # after thermal throttling
    target_framerate: int
    frame_budget_ms: float
    compute_method: str  # which processor path was used
    device_peak_gops: float
    effective_gops: float
    model_gflops: float
    utilisation_percent: float
    recommendation: str

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"  Model compute:          {self.model_gflops:.1f} GFLOPS",
            f"  Device peak:            {self.device_peak_gops:.1f} GOPS",
            f"  Effective throughput:    {self.effective_gops:.1f} GOPS ({self.compute_method})",
            f"  Est. inference time:    {self.estimated_inference_ms:.0f} ms (burst)",
            f"  Sustained inference:    {self.sustained_inference_ms:.0f} ms (after thermal)",
            f"  Target framerate:       {self.target_framerate} fps ({self.frame_budget_ms:.0f} ms budget)",
            f"  Status:                 {'✓' if self.passed else '✗'} {status}",
        ]
        if not self.passed:
            lines.append(f"  Recommendation:         {self.recommendation}")
        return "\n".join(lines)


def check_compute(
    model: ModelInfo,
    device: DeviceProfile,
    target_fps: int = 30,
) -> ComputeCheckResult:
    """Estimate inference time and check against target framerate.

    The estimation works by:
    1. Taking the model's estimated FLOPs
    2. Determining the best available compute path on the device
       (NPU > GPU > CPU with NEON > CPU only)
    3. Applying an efficiency factor (real-world vs theoretical peak)
    4. Applying quantisation speedup based on model precision
    5. Applying thermal throttling for sustained inference estimate

    Args:
        model: Parsed model information
        device: Target device profile
        target_fps: Target framerate for real-time inference (default 30)

    Returns:
        ComputeCheckResult with pass/fail and timing estimates
    """
    # Determine model compute requirement in GFLOPS
    model_gflops = (model.estimated_flops or 0) / 1e9
    if model_gflops == 0:
        # Fallback: estimate from parameter count
        # Rough heuristic: 2 FLOPs per parameter per inference
        model_gflops = (model.num_parameters or 0) * 2 / 1e9

    # Determine best compute path and effective throughput
    compute_method, peak_gops, efficiency = _select_compute_path(device, model)

    # Apply quantisation speedup
    quant_speedup = QUANTISATION_SPEEDUP.get(model.precision, 1.0)

    # Calculate effective throughput in GOPS
    effective_gops = peak_gops * efficiency * quant_speedup

    # Estimate inference time
    if effective_gops > 0:
        inference_ms = (model_gflops / effective_gops) * 1000
    else:
        inference_ms = float("inf")

    # Sustained inference with thermal throttling
    sustained_ms = inference_ms / THERMAL_THROTTLE_FACTOR

    # Frame budget
    frame_budget_ms = 1000 / target_fps if target_fps > 0 else float("inf")

    # Pass/fail — use sustained time since real deployments run continuously
    passed = sustained_ms <= frame_budget_ms

    # Utilisation as percentage of frame budget
    utilisation = (sustained_ms / frame_budget_ms) * 100 if frame_budget_ms > 0 else 100

    # Generate recommendation if failed
    recommendation = _generate_recommendation(
        model, device, sustained_ms, frame_budget_ms, compute_method
    )

    return ComputeCheckResult(
        passed=passed,
        estimated_inference_ms=round(inference_ms, 1),
        sustained_inference_ms=round(sustained_ms, 1),
        target_framerate=target_fps,
        frame_budget_ms=round(frame_budget_ms, 1),
        compute_method=compute_method,
        device_peak_gops=peak_gops,
        effective_gops=round(effective_gops, 2),
        model_gflops=round(model_gflops, 2),
        utilisation_percent=round(utilisation, 1),
        recommendation=recommendation,
    )


def _select_compute_path(
    device: DeviceProfile, model: ModelInfo
) -> tuple:
    """Select the best available compute path for the device.

    Priority: NPU > GPU > CPU with NEON > CPU only

    Returns:
        (method_name, peak_gops, efficiency_factor)
    """
    # Check for NPU (Edge TPU, NVDLA, etc.)
    if device.has_npu:
        npu_tops = device.npu.get("tops", 0) if isinstance(device.npu, dict) else 0
        if npu_tops > 0:
            # NPU specs are usually in TOPS (tera-ops)
            # Convert to GOPS: 1 TOPS = 1000 GOPS
            npu_gops = npu_tops * 1000

            # Edge TPU only supports fully quantised INT8
            if "Edge TPU" in device.npu.get("name", "") and model.precision != "INT8":
                pass  # skip NPU, fall through to GPU/CPU
            else:
                return ("NPU", npu_gops, EFFICIENCY_FACTORS["npu"])

    # Check for GPU
    if device.has_gpu:
        gpu_gflops = device.gpu.get("gflops", 0) if isinstance(device.gpu, dict) else 0
        if gpu_gflops > 0:
            # Check if TensorRT is available (Jetson devices)
            if "tensorrt" in [f.lower() for f in device.supported_frameworks]:
                return ("GPU + TensorRT", gpu_gflops, EFFICIENCY_FACTORS["tensorrt"])
            else:
                return ("GPU delegate", gpu_gflops, EFFICIENCY_FACTORS["gpu_delegate"])

    # CPU path — check if architecture supports NEON SIMD
    arm_neon_cpus = ["Cortex-A53", "Cortex-A57", "Cortex-A72", "Cortex-A76", "Cortex-A78"]
    has_neon = any(core in device.cpu for core in arm_neon_cpus)

    if has_neon:
        return ("CPU (NEON SIMD)", device.gops, EFFICIENCY_FACTORS["cpu_neon"])
    else:
        return ("CPU only", device.gops, EFFICIENCY_FACTORS["cpu_only"])


def _generate_recommendation(
    model: ModelInfo,
    device: DeviceProfile,
    sustained_ms: float,
    frame_budget_ms: float,
    compute_method: str,
) -> str:
    """Generate actionable recommendation when compute check fails."""
    if sustained_ms <= frame_budget_ms:
        return ""

    suggestions = []

    # Suggest GPU delegate if not already using it
    if "GPU" not in compute_method and device.has_gpu:
        suggestions.append("Enable GPU delegate for faster inference")

    # Suggest quantisation if not INT8
    if model.precision != "INT8":
        speedup = QUANTISATION_SPEEDUP["INT8"] / QUANTISATION_SPEEDUP.get(
            model.precision, 1.0
        )
        new_time = sustained_ms / speedup
        suggestions.append(
            f"Quantise to INT8 (~{new_time:.0f} ms, "
            f"{speedup:.1f}x speedup)"
        )

    # Suggest reduced input resolution
    if model.input_shape and len(model.input_shape) >= 3:
        spatial = model.input_shape[1] if len(model.input_shape) == 4 else model.input_shape[0]
        if spatial > 160:
            # FLOPs scale roughly quadratically with spatial dimensions
            reduced = 160
            scale = (reduced / spatial) ** 2
            new_time = sustained_ms * scale
            suggestions.append(
                f"Reduce input resolution from {spatial}x{spatial} to "
                f"{reduced}x{reduced} (~{new_time:.0f} ms)"
            )

    # Suggest lower framerate
    if frame_budget_ms < sustained_ms:
        achievable_fps = int(1000 / sustained_ms) if sustained_ms > 0 else 0
        if achievable_fps > 0:
            suggestions.append(
                f"Lower target framerate to {achievable_fps} fps "
                f"(currently achievable)"
            )

    if suggestions:
        return " | ".join(suggestions[:2])  # max 2 suggestions
    return "Consider a smaller model architecture or more powerful hardware"
