#!/usr/bin/env python3
"""
Edge AI Real-Device Benchmark — Raspberry Pi 4
===============================================

This script does everything:
1. Checks your Python and TFLite setup
2. Creates real AI models (MobileNet V2 and ResNet-50)
3. Runs actual inference on your Raspberry Pi
4. Measures real timing (not estimates)
5. Outputs results for your blog post

Run on your Raspberry Pi 4:
    python3 rpi4_benchmark.py

If TFLite isn't installed, the script will tell you
exactly what to run to install it.
"""

import sys
import os
import time
import json
from pathlib import Path


# ── Step 1: Check environment ──────────────────────────

print()
print("=" * 65)
print("  EDGE AI REAL-DEVICE BENCHMARK — Raspberry Pi 4")
print("=" * 65)
print()
print("  Step 1: Checking environment...")
print()

# Check Python version
py_version = sys.version.split()[0]
print(f"  Python version:  {py_version}")
if sys.version_info < (3, 7):
    print("  ERROR: Python 3.7+ required.")
    sys.exit(1)
print(f"  Platform:        {sys.platform}")

# Try to detect if we're on a Pi
is_pi = False
try:
    with open("/proc/cpuinfo") as f:
        cpuinfo = f.read()
    if "Raspberry" in cpuinfo or "BCM" in cpuinfo:
        is_pi = True
        print("  Device:          Raspberry Pi detected")
    else:
        print("  Device:          Not a Raspberry Pi (results still valid)")
except FileNotFoundError:
    print("  Device:          Cannot detect (not Linux?)")

# Check for numpy
try:
    import numpy as np
    print(f"  NumPy:           {np.__version__} (installed)")
except ImportError:
    print("  NumPy:           NOT INSTALLED")
    print()
    print("  Run this first:")
    print("    pip3 install numpy")
    sys.exit(1)

# Check for TFLite runtime
tflite_available = False
tflite_source = None

try:
    # Try the lightweight tflite-runtime package first (recommended for Pi)
    from tflite_runtime.interpreter import Interpreter
    tflite_available = True
    tflite_source = "tflite-runtime"
    print(f"  TFLite runtime:  installed (tflite-runtime)")
except ImportError:
    try:
        # Fall back to full TensorFlow
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        tflite_available = True
        tflite_source = "tensorflow"
        print(f"  TensorFlow:      {tf.__version__} (using tf.lite)")
    except ImportError:
        pass

if not tflite_available:
    print("  TFLite runtime:  NOT INSTALLED")
    print()
    print("  ─" * 30)
    print("  Install the lightweight TFLite runtime (recommended for Pi):")
    print()
    print("    pip3 install tflite-runtime")
    print()
    print("  Or install full TensorFlow (slower to install, uses more RAM):")
    print()
    print("    pip3 install tensorflow")
    print()
    print("  Then run this script again.")
    print("  ─" * 30)
    sys.exit(1)

print()
print("  Environment OK. Starting benchmark...")
print()


# ── Step 2: Prepare models ─────────────────────────────

print("  Step 2: Preparing models...")
print()

os.makedirs("benchmark_models", exist_ok=True)

# We need pre-built TFLite model files.
# Option A: If full TensorFlow is available, generate them
# Option B: Download pre-converted models

models_ready = []

if tflite_source == "tensorflow":
    import tensorflow as tf

    # MobileNet V2 FP32
    path_mn_fp32 = "benchmark_models/mobilenet_v2_fp32.tflite"
    if not Path(path_mn_fp32).exists():
        print("  Creating MobileNet V2 (FP32)...")
        model = tf.keras.applications.MobileNetV2(
            weights=None, input_shape=(224, 224, 3), classes=1000
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(path_mn_fp32, "wb") as f:
            f.write(tflite_model)
        print(f"    Saved: {len(tflite_model)/1024/1024:.1f} MB")
    else:
        print(f"  MobileNet V2 (FP32): already exists")
    models_ready.append((path_mn_fp32, "MobileNet V2 (FP32)"))

    # MobileNet V2 INT8
    path_mn_int8 = "benchmark_models/mobilenet_v2_int8.tflite"
    if not Path(path_mn_int8).exists():
        print("  Creating MobileNet V2 (INT8)...")
        model = tf.keras.applications.MobileNetV2(
            weights=None, input_shape=(224, 224, 3), classes=1000
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep_data_mn():
            for _ in range(50):
                yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

        converter.representative_dataset = rep_data_mn
        tflite_model = converter.convert()
        with open(path_mn_int8, "wb") as f:
            f.write(tflite_model)
        print(f"    Saved: {len(tflite_model)/1024/1024:.1f} MB")
    else:
        print(f"  MobileNet V2 (INT8): already exists")
    models_ready.append((path_mn_int8, "MobileNet V2 (INT8)"))

    # ResNet-50 FP32
    path_rn_fp32 = "benchmark_models/resnet50_fp32.tflite"
    if not Path(path_rn_fp32).exists():
        print("  Creating ResNet-50 (FP32)... (this may take a minute)")
        model = tf.keras.applications.ResNet50(
            weights=None, input_shape=(224, 224, 3), classes=1000
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(path_rn_fp32, "wb") as f:
            f.write(tflite_model)
        print(f"    Saved: {len(tflite_model)/1024/1024:.1f} MB")
    else:
        print(f"  ResNet-50 (FP32): already exists")
    models_ready.append((path_rn_fp32, "ResNet-50 (FP32)"))

    # ResNet-50 INT8
    path_rn_int8 = "benchmark_models/resnet50_int8.tflite"
    if not Path(path_rn_int8).exists():
        print("  Creating ResNet-50 (INT8)...")
        model = tf.keras.applications.ResNet50(
            weights=None, input_shape=(224, 224, 3), classes=1000
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep_data_rn():
            for _ in range(30):
                yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

        converter.representative_dataset = rep_data_rn
        tflite_model = converter.convert()
        with open(path_rn_int8, "wb") as f:
            f.write(tflite_model)
        print(f"    Saved: {len(tflite_model)/1024/1024:.1f} MB")
    else:
        print(f"  ResNet-50 (INT8): already exists")
    models_ready.append((path_rn_int8, "ResNet-50 (INT8)"))

else:
    # tflite-runtime only — can't generate models
    # Check if models already exist from a previous run
    expected = [
        ("benchmark_models/mobilenet_v2_fp32.tflite", "MobileNet V2 (FP32)"),
        ("benchmark_models/mobilenet_v2_int8.tflite", "MobileNet V2 (INT8)"),
        ("benchmark_models/resnet50_fp32.tflite", "ResNet-50 (FP32)"),
        ("benchmark_models/resnet50_int8.tflite", "ResNet-50 (INT8)"),
    ]
    for path, name in expected:
        if Path(path).exists():
            models_ready.append((path, name))
            print(f"  {name}: found")
        else:
            print(f"  {name}: NOT FOUND")

    if not models_ready:
        print()
        print("  No model files found. You have two options:")
        print()
        print("  Option A: Install full TensorFlow to generate models:")
        print("    pip3 install tensorflow")
        print("    python3 rpi4_benchmark.py")
        print()
        print("  Option B: Generate models on another machine, then copy")
        print("    the benchmark_models/ folder to your Pi.")
        sys.exit(1)

print()
print(f"  {len(models_ready)} models ready for benchmarking.")
print()


# ── Step 3: Run real benchmarks ────────────────────────

print("  Step 3: Running inference benchmarks...")
print()
print("  Each model runs 10 warmup passes, then 50 timed passes.")
print("  This measures REAL inference time on YOUR hardware.")
print()

WARMUP_RUNS = 10
TIMED_RUNS = 50

results = []

for model_path, model_name in models_ready:
    print(f"  Benchmarking: {model_name}")
    print(f"    Loading model...", end=" ", flush=True)

    # Load model
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    # Get model file size
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"OK ({file_size_mb:.1f} MB, input: {input_shape.tolist()})")

    # Create random input data
    if input_dtype == np.float32:
        input_data = np.random.rand(*input_shape).astype(np.float32)
        precision = "FP32"
    elif input_dtype == np.int8:
        input_data = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
        precision = "INT8"
    elif input_dtype == np.uint8:
        input_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
        precision = "INT8"
    else:
        input_data = np.random.rand(*input_shape).astype(np.float32)
        precision = "FP32"

    # Warmup runs (let CPU caches and frequency settle)
    print(f"    Warmup ({WARMUP_RUNS} runs)...", end=" ", flush=True)
    for _ in range(WARMUP_RUNS):
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
    print("OK")

    # Timed runs
    print(f"    Benchmarking ({TIMED_RUNS} runs)...", end=" ", flush=True)
    times = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # convert to ms

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    p50_ms = sorted(times)[len(times) // 2]
    p95_ms = sorted(times)[int(len(times) * 0.95)]
    std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5

    print("OK")
    print(f"    Results:  avg={avg_ms:.1f}ms  min={min_ms:.1f}ms  "
          f"max={max_ms:.1f}ms  p95={p95_ms:.1f}ms")
    print()

    result = {
        "model_name": model_name,
        "model_path": model_path,
        "file_size_mb": round(file_size_mb, 1),
        "input_shape": input_shape.tolist(),
        "precision": precision,
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "avg_ms": round(avg_ms, 1),
        "min_ms": round(min_ms, 1),
        "max_ms": round(max_ms, 1),
        "p50_ms": round(p50_ms, 1),
        "p95_ms": round(p95_ms, 1),
        "std_ms": round(std_ms, 1),
        "fps": round(1000 / avg_ms, 1) if avg_ms > 0 else 0,
        "meets_30fps": avg_ms <= 33.3,
        "meets_15fps": avg_ms <= 66.7,
    }
    results.append(result)


# ── Step 4: System info ────────────────────────────────

print("  Step 4: Collecting system info...")
print()

system_info = {
    "python_version": py_version,
    "tflite_source": tflite_source,
    "platform": sys.platform,
    "is_raspberry_pi": is_pi,
}

# Get CPU info on Linux
try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                system_info["cpu_model"] = line.split(":")[1].strip()
                break
            elif line.startswith("Model"):
                system_info["board_model"] = line.split(":")[1].strip()
except FileNotFoundError:
    pass

# Get RAM info
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal"):
                mem_kb = int(line.split()[1])
                system_info["ram_mb"] = mem_kb // 1024
                break
except FileNotFoundError:
    pass

# Get CPU temperature (Pi specific)
try:
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        temp_mc = int(f.read().strip())
        system_info["cpu_temp_c"] = temp_mc / 1000
except FileNotFoundError:
    pass

# Get CPU frequency
try:
    with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
        freq_khz = int(f.read().strip())
        system_info["cpu_freq_mhz"] = freq_khz // 1000
except FileNotFoundError:
    pass


# ── Step 5: Display results ───────────────────────────

print()
print("=" * 65)
print("  BENCHMARK RESULTS — Raspberry Pi 4")
print("=" * 65)
print()

# System info
print("  SYSTEM")
print("  " + "─" * 50)
for key, val in system_info.items():
    label = key.replace("_", " ").title()
    print(f"  {label:<25} {val}")
print()

# Results table
print("  INFERENCE TIMING")
print("  " + "─" * 60)
print(f"  {'Model':<24} {'Size':>6} {'Avg':>8} {'P50':>8} "
      f"{'P95':>8} {'FPS':>6} {'30fps?':>7}")
print("  " + "─" * 60)

for r in results:
    fps_ok = "YES" if r["meets_30fps"] else "NO"
    print(
        f"  {r['model_name']:<24} "
        f"{r['file_size_mb']:>5.1f}M "
        f"{r['avg_ms']:>7.1f}ms "
        f"{r['p50_ms']:>7.1f}ms "
        f"{r['p95_ms']:>7.1f}ms "
        f"{r['fps']:>5.1f} "
        f"{'  ✓ ' + fps_ok if r['meets_30fps'] else '  ✗ ' + fps_ok:>7}"
    )

print()

# Quantisation impact
fp32_models = [r for r in results if "FP32" in r["model_name"]]
int8_models = [r for r in results if "INT8" in r["model_name"]]

if fp32_models and int8_models:
    print("  QUANTISATION IMPACT")
    print("  " + "─" * 60)
    print(f"  {'Model':<20} {'FP32':>10} {'INT8':>10} {'Speedup':>10} {'Size reduction':>15}")
    print("  " + "─" * 60)

    model_bases = set()
    for r in results:
        base = r["model_name"].replace(" (FP32)", "").replace(" (INT8)", "")
        model_bases.add(base)

    for base in sorted(model_bases):
        fp32 = next((r for r in results if r["model_name"] == f"{base} (FP32)"), None)
        int8 = next((r for r in results if r["model_name"] == f"{base} (INT8)"), None)
        if fp32 and int8:
            speedup = fp32["avg_ms"] / int8["avg_ms"] if int8["avg_ms"] > 0 else 0
            size_reduction = (1 - int8["file_size_mb"] / fp32["file_size_mb"]) * 100
            print(
                f"  {base:<20} "
                f"{fp32['avg_ms']:>8.1f}ms "
                f"{int8['avg_ms']:>8.1f}ms "
                f"{speedup:>9.1f}x "
                f"{size_reduction:>13.0f}%"
            )

    print()

# Key findings
print("  KEY FINDINGS")
print("  " + "─" * 60)

fastest = min(results, key=lambda r: r["avg_ms"])
slowest = max(results, key=lambda r: r["avg_ms"])
print(f"  Fastest:         {fastest['model_name']} at {fastest['avg_ms']:.1f}ms ({fastest['fps']:.0f} fps)")
print(f"  Slowest:         {slowest['model_name']} at {slowest['avg_ms']:.1f}ms ({slowest['fps']:.0f} fps)")

real_time = [r for r in results if r["meets_30fps"]]
print(f"  30fps capable:   {len(real_time)}/{len(results)} models")

print()

# Save results to JSON
output_file = "benchmark_results.json"
output = {
    "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "system_info": system_info,
    "config": {
        "warmup_runs": WARMUP_RUNS,
        "timed_runs": TIMED_RUNS,
        "target_fps": 30,
    },
    "results": results,
}

with open(output_file, "w") as f:
    json.dump(output, f, indent=2)

print(f"  Full results saved to: {output_file}")
print()
print("=" * 65)
print("  Benchmark complete. Use the results above for your blog post!")
print("=" * 65)
print()
