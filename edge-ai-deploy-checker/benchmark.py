#!/usr/bin/env python3
"""
Edge AI Deploy Checker — Real Model Benchmark

Benchmarks real CNN models on your hardware using ONNX Runtime.
Creates models of varying sizes, runs inference, and measures
actual timing.

Requirements:
    pip install numpy onnxruntime onnx

Usage:
    python benchmark.py

Results are saved to benchmark_results.json
"""

import sys, os, time, json
import numpy as np
from pathlib import Path

print()
print("=" * 65)
print("  EDGE AI DEPLOY CHECKER — REAL MODEL BENCHMARK")
print("=" * 65)
print()

# ── Check dependencies ────────────────────────────────

print("  Checking environment...")
py_version = sys.version.split()[0]
print(f"  Python:        {py_version}")

try:
    import onnxruntime as ort
    print(f"  ONNX Runtime:  {ort.__version__}")
except ImportError:
    print("  ONNX Runtime not installed. Run: pip install onnxruntime")
    sys.exit(1)

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    print(f"  ONNX:          {onnx.__version__}")
except ImportError:
    print("  ONNX not installed. Run: pip install onnx")
    sys.exit(1)

# Detect platform
is_pi = False
try:
    with open("/proc/cpuinfo") as f:
        if "Raspberry" in f.read() or "BCM" in f.read():
            is_pi = True
            print("  Device:        Raspberry Pi detected")
except FileNotFoundError:
    pass

print()

# ── Model creation ────────────────────────────────────

print("  Creating benchmark models...")
os.makedirs("benchmark_models", exist_ok=True)


def make_conv_relu(prefix, in_ch, out_ch, input_name, stride=1):
    nodes, inits = [], []
    w = np.random.randn(out_ch, in_ch, 3, 3).astype(np.float32) * 0.01
    b = np.zeros(out_ch).astype(np.float32)
    inits.append(numpy_helper.from_array(w, name=f"{prefix}_w"))
    inits.append(numpy_helper.from_array(b, name=f"{prefix}_b"))
    nodes.append(helper.make_node(
        "Conv", [input_name, f"{prefix}_w", f"{prefix}_b"], [f"{prefix}_c"],
        kernel_shape=[3, 3], strides=[stride, stride], pads=[1, 1, 1, 1]
    ))
    nodes.append(helper.make_node("Relu", [f"{prefix}_c"], [f"{prefix}_o"]))
    return nodes, inits, f"{prefix}_o"


def create_model(name, num_blocks, input_size, base_ch=32, num_classes=1000):
    nodes, inits = [], []
    cur, cur_ch = "input", 3

    for i in range(num_blocks):
        out_ch = min(base_ch * (2 ** (i // 2)), 512)
        stride = 2 if (i % 2 == 0 and i > 0) else 1
        n, ini, out = make_conv_relu(f"b{i}", cur_ch, out_ch, cur, stride)
        nodes.extend(n)
        inits.extend(ini)
        cur, cur_ch = out, out_ch

    nodes.append(helper.make_node("GlobalAveragePool", [cur], ["gap"]))
    nodes.append(helper.make_node("Flatten", ["gap"], ["flat"], axis=1))

    fc_w = np.random.randn(cur_ch, num_classes).astype(np.float32) * 0.01
    fc_b = np.random.randn(1, num_classes).astype(np.float32) * 0.01
    inits.append(numpy_helper.from_array(fc_w, name="fc_w"))
    inits.append(numpy_helper.from_array(fc_b, name="fc_b"))
    nodes.append(helper.make_node("MatMul", ["flat", "fc_w"], ["mm"]))
    nodes.append(helper.make_node("Add", ["mm", "fc_b"], ["output"]))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, input_size, input_size])
    outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, num_classes])
    graph = helper.make_graph(nodes, name, [inp], [outp], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    fp = f"benchmark_models/{name}.onnx"
    onnx.save(model, fp)
    sz = os.path.getsize(fp) / (1024 * 1024)
    pc = sum(int(np.prod(i.dims)) for i in inits)
    print(f"    {name}: {sz:.1f} MB, {pc / 1e6:.1f}M params")
    return fp, sz, pc, input_size


models = []
models.append(("Small CNN (224x224)", *create_model("small_cnn_224", 8, 224, 32)))
models.append(("Medium CNN (224x224)", *create_model("medium_cnn_224", 14, 224, 64)))
models.append(("Large CNN (224x224)", *create_model("large_cnn_224", 20, 224, 64)))
models.append(("Small CNN (640x640)", *create_model("small_cnn_640", 8, 640, 32)))
print()

# ── Benchmark ─────────────────────────────────────────

print("  Running inference benchmarks (5 warmup + 20 timed)...")
print()

WARMUP, TIMED = 5, 20
results = []

for mname, mpath, fsz, pcount, ires in models:
    print(f"  {mname}...", end=" ", flush=True)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(mpath, opts, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    data = np.random.rand(*ishape).astype(np.float32)

    for _ in range(WARMUP):
        sess.run(None, {iname: data})

    times = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        sess.run(None, {iname: data})
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p95 = sorted(times)[int(len(times) * 0.95)]
    fps = 1000 / avg if avg > 0 else 0

    print(f"avg={avg:.1f}ms  p95={p95:.1f}ms  ({fps:.1f} fps)")

    results.append({
        "model_name": mname, "file_size_mb": round(fsz, 1),
        "parameters": int(pcount), "input_resolution": ires,
        "avg_ms": round(avg, 1), "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1), "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1), "fps": round(fps, 1),
        "meets_30fps": avg <= 33.3, "meets_15fps": avg <= 66.7,
    })

# ── System info ───────────────────────────────────────

si = {"python": py_version, "onnxruntime": ort.__version__, "threads": 4}
try:
    with open("/proc/cpuinfo") as f:
        for l in f:
            if "model name" in l:
                si["cpu"] = l.split(":")[1].strip()
                break
            elif l.startswith("Model"):
                si["board"] = l.split(":")[1].strip()
except:
    pass
try:
    with open("/proc/meminfo") as f:
        for l in f:
            if "MemTotal" in l:
                si["ram_mb"] = int(l.split()[1]) // 1024
                break
except:
    pass
try:
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        si["cpu_temp_c"] = int(f.read().strip()) / 1000
except:
    pass

# ── Display results ───────────────────────────────────

print()
print("=" * 70)
print("  RESULTS")
print("=" * 70)
print()
print(f"  {'Model':<25} {'Size':>6} {'Params':>8} {'Avg':>9} {'P95':>9} {'FPS':>7} {'30fps':>6}")
print("  " + "-" * 67)
for r in results:
    p = f"{r['parameters'] / 1e6:.1f}M"
    s = "YES" if r["meets_30fps"] else " NO"
    print(f"  {r['model_name']:<25} {r['file_size_mb']:>5.1f}M {p:>8} "
          f"{r['avg_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms {r['fps']:>6.1f} {s:>5}")

# Resolution impact
s224 = next((r for r in results if "Small" in r["model_name"] and r["input_resolution"] == 224), None)
s640 = next((r for r in results if "Small" in r["model_name"] and r["input_resolution"] == 640), None)
if s224 and s640:
    sd = s640["avg_ms"] / s224["avg_ms"] if s224["avg_ms"] > 0 else 0
    print(f"\n  Resolution impact: 224x224={s224['avg_ms']:.1f}ms vs "
          f"640x640={s640['avg_ms']:.1f}ms ({sd:.1f}x slower, {(640/224)**2:.1f}x more pixels)")

# Size impact
m224 = sorted([r for r in results if r["input_resolution"] == 224], key=lambda r: r["parameters"])
if len(m224) >= 2:
    rp = m224[-1]["parameters"] / m224[0]["parameters"] if m224[0]["parameters"] > 0 else 0
    rt = m224[-1]["avg_ms"] / m224[0]["avg_ms"] if m224[0]["avg_ms"] > 0 else 0
    print(f"  Size impact: {rp:.0f}x more params = {rt:.1f}x slower (not linear)")

print(f"\n  30fps capable: {sum(1 for r in results if r['meets_30fps'])}/{len(results)}")

# Save
with open("benchmark_results.json", "w") as f:
    json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"), "system": si, "results": results}, f, indent=2)
print(f"\n  Results saved to: benchmark_results.json")
print("=" * 70)
print()
