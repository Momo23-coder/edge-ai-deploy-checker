# Edge AI Deploy Checker

A command-line tool that checks whether an AI model is ready to deploy on a target edge device. It analyses model size, memory requirements, compute demands, and framework compatibility — catching deployment issues before they happen.

Built by a hardware engineer who got tired of AI teams handing over models that wouldn't fit on the target device.

## The problem

Most AI deployment failures aren't model failures — they're hardware mismatches. A model that runs perfectly in the cloud can fail on an edge device because:

- The model's memory footprint exceeds available RAM
- The inference time is too slow for real-time requirements
- The model format isn't compatible with the device's framework/runtime
- Power and thermal constraints aren't accounted for
- The device's processor doesn't support the required operations

This tool catches these issues early, before you spend hours debugging on-device.

## Features

- **Model analysis** — Parse `.tflite`, `.onnx`, and saved model files to extract size, parameters, estimated FLOPs, and operation types
- **Device profiles** — Built-in profiles for common edge devices (Raspberry Pi 4, NVIDIA Jetson Nano, Coral Dev Board, generic ARM Cortex) with customisable specs
- **Memory check** — Compare model memory footprint (weights + activations + runtime overhead) against available device RAM, accounting for OS and system overhead
- **Compute estimation** — Estimate inference time based on device compute capability and model FLOPs
- **Framework compatibility** — Check if model operations are supported by the target runtime (TFLite, ONNX Runtime, OpenVINO)
- **Quantisation advisor** — Recommend quantisation strategies if the model is too large, with estimated size and speed improvements
- **Deployment report** — Generate a clear pass/fail report with actionable recommendations

## Installation

```bash
pip install edge-ai-deploy-checker
```

Or install from source:

```bash
git clone https://github.com/Momo23-coder/edge-ai-deploy-checker.git
cd edge-ai-deploy-checker
pip install -e .
```

## Quick start

```bash
# Check a TFLite model against a Raspberry Pi 4
edge-check model.tflite --device rpi4

# Check an ONNX model against a Jetson Nano
edge-check model.onnx --device jetson-nano

# Use custom device specs
edge-check model.tflite --ram 2048 --cpu "ARM Cortex-A72" --freq 1.5 --cores 4

# Generate a detailed report
edge-check model.tflite --device rpi4 --report deployment_report.json
```

## Example output

```
═══════════════════════════════════════════════════════
  Edge AI Deploy Checker — Deployment Readiness Report
═══════════════════════════════════════════════════════

Model:    mobilenet_v2.tflite
Device:   Raspberry Pi 4 (4GB)
Runtime:  TensorFlow Lite

── Model Analysis ──────────────────────────────────────
  Parameters:     3.4M
  Model size:     13.2 MB (FP32) → 3.4 MB (INT8)
  Est. FLOPs:     600M
  Input shape:    [1, 224, 224, 3]
  Operations:     Conv2D, DepthwiseConv2D, ReLU6, Softmax

── Memory Check ────────────────────────────────────────
  Model weights:          3.4 MB (INT8)
  Activation memory:      ~12.8 MB
  Runtime overhead:       ~50 MB
  Total required:         ~66.2 MB
  Available (est.):       3,200 MB
  Status:                 ✓ PASS

── Compute Check ───────────────────────────────────────
  Est. inference time:    85 ms (CPU, 4 cores)
  Target framerate:       30 fps (33 ms budget)
  Status:                 ✗ FAIL — too slow for 30fps
  Recommendation:         Use GPU delegate or reduce
                          input resolution to 160x160

── Framework Compatibility ─────────────────────────────
  Unsupported ops:        None
  Delegate support:       GPU delegate available
  Status:                 ✓ PASS

── Quantisation Advisor ────────────────────────────────
  Current precision:      INT8
  Further reduction:      INT4 → ~1.7 MB (experimental)
  Dynamic range quant:    Not applicable (already INT8)
  Status:                 Already optimised

── Thermal Estimate ────────────────────────────────────
  Sustained load:         ~3.5W under inference
  Device TDP:             ~7.5W
  Throttle risk:          LOW (with passive heatsink)
  Status:                 ✓ PASS

═══════════════════════════════════════════════════════
  RESULT: 4/5 checks passed — 1 issue to resolve
═══════════════════════════════════════════════════════
```

## Device profiles

Built-in profiles for common edge hardware:

| Device | RAM | CPU | Compute | ID |
|--------|-----|-----|---------|-----|
| Raspberry Pi 4 (4GB) | 4 GB | Cortex-A72 x4 @ 1.5GHz | ~13.5 GOPS | `rpi4` |
| Raspberry Pi 5 | 8 GB | Cortex-A76 x4 @ 2.4GHz | ~32 GOPS | `rpi5` |
| NVIDIA Jetson Nano | 4 GB | Cortex-A57 x4 + 128 CUDA | ~472 GFLOPS (GPU) | `jetson-nano` |
| Coral Dev Board | 1 GB | Cortex-A53 x4 + Edge TPU | 4 TOPS (TPU) | `coral` |
| Generic ARM Cortex-A53 | 1 GB | Cortex-A53 x4 @ 1.2GHz | ~4.8 GOPS | `arm-a53` |
| Generic ARM Cortex-A72 | 2 GB | Cortex-A72 x4 @ 1.8GHz | ~16 GOPS | `arm-a72` |

Add custom profiles in `~/.edge-check/devices.json`.

## Project structure

```
edge-ai-deploy-checker/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── edge_check/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── model_parser.py     # Parse TFLite, ONNX model files
│   ├── device_profiles.py  # Built-in device specs
│   ├── memory_check.py     # RAM and memory analysis
│   ├── compute_check.py    # Inference time estimation
│   ├── framework_check.py  # Op compatibility checking
│   ├── quantisation.py     # Quantisation recommendations
│   ├── thermal.py          # Power and thermal estimates
│   └── report.py           # Report generation
├── devices/
│   └── profiles.json       # Device profile definitions
├── tests/
│   ├── test_model_parser.py
│   ├── test_memory_check.py
│   ├── test_compute_check.py
│   └── test_sample_models/
│       └── README.md       # Instructions for test models
└── examples/
    ├── check_mobilenet.py
    └── custom_device.py
```

## Roadmap

- [x] Project structure and CLI skeleton
- [ ] TFLite model parser (size, params, ops)
- [ ] ONNX model parser
- [ ] Device profile system with JSON configs
- [ ] Memory check module
- [ ] Compute estimation module
- [ ] Framework compatibility checker
- [ ] Quantisation advisor
- [ ] Thermal / power estimation
- [ ] Report generation (terminal + JSON)
- [ ] Unit tests
- [ ] PyPI package release

## Contributing

Contributions are welcome! See the [issues](https://github.com/Momo23-coder/edge-ai-deploy-checker/issues) for good starting points. If you work with edge AI hardware and want to add a device profile, PRs are especially appreciated.

## Why this exists

Most AI tooling focuses on model training and cloud deployment. But in edge AI — where models run on physical devices in vehicles, factories, and remote locations — the hardware is the bottleneck. This tool exists because the gap between "my model works" and "my model works on this device" shouldn't take days of trial and error to close.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Moses Thlama James** — Edge AI Hardware Engineer
- [Medium](https://medium.com/@mosesjamesthlama)
- [LinkedIn](https://www.linkedin.com/in/moses-tjames01/)
