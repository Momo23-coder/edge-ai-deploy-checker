# Edge AI Deploy Checker

A command-line tool that checks whether an AI model is ready to deploy on a target edge device. It analyses model size, memory requirements, compute demands, and framework compatibility — catching deployment issues before they happen.

Built by a hardware engineer who got tired of AI teams handing over models that wouldn't fit on the target device.

## Real-World Benchmarks

We benchmarked 4 CNN models on a Raspberry Pi 4 (ARM Cortex-A72, 4GB RAM) using ONNX Runtime. Here's what we found:

| Model | Size | Params | Avg Inference | FPS | 30fps? |
|-------|------|--------|--------------|-----|--------|
| Small CNN (224×224) | 5.5 MB | 1.4M | 189.4 ms | 5.3 | No |
| Medium CNN (224×224) | 73.8 MB | 19.4M | 814.8 ms | 1.2 | No |
| Large CNN (224×224) | 127.9 MB | 33.5M | 830.4 ms | 1.2 | No |
| Small CNN (640×640) | 5.5 MB | 1.4M | 1686.5 ms | 0.6 | No |

**Key findings:**
- CPU-only inference is not enough for real-time AI — even a 1.4M param model only manages 5 fps
- Input resolution scales almost linearly with pixel count (8.2× more pixels = 8.9× slower)
- 23× more parameters only means 4.4× slower — model size and inference time are not linear

Read the full analysis: [I Benchmarked 4 AI Models on a Raspberry Pi 4](https://medium.com/@mosesjamesthlama)

## Features

- **Model analysis** — Parse `.tflite` and `.onnx` model files to extract size, parameters, estimated FLOPs, and operation types
- **Device profiles** — Built-in profiles for common edge devices (Raspberry Pi 4/5, NVIDIA Jetson Nano, Coral Dev Board, generic ARM) with customisable specs
- **Memory check** — Compare model memory footprint against available device RAM, accounting for OS overhead and activation memory
- **Compute check** — Estimate inference time based on device compute capability, with thermal throttling for sustained workloads
- **Real benchmarking** — Run actual inference on your hardware with the included benchmark script
- **Deployment report** — Clear pass/fail output with actionable recommendations

## Installation

```bash
git clone https://github.com/Momo23-coder/edge-ai-deploy-checker.git
cd edge-ai-deploy-checker
pip install -e .
```

For benchmarking:
```bash
pip install numpy onnxruntime onnx
```

## Quick Start

```bash
# List available device profiles
python -m edge_check.cli --list-devices

# Check a TFLite model against a Raspberry Pi 4
python -m edge_check.cli model.tflite --device rpi4

# Check an ONNX model against a Jetson Nano
python -m edge_check.cli model.onnx --device jetson-nano

# Use custom device specs
python -m edge_check.cli model.tflite --ram 2048 --cpu "ARM Cortex-A72" --freq 1.5 --cores 4

# Run the full benchmark on your hardware
python benchmark.py
```

## Example Output

```
═══════════════════════════════════════════════════════
  Edge AI Deploy Checker — Deployment Readiness Report
═══════════════════════════════════════════════════════

  Model:    mobilenet_v2.tflite
  Device:   Raspberry Pi 4 (4GB)
  Runtime:  TFLITE

── Model Analysis ──────────────────────────────────────
  Parameters:     3.5M
  Model size:     13.3 MB (FP32)
  Est. FLOPs:     7.1 MFLOPS
  Input shape:    [1, 224, 224, 3]

── Memory Check ────────────────────────────────────────
  Total required:         ~103 MB
  Available (est.):       3,296 MB
  Status:                 ✓ PASS

── Compute Check ───────────────────────────────────────
  Est. inference time:    2 ms (burst)
  Sustained inference:    3 ms (after thermal)
  Target framerate:       30 fps (33 ms budget)
  Status:                 ✓ PASS

═══════════════════════════════════════════════════════
  RESULT: 2/2 checks passed
  Model is ready for deployment on this device
═══════════════════════════════════════════════════════
```

## Device Profiles

Built-in profiles for common edge hardware:

| Device | RAM | CPU | ID |
|--------|-----|-----|----|
| Raspberry Pi 4 (4GB) | 4 GB | Cortex-A72 x4 @ 1.5GHz | `rpi4` |
| Raspberry Pi 5 (8GB) | 8 GB | Cortex-A76 x4 @ 2.4GHz | `rpi5` |
| NVIDIA Jetson Nano | 4 GB | Cortex-A57 x4 + 128 CUDA | `jetson-nano` |
| Google Coral Dev Board | 1 GB | Cortex-A53 x4 + Edge TPU | `coral` |
| Generic ARM Cortex-A53 | 1 GB | Cortex-A53 x4 @ 1.2GHz | `arm-a53` |
| Generic ARM Cortex-A72 | 2 GB | Cortex-A72 x4 @ 1.8GHz | `arm-a72` |

Add custom profiles in `~/.edge-check/devices.json`.

## Project Structure

```
edge-ai-deploy-checker/
├── README.md
├── LICENSE
├── setup.py
├── benchmark.py              # Real hardware benchmark (ONNX Runtime)
├── benchmark_results.json    # Results from Raspberry Pi 4
├── edge_check/
│   ├── __init__.py
│   ├── cli.py                # Command-line interface
│   ├── model_parser.py       # Parse TFLite and ONNX models
│   ├── device_profiles.py    # Built-in device specs
│   ├── memory_check.py       # RAM and memory analysis
│   └── compute_check.py      # Inference time estimation
├── devices/
│   └── profiles.json         # Device profile definitions
└── requirements.txt
```

## Roadmap

- [x] Project structure and CLI
- [x] TFLite and ONNX model parser
- [x] Device profile system with JSON configs
- [x] Memory check module
- [x] Compute check with thermal throttling
- [x] Real hardware benchmark (ONNX Runtime)
- [x] Raspberry Pi 4 benchmark results
- [ ] Framework compatibility checker
- [ ] Quantisation advisor
- [ ] Report export (JSON / PDF)
- [ ] Unit tests
- [ ] PyPI package release

## Why This Exists

Most AI tooling focuses on model training and cloud deployment. But in edge AI — where models run on physical devices in vehicles, factories, and remote locations — the hardware is the bottleneck.

This tool exists because the gap between "my model works" and "my model works on this device" shouldn't take days of trial and error to close.

## Related Posts

- [Cloud AI vs Edge AI: What Changes When the Model Leaves the Server](https://medium.com/@mosesjamesthlama)
- [I Benchmarked 4 AI Models on a Raspberry Pi 4 — Here's What the Numbers Actually Tell You](https://medium.com/@mosesjamesthlama)

## Contributing

Contributions are welcome. If you work with edge AI hardware and want to add a device profile or improve the estimation models, PRs are appreciated.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Moses Thlama James** — Edge AI Hardware Engineer

- [Medium](https://medium.com/@mosesjamesthlama)
- [LinkedIn](https://www.linkedin.com/in/moses-tjames01/)
