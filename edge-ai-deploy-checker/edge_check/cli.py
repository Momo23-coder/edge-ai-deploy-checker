"""
Command-line interface for Edge AI Deploy Checker.

Usage:
    edge-check model.tflite --device rpi4
    edge-check model.onnx --device jetson-nano
    edge-check model.tflite --ram 2048 --cpu "ARM Cortex-A72" --freq 1.5 --cores 4
"""

import argparse
import sys
from .model_parser import parse_model
from .device_profiles import get_device, list_devices, create_custom_device
from .memory_check import check_memory
from .compute_check import check_compute


def print_header():
    print()
    print("═" * 55)
    print("  Edge AI Deploy Checker — Deployment Readiness Report")
    print("═" * 55)
    print()


def print_section(title: str):
    print(f"── {title} " + "─" * (50 - len(title)))


def main():
    parser = argparse.ArgumentParser(
        prog="edge-check",
        description="Check if an AI model is ready to deploy on edge hardware.",
    )
    parser.add_argument("model", nargs="?", help="Path to model file (.tflite or .onnx)")
    parser.add_argument("--device", "-d", help="Device profile ID (e.g., rpi4, jetson-nano)")
    parser.add_argument("--list-devices", action="store_true", help="List available device profiles")
    parser.add_argument("--ram", type=int, help="Custom device RAM in MB")
    parser.add_argument("--cpu", type=str, help="Custom device CPU name")
    parser.add_argument("--cores", type=int, default=4, help="Custom device CPU cores")
    parser.add_argument("--freq", type=float, default=1.5, help="Custom device CPU frequency (GHz)")
    parser.add_argument("--fps", type=int, default=30, help="Target framerate for real-time check (default: 30)")
    parser.add_argument("--report", type=str, help="Save report to JSON file")
    parser.add_argument("--version", action="version", version="edge-check 0.1.0")

    args = parser.parse_args()

    # List devices mode
    if args.list_devices:
        print("\nAvailable device profiles:\n")
        print(f"  {'ID':<18} {'Name':<30} {'RAM':<12} {'CPU'}")
        print(f"  {'─'*18} {'─'*30} {'─'*12} {'─'*25}")
        for device_id, name, ram, cpu in list_devices():
            print(f"  {device_id:<18} {name:<30} {ram:<12} {cpu}")
        print()
        print("  Use --device <ID> to select a device profile.")
        print("  Add custom profiles in ~/.edge-check/devices.json")
        print()
        return

    # Require model path for checks
    if not args.model:
        parser.print_help()
        sys.exit(1)

    # create device profile
    device = None
    if args.device:
        device = get_device(args.device)
        if not device:
            print(f"\nError: Unknown device '{args.device}'.")
            print("Use --list-devices to see available profiles.\n")
            sys.exit(1)
    elif args.ram and args.cpu:
        device = create_custom_device(
            ram_mb=args.ram,
            cpu=args.cpu,
            cores=args.cores,
            freq_ghz=args.freq,
        )
    else:
        print("\nError: Specify --device <ID> or provide custom specs (--ram, --cpu).\n")
        sys.exit(1)

    # Parse model
    try:
        model = parse_model(args.model)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}\n")
        sys.exit(1)

    # Run checks
    print_header()

    print(f"  Model:    {model.filepath}")
    print(f"  Device:   {device.name}")
    print(f"  Runtime:  {model.format.upper()}")
    print()

    # Model analysis
    print_section("Model Analysis")
    print(model.summary())
    print()

    # Memory check
    print_section("Memory Check")
    mem_result = check_memory(model, device)
    print(mem_result.summary())
    print()

    # Compute check
    print_section("Compute Check")
    compute_result = check_compute(model, device, target_fps=args.fps)
    print(compute_result.summary())
    print()

    # Results summary
    results = [mem_result.passed, compute_result.passed]
    checks_passed = sum(results)
    total_checks = len(results)

    print("═" * 55)
    print(f"  RESULT: {checks_passed}/{total_checks} checks passed")
    if checks_passed < total_checks:
        print(f"  {total_checks - checks_passed} issue(s) to resolve")
    else:
        print("  Model is ready for deployment on this device")
    print("═" * 55)
    print()


if __name__ == "__main__":
    main()
