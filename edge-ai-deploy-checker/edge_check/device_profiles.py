"""
Device profile management for edge AI hardware.

Loads built-in device profiles and supports custom profiles
from ~/.edge-check/devices.json
"""

import json
import os
from pathlib import Path
from typing import Optional


DEFAULT_PROFILES_PATH = Path(__file__).parent.parent / "devices" / "profiles.json"
USER_PROFILES_PATH = Path.home() / ".edge-check" / "devices.json"


class DeviceProfile:
    """Represents an edge device's hardware capabilities."""

    def __init__(self, device_id: str, config: dict):
        self.device_id = device_id
        self.name = config["name"]
        self.ram_mb = config["ram_mb"]
        self.cpu = config["cpu"]
        self.cores = config["cores"]
        self.freq_ghz = config["freq_ghz"]
        self.arch = config["arch"]
        self.gops = config["gops"]
        self.gpu = config.get("gpu")
        self.npu = config.get("npu")
        self.tdp_watts = config["tdp_watts"]
        self.max_temp_c = config["max_temp_c"]
        self.supported_frameworks = config["supported_frameworks"]
        self.os_overhead_mb = config["os_overhead_mb"]
        self.notes = config.get("notes", "")

    @property
    def available_ram_mb(self) -> int:
        """RAM available after OS and system overhead."""
        return self.ram_mb - self.os_overhead_mb

    @property
    def has_gpu(self) -> bool:
        return self.gpu is not None

    @property
    def has_npu(self) -> bool:
        return self.npu is not None

    def supports_framework(self, framework: str) -> bool:
        """Check if the device supports a given runtime framework."""
        return framework.lower() in [f.lower() for f in self.supported_frameworks]

    def __repr__(self):
        return f"DeviceProfile('{self.device_id}': {self.name}, {self.ram_mb}MB RAM, {self.cpu})"


def load_profiles() -> dict:
    """Load device profiles from built-in and user-defined files."""
    profiles = {}

    # Load built-in profiles
    if DEFAULT_PROFILES_PATH.exists():
        with open(DEFAULT_PROFILES_PATH) as f:
            data = json.load(f)
            for device_id, config in data["devices"].items():
                profiles[device_id] = DeviceProfile(device_id, config)

    # Load user profiles (override built-in if same ID)
    if USER_PROFILES_PATH.exists():
        with open(USER_PROFILES_PATH) as f:
            data = json.load(f)
            for device_id, config in data.get("devices", {}).items():
                profiles[device_id] = DeviceProfile(device_id, config)

    return profiles


def get_device(device_id: str) -> Optional[DeviceProfile]:
    """Get a device profile by ID."""
    profiles = load_profiles()
    return profiles.get(device_id)


def list_devices() -> list:
    """List all available device profiles."""
    profiles = load_profiles()
    return [(pid, p.name, f"{p.ram_mb}MB RAM", p.cpu) for pid, p in profiles.items()]


def create_custom_device(
    ram_mb: int,
    cpu: str,
    cores: int,
    freq_ghz: float,
    tdp_watts: float = 10.0,
    frameworks: list = None,
) -> DeviceProfile:
    """Create a custom device profile from manual specs."""
    config = {
        "name": f"Custom ({cpu})",
        "ram_mb": ram_mb,
        "cpu": cpu,
        "cores": cores,
        "freq_ghz": freq_ghz,
        "arch": "aarch64",
        "gops": cores * freq_ghz * 2,  # rough estimate
        "gpu": None,
        "npu": None,
        "tdp_watts": tdp_watts,
        "max_temp_c": 85,
        "supported_frameworks": frameworks or ["tflite", "onnxruntime"],
        "os_overhead_mb": min(int(ram_mb * 0.25), 1000),
        "notes": "Custom device profile",
    }
    return DeviceProfile("custom", config)
