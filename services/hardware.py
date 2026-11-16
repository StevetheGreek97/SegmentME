import torch
import platform
from dataclasses import dataclass

from PyQt6.QtWidgets import QMessageBox


@dataclass(frozen=True)
class TrainingDevice:
    """Where training runs, and how to say so in one line."""
    device: object        # what ultralytics takes: 0, [0, 1], "mps" or "cpu"
    label: str            # e.g. "NVIDIA GeForce RTX 3060 (12 GB)"
    fast: bool            # a GPU; False: the CPU (much slower)


def describe_device() -> TrainingDevice:
    """The best device for training, without showing anything (CUDA GPU,
    then Apple Silicon, then the CPU)."""
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        label = f"{torch.cuda.get_device_name(0)} ({props.total_memory / 1024 ** 3:.0f} GB)"
        if count > 1:
            label += f" + {count - 1} more GPU{'s' if count > 2 else ''}"
        return TrainingDevice(list(range(count)) if count > 1 else 0, label, True)
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return TrainingDevice("mps", "Apple Silicon GPU", True)
    return TrainingDevice("cpu", "CPU (no graphics card found)", False)


def auto_detect_device(parent=None):
    """
    Detects the best available device for training and shows detailed system info.

    Priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback with warning)
    """
    torch_version = torch.__version__
    cuda_version = torch.version.cuda or "N/A"
    system = platform.system()
    message = (
        f"🧠 PyTorch version: {torch_version}\n"
        f"⚙️ CUDA Toolkit: {cuda_version}\n"
        f"🖥️ Platform: {system}\n\n"
    )

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = []
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            devices.append(f"GPU {i}: {name} • Compute Capability {cap[0]}.{cap[1]} • {mem:.1f} GB VRAM")
        message += f"✅ {num_gpus} CUDA GPU(s) detected:\n- " + "\n- ".join(devices)
        message += "\n\n🚀 Training will use CUDA device(s)."
        QMessageBox.information(parent, "Device Info", message)
        return list(range(num_gpus)) if num_gpus > 1 else 0

    elif system == "Darwin" and torch.backends.mps.is_available():
        message += (
            "✅ Apple Silicon detected.\n"
            "🛠️ MPS (Metal Performance Shaders) is available.\n\n"
            "🚀 Training will use MPS."
        )
        QMessageBox.information(parent, "Device Info", message)
        return "mps"

    else:
        message += (
            "⚠️ No compatible GPU found.\n"
            "🪫 Training will fall back to CPU. Expect significantly slower performance."
        )
        QMessageBox.warning(parent, "Device Warning", message)
        return "cpu"
