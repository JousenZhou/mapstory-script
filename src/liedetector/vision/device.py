"""Device selection: which backend can actually run, and why not.

Three probes make up "GPU supported", in dependency order:
  1. NVIDIA driver present (nvidia-smi answers)   - the hardware exists
  2. torch.cuda.is_available()                    - this torch build can drive it
  3. import tensorrt                              - the INT8 engine can be built/loaded
The UI disables the GPU option with the first failed probe's reason; the detector walks the
same probes and degrades TensorRT INT8 -> torch CUDA -> CPU, never silently.
"""

import shutil
import subprocess
from dataclasses import dataclass

from src.liedetector.vision.logger import get_logger

log = get_logger("device")

BACKEND_TRT = "tensorrt-int8"
BACKEND_TRT_CACHE = "tensorrt-int8(缓存)"
BACKEND_CUDA = "cuda(回退)"
BACKEND_CPU = "cpu"

REQUEST_CPU = "cpu"
REQUEST_GPU = "gpu"
REQUEST_AUTO = "auto"


@dataclass
class GPUOption:
    """What the UI needs to render the device combo box."""

    enabled: bool
    reason: str  # shown as tooltip; states why the option is (dis)abled
    gpu_name: str = ""


def nvidia_driver_present() -> tuple[bool, str]:
    """A GPU exists if nvidia-smi answers; its first line names the card."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False, ""
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    if out.returncode != 0:
        return False, ""
    name = (out.stdout or "").strip().splitlines()
    return True, name[0].strip() if name else ""


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


def gpu_option_state() -> GPUOption:
    """First failed probe wins; each failure is one actionable sentence."""
    driver_ok, gpu_name = nvidia_driver_present()
    if not driver_ok:
        return GPUOption(False, "未检测到 NVIDIA 显卡/驱动（nvidia-smi 不可用），仅可使用 CPU")
    if not torch_cuda_available():
        return GPUOption(
            False,
            "检测到 NVIDIA 显卡，但当前 PyTorch 不支持 CUDA。\n"
            "需安装 CUDA 版：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130",
            gpu_name,
        )
    if not tensorrt_available():
        return GPUOption(
            False,
            f"检测到{(' ' + gpu_name) if gpu_name else ' NVIDIA 显卡'}，但缺少 TensorRT，无法构建 INT8 引擎。\n"
            "安装：pip install tensorrt onnx onnxslim",
            gpu_name,
        )
    return GPUOption(True, f"GPU：{gpu_name or 'NVIDIA 显卡'}，TensorRT INT8 引擎就绪", gpu_name)


def resolve_request(requested: str) -> str:
    """auto -> gpu when the full stack is present, else cpu. Explicit choices pass through;
    the detector handles a request the stack cannot honor by degrading with a logged reason."""
    if requested in (REQUEST_CPU, REQUEST_GPU):
        return requested
    return REQUEST_GPU if gpu_option_state().enabled else REQUEST_CPU


def backend_label(backend: str) -> str:
    return {
        BACKEND_TRT: "GPU · TensorRT INT8",
        BACKEND_TRT_CACHE: "GPU · TensorRT INT8（缓存引擎）",
        BACKEND_CUDA: "GPU · CUDA（TRT 不可用，已回退）",
        BACKEND_CPU: "CPU",
    }.get(backend, backend)
