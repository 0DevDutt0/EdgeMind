"""GPU detection, CUDA compatibility checks, and VRAM estimation utilities."""

from __future__ import annotations

from edgemind.models.benchmark_models import QuantizationMethod

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console(legacy_windows=False)

_BYTES_PER_PARAM: dict[str, float] = {
    QuantizationMethod.BF16: 2.0,
    QuantizationMethod.FP16: 2.0,
    QuantizationMethod.INT8: 1.0,
    QuantizationMethod.INT4_NF4: 0.5,
    QuantizationMethod.INT4_FP4: 0.5,
    QuantizationMethod.GPTQ_4BIT: 0.5,
    QuantizationMethod.AWQ_4BIT: 0.5,
    QuantizationMethod.GGUF_Q2K: 0.31,
    QuantizationMethod.GGUF_Q3KM: 0.43,
    QuantizationMethod.GGUF_Q4KM: 0.55,
    QuantizationMethod.GGUF_Q5KM: 0.68,
    QuantizationMethod.GGUF_Q8: 1.0,
    QuantizationMethod.GGUF_F16: 2.0,
}


class GPUInfo:
    """Complete GPU information and compatibility checks for EdgeMind."""

    def detect(self) -> dict:
        """Detect GPU capabilities and return a complete info dict.

        Returns:
            dict with keys: device_name, compute_capability, vram_gb,
            cuda_version, pytorch_version, is_blackwell, full_support,
            backend (cuda/mps/cpu), and system_ram_gb.
        """
        info: dict = {
            "device_name": "CPU",
            "compute_capability": None,
            "vram_gb": 0.0,
            "cuda_version": None,
            "pytorch_version": None,
            "is_blackwell": False,
            "full_support": False,
            "backend": "cpu",
            "system_ram_gb": 0.0,
            "torch_available": _TORCH_AVAILABLE,
        }

        if _PSUTIL_AVAILABLE:
            info["system_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)

        if not _TORCH_AVAILABLE:
            return info

        info["pytorch_version"] = torch.__version__

        if torch.cuda.is_available():
            info["backend"] = "cuda"
            info["device_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"sm_{major}{minor}"
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["is_blackwell"] = major == 12
            cuda_ver = info["cuda_version"]
            try:
                info["full_support"] = info["is_blackwell"] and (
                    float(cuda_ver.split(".")[0] + "." + cuda_ver.split(".")[1]) >= 13.0
                )
            except (ValueError, IndexError):
                info["full_support"] = False
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["backend"] = "mps"
            info["device_name"] = "Apple Silicon (MPS)"
            info["full_support"] = True
        else:
            info["backend"] = "cpu"

        return info

    def print_setup_guide(self, gpu_info: dict) -> None:
        """Print RTX 5090 setup instructions when CUDA 13 is not detected.

        Args:
            gpu_info: Dict returned by detect().
        """
        pytorch_ver = gpu_info.get("pytorch_version", "unknown")
        cuda_ver = gpu_info.get("cuda_version", "unknown")

        content = Text()
        content.append("Your RTX 5090 (Blackwell, sm_120) requires ", style="white")
        content.append("PyTorch nightly with CUDA 13", style="bold yellow")
        content.append(".\n", style="white")
        content.append(f"Current: PyTorch {pytorch_ver}, CUDA {cuda_ver}\n\n", style="dim")
        content.append("Run these commands to fix:\n\n", style="white")
        content.append(
            "  pip uninstall torch torchvision torchaudio -y\n"
            "  pip install --pre torch torchvision torchaudio \\\n"
            "      --index-url https://download.pytorch.org/whl/nightly/cu130\n\n",
            style="bold green",
        )
        content.append("Verify with:\n", style="white")
        content.append(
            '  python -c "import torch; print(torch.cuda.get_device_capability(0))"\n',
            style="cyan",
        )
        content.append("  # Expected: (12, 0)\n\n", style="dim")
        content.append(
            "EdgeMind will continue in limited CPU mode until fixed.", style="yellow"
        )

        panel = Panel(
            content,
            title="[bold yellow]âš   RTX 5090 Setup Required[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
        console.print(panel)

    def estimate_vram_requirement(
        self,
        param_billions: float,
        method: QuantizationMethod,
    ) -> float:
        """Estimate total VRAM required for a model at a given quantization level.

        Accounts for model weights, KV cache (15%), and activation overhead (5%).

        Args:
            param_billions: Model parameter count in billions.
            method: Target quantization method.

        Returns:
            Estimated VRAM requirement in GB.
        """
        bytes_per_param = _BYTES_PER_PARAM.get(method, 2.0)
        model_bytes = param_billions * 1e9 * bytes_per_param
        kv_cache_overhead = model_bytes * 0.15
        activation_overhead = model_bytes * 0.05
        total_bytes = model_bytes + kv_cache_overhead + activation_overhead
        return round(total_bytes / 1e9, 2)

    def check_fits_in_vram(
        self,
        param_billions: float,
        method: QuantizationMethod,
        available_vram_gb: float,
    ) -> tuple[bool, float]:
        """Check whether a model fits in available VRAM.

        Args:
            param_billions: Model parameter count in billions.
            method: Target quantization method.
            available_vram_gb: Available VRAM on the target device.

        Returns:
            Tuple of (fits: bool, required_gb: float).
        """
        required_gb = self.estimate_vram_requirement(param_billions, method)
        fits = required_gb <= available_vram_gb
        return fits, required_gb

    def print_gpu_status(self, gpu_info: dict) -> None:
        """Print a concise GPU status line to the console.

        Args:
            gpu_info: Dict returned by detect().
        """
        backend = gpu_info.get("backend", "cpu")
        if backend == "cuda":
            name = gpu_info["device_name"]
            cc = gpu_info["compute_capability"]
            vram = gpu_info["vram_gb"]
            if gpu_info.get("is_blackwell") and not gpu_info.get("full_support"):
                self.print_setup_guide(gpu_info)
                console.print(
                    "[yellow]Running in CPU-fallback mode (sm_120 without CUDA 13)[/yellow]"
                )
            elif gpu_info.get("full_support"):
                console.print(
                    f"[green]GPU:[/green] {name} ({cc}) â€” {vram} GB VRAM "
                    f"[green]âœ“ Full support[/green]"
                )
            else:
                console.print(
                    f"[green]GPU:[/green] {name} ({cc}) â€” {vram} GB VRAM"
                )
        elif backend == "mps":
            ram = gpu_info.get("system_ram_gb", 0)
            console.print(f"[green]Backend:[/green] Apple MPS â€” {ram} GB unified memory")
        else:
            ram = gpu_info.get("system_ram_gb", 0)
            console.print(
                f"[yellow]Backend:[/yellow] CPU only â€” {ram} GB RAM "
                f"[dim](no GPU detected)[/dim]"
            )

