"""VRAM and system RAM profiling for loaded models."""

from __future__ import annotations

from typing import Any

from edgemind.core.logging import get_logger
from edgemind.models.benchmark_models import MemoryResult

logger = get_logger(__name__)


class MemoryProfiler:
    """Profile GPU VRAM and system RAM usage at model-load and inference time.

    Captures three key snapshots:
    1. After model load: how much VRAM the weights consume.
    2. During inference: peak VRAM including KV cache and activations.
    3. System RAM: RSS memory of the Python process.
    """

    def profile(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "auto",
    ) -> MemoryResult:
        """Profile memory usage for a loaded model.

        Args:
            model: Loaded model (HF transformer or llama_cpp.Llama).
            tokenizer: HuggingFace tokenizer (None for GGUF models).
            device: Device string used during loading.

        Returns:
            MemoryResult with VRAM and RAM measurements.
        """
        try:
            import psutil
            process = psutil.Process()
            system_ram_gb = round(process.memory_info().rss / 1e9, 2)
        except ImportError:
            system_ram_gb = 0.0

        is_gguf = tokenizer is None or _is_llama_cpp(model)

        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        if cuda_available:
            return self._profile_cuda(model, tokenizer, system_ram_gb, is_gguf)
        else:
            return self._profile_cpu(model, system_ram_gb, is_gguf)

    def _profile_cuda(
        self,
        model: Any,
        tokenizer: Any,
        system_ram_gb: float,
        is_gguf: bool,
    ) -> MemoryResult:
        """Profile CUDA VRAM usage."""
        import torch

        device_name = torch.cuda.get_device_name(0)
        available_vram_gb = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
        model_vram_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

        # Run 3 inference passes to capture peak VRAM with KV cache
        torch.cuda.reset_peak_memory_stats()
        if not is_gguf and tokenizer is not None:
            self._run_inference_passes(model, tokenizer, n_passes=3)

        peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        peak_vram_gb = max(peak_vram_gb, model_vram_gb)

        fits_on_device = peak_vram_gb < (available_vram_gb * 0.95)

        return MemoryResult(
            model_vram_gb=model_vram_gb,
            peak_vram_gb=peak_vram_gb,
            available_vram_gb=available_vram_gb,
            system_ram_gb=system_ram_gb,
            fits_on_device=fits_on_device,
            device_name=device_name,
        )

    def _profile_cpu(
        self,
        model: Any,
        system_ram_gb: float,
        is_gguf: bool,
    ) -> MemoryResult:
        """Profile CPU/MPS mode (no dedicated VRAM counter)."""
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_name = "Apple Silicon (MPS)"
            else:
                device_name = "CPU"
        except ImportError:
            device_name = "CPU"

        try:
            import psutil
            available_ram = round(psutil.virtual_memory().available / 1e9, 1)
            total_ram = round(psutil.virtual_memory().total / 1e9, 1)
        except ImportError:
            available_ram = 0.0
            total_ram = 0.0

        fits = system_ram_gb < (total_ram * 0.95) if total_ram > 0 else True

        return MemoryResult(
            model_vram_gb=0.0,
            peak_vram_gb=0.0,
            available_vram_gb=available_ram,
            system_ram_gb=system_ram_gb,
            fits_on_device=fits,
            device_name=device_name,
        )

    def _run_inference_passes(self, model: Any, tokenizer: Any, n_passes: int = 3) -> None:
        """Run short inference passes to capture peak activation + KV cache VRAM."""
        import torch

        device = next(model.parameters()).device
        prompt = "The quick brown fox"
        enc = tokenizer(prompt, return_tensors="pt").to(device)

        for _ in range(n_passes):
            with torch.no_grad():
                try:
                    model.generate(**enc, max_new_tokens=20, do_sample=False)
                except Exception as exc:
                    logger.debug(f"Inference pass failed during memory profiling: {exc}")
                    break
            torch.cuda.empty_cache()


def _is_llama_cpp(model: Any) -> bool:
    """Detect if model is a llama_cpp.Llama instance."""
    return type(model).__name__ == "Llama" and hasattr(model, "create_completion")
