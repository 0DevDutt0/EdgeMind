"""Abstract base class for all EdgeMind quantization methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from edgemind.core.logging import get_logger
from edgemind.models.benchmark_models import QuantizationResult

logger = get_logger(__name__)


class BaseQuantizer(ABC):
    """Abstract base for all quantization methods.

    Subclasses implement quantize() for their specific method.
    Common utilities (size calculation, GPU checks) are provided here.
    """

    @abstractmethod
    def quantize(
        self,
        model_id: str,
        output_dir: str,
        **kwargs: object,
    ) -> QuantizationResult:
        """Quantize the given model and save to output_dir.

        Args:
            model_id: HuggingFace model ID or local directory path.
            output_dir: Destination directory for the quantized model.
            **kwargs: Method-specific quantization parameters.

        Returns:
            QuantizationResult with size, method, and config details.
        """

    def get_output_size_gb(self, output_dir: str) -> float:
        """Calculate total disk size of all files in a directory.

        Args:
            output_dir: Path to the directory containing quantized model files.

        Returns:
            Total size in gigabytes.
        """
        total_bytes = sum(
            f.stat().st_size
            for f in Path(output_dir).rglob("*")
            if f.is_file()
        )
        return round(total_bytes / 1e9, 3)

    def _check_gpu_requirements(self) -> None:
        """Verify that a CUDA-capable GPU is available.

        Raises:
            RuntimeError: If CUDA is not available.
        """
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is not installed. For RTX 5090 (sm_120), install:\n"
                "  pip install --pre torch torchvision torchaudio \\\n"
                "      --index-url https://download.pytorch.org/whl/nightly/cu130"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU not detected. Quantization with bitsandbytes/GPTQ/AWQ "
                "requires a CUDA-capable GPU.\n"
                "Run 'edgemind info' to check your hardware setup."
            )
