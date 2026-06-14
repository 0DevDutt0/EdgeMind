"""AWQ (Activation-aware Weight Quantization) via AutoAWQ."""

from __future__ import annotations

import time
from pathlib import Path

from edgemind.core.logging import console, get_logger
from edgemind.models.benchmark_models import QuantizationMethod, QuantizationResult
from edgemind.quantization.base import BaseQuantizer

logger = get_logger(__name__)


class AWQQuantizer(BaseQuantizer):
    """Quantize models using AWQ (Activation-aware Weight Quantization).

    AWQ protects the most salient 1% of weights (determined by activation
    magnitudes) from quantization error. This gives better quality than
    vanilla INT4 and is faster to quantize than GPTQ (no per-layer Hessian).

    AWQ models are also hardware-efficient — purpose-built CUDA kernels
    deliver faster inference than bitsandbytes at the same bit width.

    Install: pip install autoawq
    """

    def quantize(
        self,
        model_id: str,
        output_dir: str,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        calibration_samples: int = 128,
    ) -> QuantizationResult:
        """Run AWQ quantization on a model.

        Args:
            model_id: HuggingFace model ID or local path.
            output_dir: Directory to save the quantized model.
            bits: Target bit width (default 4).
            group_size: Quantization group size (128 is standard).
            zero_point: Use zero-point quantization (recommended).
            calibration_samples: Number of calibration samples (fewer than GPTQ).

        Returns:
            QuantizationResult with size and AWQ config.

        Raises:
            ImportError: If autoawq is not installed.
            RuntimeError: If CUDA is not available.
        """
        self._check_gpu_requirements()

        try:
            from awq import AutoAWQForCausalLM  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "autoawq is required for AWQ quantization.\n"
                "Install: pip install autoawq"
            ) from exc

        from transformers import AutoTokenizer  # type: ignore[import]

        console.print(f"[cyan]AWQ quantization:[/cyan] {model_id} (bits={bits})")
        console.print(
            "[dim]AWQ is faster than GPTQ (~5-20 min). "
            "Activation-aware weights selected automatically.[/dim]"
        )

        start = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoAWQForCausalLM.from_pretrained(model_id, trust_remote_code=True)

        quant_config = {
            "zero_point": zero_point,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": "GEMM",
        }

        console.print(
            f"[cyan]Running AWQ calibration ({calibration_samples} samples)...[/cyan]"
        )
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data="pileval",
            n_samples=calibration_samples,
        )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_quantized(output_dir, safetensors=True)
        tokenizer.save_pretrained(output_dir)

        duration = time.perf_counter() - start
        size_gb = self.get_output_size_gb(output_dir)
        logger.info(f"AWQ quantization complete: {size_gb:.2f} GB in {duration:.0f}s")

        return QuantizationResult(
            model_id=model_id,
            method=QuantizationMethod.AWQ_4BIT,
            output_dir=output_dir,
            size_gb=size_gb,
            quantization_config=quant_config,
            calibration_samples_used=calibration_samples,
            duration_seconds=round(duration, 1),
        )
