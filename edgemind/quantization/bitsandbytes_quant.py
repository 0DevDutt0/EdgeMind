"""INT8/INT4-NF4/INT4-FP4 quantization via bitsandbytes."""

from __future__ import annotations

import time
from pathlib import Path

from edgemind.core.logging import console, get_logger
from edgemind.models.benchmark_models import QuantizationMethod, QuantizationResult
from edgemind.quantization.base import BaseQuantizer

logger = get_logger(__name__)


class BitsAndBytesQuantizer(BaseQuantizer):
    """Quantize models using bitsandbytes library.

    Supports INT8, INT4-NF4, and INT4-FP4 quantization.
    Fastest method — no calibration data required.
    Trade-off: slightly lower quality than GPTQ at the same bit width.

    Double quantization (use_double_quant=True) additionally quantizes
    the quantization constants themselves, saving ~0.4 bits/param (~400MB
    for a 7B model) at negligible quality cost.
    """

    def quantize(
        self,
        model_id: str,
        output_dir: str,
        bits: int = 4,
        quant_type: str = "nf4",
        use_double_quant: bool = True,
        compute_dtype: str = "bfloat16",
    ) -> QuantizationResult:
        """Quantize a model with bitsandbytes and save to disk.

        Args:
            model_id: HuggingFace model ID or local path.
            output_dir: Directory to save the quantized model.
            bits: Target bit width (4 or 8).
            quant_type: NF4-specific quant type ("nf4" or "fp4").
            use_double_quant: Enable double quantization for smaller files.
            compute_dtype: Dtype for compute operations ("bfloat16", "float16").

        Returns:
            QuantizationResult with size, method, and bitsandbytes config.

        Raises:
            RuntimeError: If CUDA is not available.
            ValueError: If bits is not 4 or 8.
        """
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")

        self._check_gpu_requirements()

        import torch
        from transformers import (  # type: ignore[import]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        compute_dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = compute_dtype_map.get(compute_dtype, torch.bfloat16)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_compute_dtype=torch_dtype,
        )

        method_label = (
            QuantizationMethod.INT4_NF4
            if bits == 4 and quant_type == "nf4"
            else QuantizationMethod.INT4_FP4
            if bits == 4
            else QuantizationMethod.INT8
        )

        console.print(
            f"[cyan]Loading[/cyan] {model_id} "
            f"[dim]({bits}-bit {quant_type if bits == 4 else 'int8'})[/dim]"
        )

        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

        console.print(f"[cyan]Saving quantized model to[/cyan] {output_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        duration = time.perf_counter() - start
        size_gb = self.get_output_size_gb(output_dir)

        config_dict = bnb_config.to_dict() if hasattr(bnb_config, "to_dict") else {}

        logger.info(
            f"Quantization complete: {size_gb:.2f} GB, {vram_gb:.2f} GB VRAM at load"
        )

        return QuantizationResult(
            model_id=model_id,
            method=method_label,
            output_dir=output_dir,
            size_gb=size_gb,
            vram_at_load_gb=vram_gb,
            quantization_config=config_dict,
            duration_seconds=round(duration, 1),
        )
