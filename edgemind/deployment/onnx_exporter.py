"""Export quantized HuggingFace models to ONNX format via Optimum."""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console

from edgemind.core.logging import get_logger

logger = get_logger(__name__)
console = Console(legacy_windows=False)


class ONNXExporter:
    """Export HuggingFace models to ONNX format for cross-platform inference.

    ONNX Runtime enables inference on CPU, CUDA, DirectML (Windows GPU),
    CoreML (Apple), and TensorRT without GPU-specific frameworks.
    """

    def export(
        self,
        model_id: str,
        output_dir: str,
        task: str = "text-generation-with-past",
        dtype: str = "fp32",
        optimize: bool = True,
    ) -> dict:
        """Export a model to ONNX format.

        Args:
            model_id: HuggingFace model ID or local path.
            output_dir: Directory to save ONNX model files.
            task: Optimum task name for the export.
            dtype: Export dtype ("fp32", "fp16", "int8").
            optimize: Run ONNX graph optimization after export.

        Returns:
            Dict with output_path, export_time_s, and onnx_size_gb.

        Raises:
            ImportError: If optimum is not installed.
            RuntimeError: If export fails.
        """
        try:
            from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "optimum is required for ONNX export.\n"
                "Install: pip install optimum[onnxruntime]"
            ) from exc

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]Exporting to ONNX:[/cyan] {model_id}")
        console.print(f"[dim]Task: {task} | dtype: {dtype}[/dim]")

        t0 = time.perf_counter()

        kwargs: dict = {"export": True}
        if dtype == "fp16":
            import torch
            kwargs["torch_dtype"] = torch.float16

        model = ORTModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.save_pretrained(output_dir)

        from transformers import AutoTokenizer  # type: ignore[import]

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(output_dir)

        duration = time.perf_counter() - t0

        if optimize:
            console.print("[cyan]Running ONNX graph optimization...[/cyan]")
            self._optimize(output_dir)

        size_gb = sum(
            f.stat().st_size for f in Path(output_dir).rglob("*.onnx")
        ) / 1e9

        test_result = self._test_onnx_inference(output_dir, model_id)

        console.print(
            f"\n[bold]ONNX Inference Code Snippet:[/bold]\n"
            f"[cyan]"
            f"from optimum.onnxruntime import ORTModelForCausalLM\n"
            f"from transformers import AutoTokenizer\n"
            f"\n"
            f'model = ORTModelForCausalLM.from_pretrained("{output_dir}")\n'
            f'tokenizer = AutoTokenizer.from_pretrained("{output_dir}")\n'
            f'\ninputs = tokenizer("Hello", return_tensors="pt")\n'
            f"output = model.generate(**inputs, max_new_tokens=50)\n"
            f'print(tokenizer.decode(output[0], skip_special_tokens=True))\n'
            f"[/cyan]"
        )

        return {
            "output_path": output_dir,
            "export_time_s": round(duration, 1),
            "onnx_size_gb": round(size_gb, 3),
            "test_result": test_result,
        }

    def _optimize(self, model_dir: str) -> None:
        """Run ONNX Runtime graph optimization on the exported model."""
        try:
            from optimum.onnxruntime import OptimizationConfig, ORTOptimizer  # type: ignore[import]

            optimizer = ORTOptimizer.from_pretrained(model_dir)
            config = OptimizationConfig(optimization_level=2)
            optimizer.optimize(save_dir=model_dir, optimization_config=config)
            console.print("[green]âœ“ ONNX graph optimization complete[/green]")
        except Exception as exc:
            logger.warning(f"ONNX optimization skipped: {exc}")

    def _test_onnx_inference(self, model_dir: str, model_id: str) -> dict:
        """Run a short inference pass to validate the ONNX export.

        Args:
            model_dir: Directory containing the exported ONNX model.
            model_id: Original model ID for tokenizer loading.

        Returns:
            Dict with success (bool) and output (str).
        """
        try:
            from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore[import]
            from transformers import AutoTokenizer  # type: ignore[import]

            model = ORTModelForCausalLM.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            inputs = tokenizer("Hello", return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=10)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            console.print(f"[green]âœ“ ONNX inference test:[/green] {text!r}")
            return {"success": True, "output": text}
        except Exception as exc:
            return {"success": False, "output": "", "error": str(exc)}

