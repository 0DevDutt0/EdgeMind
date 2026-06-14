"""HuggingFace → GGUF conversion via llama-cpp-python."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from edgemind.core.logging import console, get_logger
from edgemind.models.benchmark_models import QuantizationMethod, QuantizationResult
from edgemind.quantization.base import BaseQuantizer

logger = get_logger(__name__)

QUANTIZATION_DESCRIPTIONS: dict[str, str] = {
    "q2_k": (
        "2-bit quantization. Smallest files. Significant quality loss. "
        "Use only for extremely memory-constrained devices."
    ),
    "q3_k_m": (
        "3-bit quantization. Small files. Noticeable quality loss. "
        "Suitable for Raspberry Pi or very old phones."
    ),
    "q4_k_m": (
        "4-bit quantization. RECOMMENDED. Best quality/size tradeoff. "
        "Industry standard for production edge deployment."
    ),
    "q5_k_m": (
        "5-bit quantization. Higher quality, larger files. "
        "Good for Jetson or when 4-bit quality is insufficient."
    ),
    "q8_0": (
        "8-bit quantization. Near full precision quality. "
        "Files are larger but quality is excellent."
    ),
    "f16": (
        "Full Float16 precision. Reference quality. "
        "No quantization. Use for highest quality testing."
    ),
}

_METHOD_MAP: dict[str, QuantizationMethod] = {
    "q2_k": QuantizationMethod.GGUF_Q2K,
    "q3_k_m": QuantizationMethod.GGUF_Q3KM,
    "q4_k_m": QuantizationMethod.GGUF_Q4KM,
    "q5_k_m": QuantizationMethod.GGUF_Q5KM,
    "q8_0": QuantizationMethod.GGUF_Q8,
    "f16": QuantizationMethod.GGUF_F16,
}


class GGUFConverter(BaseQuantizer):
    """Convert HuggingFace models to GGUF format for llama.cpp.

    GGUF enables CPU+GPU hybrid inference, cross-platform deployment,
    and efficient memory mapping. Required for Ollama and llama.cpp backends.

    Two-step pipeline:
    1. Convert HF checkpoint → GGUF F16 (lossless)
    2. Quantize GGUF F16 → target quantization (q4_k_m, q8_0, etc.)

    Requires: pip install llama-cpp-python
    For RTX 5090 CUDA: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python
    """

    def quantize(
        self,
        model_id: str,
        output_dir: str,
        **kwargs: object,
    ) -> QuantizationResult:
        """Wrapper to satisfy BaseQuantizer interface — delegates to convert()."""
        quantization = str(kwargs.get("quantization", "q4_k_m"))
        return self.convert(model_id=model_id, output_dir=output_dir, quantization=quantization)

    def convert(
        self,
        model_id: str,
        output_dir: str,
        quantization: str = "q4_k_m",
    ) -> QuantizationResult:
        """Convert and quantize a HuggingFace model to GGUF format.

        Downloads the model if a HF model ID is given, then runs a two-step
        conversion: HF → GGUF F16 → target quantization level.

        Args:
            model_id: HuggingFace model ID or local path to downloaded model.
            output_dir: Output directory for GGUF files.
            quantization: Target quantization level (e.g. "q4_k_m").

        Returns:
            QuantizationResult pointing to the final GGUF file.

        Raises:
            ImportError: If llama-cpp-python is not installed.
            RuntimeError: If conversion or quantization fails.
        """
        self._check_llama_cpp()

        quantization = quantization.lower()
        if quantization not in QUANTIZATION_DESCRIPTIONS:
            raise ValueError(
                f"Unknown GGUF quantization: {quantization}. "
                f"Valid: {list(QUANTIZATION_DESCRIPTIONS.keys())}"
            )

        desc = QUANTIZATION_DESCRIPTIONS[quantization]
        console.print(f"[cyan]GGUF conversion:[/cyan] {model_id} → {quantization.upper()}")
        console.print(f"[dim]{desc}[/dim]")

        start = time.perf_counter()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # If model_id is a HF ID (not a local path), download it first
        model_local_path = model_id
        if not Path(model_id).exists():
            model_local_path = self._download_hf_model(model_id, output_dir)

        model_name = Path(model_local_path).name.replace("/", "_")

        # Step 1: Convert to GGUF F16
        f16_path = Path(output_dir) / f"{model_name}.f16.gguf"
        console.print(f"[cyan]Step 1/2:[/cyan] Converting to GGUF F16 → {f16_path.name}")
        self._convert_to_f16_gguf(model_local_path, str(f16_path))

        # Step 2: Quantize to target format
        if quantization == "f16":
            final_path = f16_path
        else:
            final_path = Path(output_dir) / f"{model_name}.{quantization}.gguf"
            console.print(
                f"[cyan]Step 2/2:[/cyan] Quantizing to {quantization.upper()} "
                f"→ {final_path.name}"
            )
            self._quantize_gguf(str(f16_path), str(final_path), quantization)
            # Remove intermediate F16 to save space
            if f16_path.exists() and quantization != "f16":
                f16_path.unlink()
                logger.info("Removed intermediate F16 GGUF to save disk space")

        duration = time.perf_counter() - start
        size_gb = self.get_output_size_gb(output_dir)

        # Smoke test
        test_result = self.test_gguf_inference(str(final_path))
        if test_result["success"]:
            console.print(
                f"[green]✓ GGUF inference test passed[/green] "
                f"(output: {test_result['output'][:40]!r})"
            )
        else:
            console.print(f"[yellow]⚠ GGUF inference test failed: {test_result.get('error')}[/yellow]")

        method = _METHOD_MAP.get(quantization, QuantizationMethod.GGUF_Q4KM)
        logger.info(f"GGUF conversion complete: {final_path}, {size_gb:.2f} GB")

        return QuantizationResult(
            model_id=model_id,
            method=method,
            output_dir=str(final_path),
            size_gb=size_gb,
            quantization_config={"quantization": quantization, "format": "gguf"},
            duration_seconds=round(duration, 1),
        )

    def test_gguf_inference(self, gguf_path: str) -> dict:
        """Quick smoke test: load GGUF model and generate 5 tokens.

        Args:
            gguf_path: Path to the .gguf model file.

        Returns:
            Dict with keys: success (bool), output (str), load_time_s (float),
            and optionally error (str) on failure.
        """
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError:
            return {"success": False, "output": "", "load_time_s": 0.0,
                    "error": "llama-cpp-python not installed"}

        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        t0 = time.perf_counter()
        try:
            llm = Llama(
                model_path=gguf_path,
                n_ctx=512,
                n_gpu_layers=-1 if cuda_available else 0,
                verbose=False,
            )
            output = llm("Hello", max_tokens=5, echo=False)
            load_time = time.perf_counter() - t0
            text = output["choices"][0]["text"] if output.get("choices") else ""
            return {"success": True, "output": text, "load_time_s": round(load_time, 2)}
        except Exception as exc:
            return {"success": False, "output": "", "load_time_s": 0.0, "error": str(exc)}

    def _check_llama_cpp(self) -> None:
        """Verify llama-cpp-python is importable."""
        try:
            import llama_cpp  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for GGUF conversion.\n"
                "CPU-only: pip install llama-cpp-python\n"
                "RTX 5090 CUDA support:\n"
                "  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            ) from exc

    def _download_hf_model(self, model_id: str, cache_dir: str) -> str:
        """Download a HuggingFace model to a local directory.

        Args:
            model_id: HuggingFace model repository ID.
            cache_dir: Directory to download the model into.

        Returns:
            Local path to the downloaded model directory.
        """
        from huggingface_hub import snapshot_download  # type: ignore[import]

        console.print(f"[cyan]Downloading {model_id} from HuggingFace...[/cyan]")
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(Path(cache_dir) / "hf_cache" / model_id.replace("/", "_")),
        )
        logger.info(f"Downloaded to {local_path}")
        return local_path

    def _convert_to_f16_gguf(self, model_path: str, output_path: str) -> None:
        """Run llama.cpp convert script to produce GGUF F16.

        Args:
            model_path: Local directory of the HF model.
            output_path: Destination path for the F16 GGUF file.

        Raises:
            RuntimeError: If the conversion script fails.
        """
        # Try the standard convert path first
        convert_candidates = [
            [sys.executable, "-m", "llama_cpp.tools.convert_hf_to_gguf",
             model_path, "--outtype", "f16", "--outfile", output_path],
            [sys.executable, "-m", "llama_cpp.convert_hf_to_gguf",
             model_path, "--outtype", "f16", "--outfile", output_path],
        ]
        for cmd in convert_candidates:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return
            logger.debug(f"Convert command failed: {result.stderr[:200]}")

        raise RuntimeError(
            f"GGUF F16 conversion failed for {model_path}.\n"
            "Ensure llama-cpp-python is installed and the model is in HF safetensors format.\n"
            "Alternatively, download llama.cpp and run convert_hf_to_gguf.py manually."
        )

    def _quantize_gguf(self, f16_path: str, output_path: str, quantization: str) -> None:
        """Quantize a GGUF F16 file to the target quantization level.

        Args:
            f16_path: Path to the F16 GGUF input file.
            output_path: Path for the quantized GGUF output.
            quantization: Target quantization type (e.g. "q4_k_m").

        Raises:
            RuntimeError: If the quantize step fails.
        """
        try:
            from llama_cpp import llama_cpp  # type: ignore[import]
            llama_cpp.llama_backend_init(False)
            ftype_map = {
                "q2_k": 10,
                "q3_k_m": 12,
                "q4_k_m": 15,
                "q5_k_m": 17,
                "q8_0": 7,
            }
            ftype = ftype_map.get(quantization, 15)
            # llama_model_quantize requires a params struct as 4th arg in newer
            # llama-cpp-python versions. Build it if available, else fall through.
            if hasattr(llama_cpp, "llama_model_quantize_params"):
                params = llama_cpp.llama_model_quantize_params()
                params.ntype = ftype
                result = llama_cpp.llama_model_quantize(
                    f16_path.encode(),
                    output_path.encode(),
                    llama_cpp.ctypes.byref(params),
                )
            else:
                result = llama_cpp.llama_model_quantize(
                    f16_path.encode(),
                    output_path.encode(),
                    ftype,
                )
            if result != 0:
                raise RuntimeError(f"llama_model_quantize returned {result}")
        except (ImportError, AttributeError, TypeError):
            # Fallback: try llama-quantize binary via subprocess
            cmd = ["llama-quantize", f16_path, output_path, quantization.upper()]
            result_proc = subprocess.run(cmd, capture_output=True, text=True)
            if result_proc.returncode != 0:
                raise RuntimeError(
                    f"GGUF quantization failed.\n"
                    f"stderr: {result_proc.stderr[:400]}\n"
                    "Install llama.cpp binary and ensure 'llama-quantize' is on PATH."
                )
