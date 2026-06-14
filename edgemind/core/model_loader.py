"""Unified HuggingFace model loader with quantization-aware dispatch."""

from __future__ import annotations

from typing import Any

from edgemind.core.logging import get_logger
from edgemind.models.benchmark_models import QuantizationMethod

logger = get_logger(__name__)

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def load_model_and_tokenizer(
    model_path: str,
    method: QuantizationMethod,
    device: str = "auto",
) -> tuple[Any, Any]:
    """Load a model and tokenizer with the appropriate backend for the given method.

    Dispatches to bitsandbytes, AutoGPTQ, AutoAWQ, or llama_cpp based on method.
    Falls back to CPU for non-CUDA environments.

    Args:
        model_path: HuggingFace model ID or local directory path.
        method: Quantization method that determines how to load the model.
        device: Target device string ("auto", "cuda", "cpu", "mps").

    Returns:
        Tuple of (model, tokenizer). For GGUF models, tokenizer is None.

    Raises:
        ImportError: If required library for the given method is not installed.
        RuntimeError: If model loading fails.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    logger.info(f"Loading model: {model_path} (method={method})")

    if method in (QuantizationMethod.GGUF_Q2K, QuantizationMethod.GGUF_Q3KM,
                  QuantizationMethod.GGUF_Q4KM, QuantizationMethod.GGUF_Q5KM,
                  QuantizationMethod.GGUF_Q8, QuantizationMethod.GGUF_F16):
        return _load_gguf(model_path)

    if method == QuantizationMethod.GPTQ_4BIT:
        return _load_gptq(model_path, device)

    if method == QuantizationMethod.AWQ_4BIT:
        return _load_awq(model_path, device)

    if method in (QuantizationMethod.INT4_NF4, QuantizationMethod.INT4_FP4):
        return _load_bitsandbytes(model_path, bits=4, device=device)

    if method == QuantizationMethod.INT8:
        return _load_bitsandbytes(model_path, bits=8, device=device)

    # BF16 / FP16 — standard load
    dtype_map = {
        QuantizationMethod.BF16: "bfloat16",
        QuantizationMethod.FP16: "float16",
    }
    torch_dtype_str = dtype_map.get(method, "bfloat16")

    if _TORCH_AVAILABLE:
        import torch
        dtype = getattr(torch, torch_dtype_str, torch.float32)
        if not torch.cuda.is_available():
            dtype = torch.float32
    else:
        dtype = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device if _TORCH_AVAILABLE and _torch_cuda_ok(device) else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    logger.info(f"Loaded {model_path} at {torch_dtype_str}")
    return model, tokenizer


def _torch_cuda_ok(device: str) -> bool:
    """Return True if CUDA is available and device mapping is appropriate."""
    if not _TORCH_AVAILABLE:
        return False
    import torch
    return torch.cuda.is_available() and device in ("auto", "cuda")


def _load_bitsandbytes(model_path: str, bits: int, device: str) -> tuple[Any, Any]:
    """Load model with bitsandbytes quantization."""
    import torch
    from transformers import (  # type: ignore[import]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
    )
    return model, tokenizer


def _load_gptq(model_path: str, device: str) -> tuple[Any, Any]:
    """Load a pre-quantized GPTQ model."""
    try:
        from auto_gptq import AutoGPTQForCausalLM  # type: ignore[import]
        from transformers import AutoTokenizer  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "auto-gptq is required for GPTQ models. Install: pip install auto-gptq"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        device=device if device != "auto" else "cuda:0",
        use_safetensors=True,
        inject_fused_attention=False,
    )
    return model, tokenizer


def _load_awq(model_path: str, device: str) -> tuple[Any, Any]:
    """Load a pre-quantized AWQ model."""
    try:
        from awq import AutoAWQForCausalLM  # type: ignore[import]
        from transformers import AutoTokenizer  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "autoawq is required for AWQ models. Install: pip install autoawq"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True)
    return model, tokenizer


def _load_gguf(model_path: str) -> tuple[Any, None]:
    """Load a GGUF model via llama-cpp-python."""
    try:
        from llama_cpp import Llama  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required for GGUF models.\n"
            "CPU: pip install llama-cpp-python\n"
            "RTX 5090 CUDA: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
        ) from exc

    import torch
    n_gpu_layers = -1 if (torch.cuda.is_available() if _TORCH_AVAILABLE else False) else 0
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    return model, None
