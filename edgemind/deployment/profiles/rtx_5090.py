"""Hardware profile for NVIDIA RTX 5090 (Blackwell, sm_120, 24GB VRAM)."""

from edgemind.deployment.profiles.base_profile import HardwareProfile

RTX_5090_PROFILE = HardwareProfile(
    name="NVIDIA RTX 5090",
    vram_gb=24.0,
    ram_gb=32.0,
    cuda_compute="sm_120",
    architecture="Blackwell",
    recommended_inference="vllm",
    suitable_model_sizes=["1B-70B (with INT4)"],
    max_recommended_params_billions=32.0,
    quantization_recommendations={
        "1B-3B": {
            "method": "bfloat16",
            "expected_tps": "400-600",
            "vram_required_gb": 3.0,
            "notes": "Run at full BF16. Lightning fast. Entire model fits in VRAM with headroom.",
        },
        "7B": {
            "method": "bfloat16",
            "expected_tps": "150-200",
            "vram_required_gb": 14.0,
            "notes": "Full BF16 fits in 24GB. Recommended over quantized — no quality loss.",
        },
        "13B": {
            "method": "int8",
            "expected_tps": "80-110",
            "vram_required_gb": 13.0,
            "notes": (
                "BF16 is borderline at ~26GB. INT8 fits at ~13GB with ~97% quality retention."
            ),
        },
        "32B": {
            "method": "int4_nf4",
            "expected_tps": "40-60",
            "vram_required_gb": 18.0,
            "notes": "INT4 fits comfortably in 24GB. 90-93% quality retention.",
        },
        "70B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "10-20",
            "vram_required_gb": 40.0,
            "notes": (
                "Exceeds 24GB VRAM. Use llama.cpp with n_gpu_layers=-1 "
                "to offload as many layers as fit. Remaining layers run on CPU."
            ),
        },
    },
    setup_notes="""RTX 5090 (Blackwell, sm_120) requires PyTorch nightly with CUDA 13:

  pip install --pre torch torchvision torchaudio \\
      --index-url https://download.pytorch.org/whl/nightly/cu130

vLLM (recommended for production serving):
  pip install vllm  # v0.17+ supports sm_120 natively

Ollama (easiest for local chatbot use):
  Download from https://ollama.com — supports RTX 5090 out of the box.

Verify GPU detection:
  python -c "import torch; print(torch.cuda.get_device_capability(0))"
  # Expected: (12, 0)
""",
)
