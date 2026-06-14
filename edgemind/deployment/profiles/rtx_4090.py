"""Hardware profile for NVIDIA RTX 4090 (Ada Lovelace, sm_89, 24GB VRAM)."""

from edgemind.deployment.profiles.base_profile import HardwareProfile

RTX_4090_PROFILE = HardwareProfile(
    name="NVIDIA RTX 4090",
    vram_gb=24.0,
    ram_gb=32.0,
    cuda_compute="sm_89",
    architecture="Ada Lovelace",
    recommended_inference="vllm",
    suitable_model_sizes=["1B-32B (with INT4)"],
    max_recommended_params_billions=32.0,
    quantization_recommendations={
        "1B-3B": {
            "method": "bfloat16",
            "expected_tps": "300-450",
            "vram_required_gb": 3.0,
            "notes": "Full BF16. Excellent speed. No quantization needed.",
        },
        "7B": {
            "method": "bfloat16",
            "expected_tps": "100-140",
            "vram_required_gb": 14.0,
            "notes": "Full BF16 fits in 24GB. Use BF16 for maximum quality.",
        },
        "13B": {
            "method": "int8",
            "expected_tps": "55-80",
            "vram_required_gb": 13.0,
            "notes": "INT8 provides safe fit with ~97% quality retention.",
        },
        "32B": {
            "method": "int4_nf4",
            "expected_tps": "25-40",
            "vram_required_gb": 18.0,
            "notes": "INT4 NF4 fits in 24GB. 90-93% quality retention. Best option.",
        },
        "70B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "6-12",
            "vram_required_gb": 40.0,
            "notes": (
                "Requires CPU offloading. Use llama.cpp with partial GPU layers. "
                "Speed will be limited by PCIe bandwidth."
            ),
        },
    },
    setup_notes="""RTX 4090 (Ada Lovelace, sm_89) — standard PyTorch works out of the box:

  pip install torch torchvision torchaudio \\
      --index-url https://download.pytorch.org/whl/cu121

vLLM for high-throughput serving:
  pip install vllm

Ollama:
  Download from https://ollama.com — full RTX 4090 support.
""",
)
