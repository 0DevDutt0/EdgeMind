"""Hardware profile for NVIDIA Jetson Nano (sm_53, 4GB LPDDR4)."""

from edgemind.deployment.profiles.base_profile import HardwareProfile

JETSON_NANO_PROFILE = HardwareProfile(
    name="NVIDIA Jetson Nano (4GB)",
    vram_gb=4.0,
    ram_gb=4.0,
    cuda_compute="sm_53",
    architecture="Maxwell (embedded)",
    recommended_inference="llama_cpp",
    suitable_model_sizes=["1B-3B (INT4 GGUF only)"],
    max_recommended_params_billions=3.0,
    quantization_recommendations={
        "1B-3B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "5-15",
            "vram_required_gb": 2.0,
            "notes": (
                "Q4_K_M is the maximum viable option on Nano 4GB. "
                "Use Q3_K_M for very tight memory situations. "
                "Models above 3B will swap to disk and become unusably slow."
            ),
        },
        "7B": {
            "method": "gguf_q2_k",
            "expected_tps": "1-4",
            "vram_required_gb": 3.0,
            "notes": (
                "Q2_K barely fits in 4GB but quality is severely degraded. "
                "Not recommended for production. Upgrade to Orin for 7B models."
            ),
        },
    },
    setup_notes="""Jetson Nano — JetPack 4.6.x (CUDA 10.2):

Note: Jetson Nano uses the older Maxwell architecture (sm_53).
Many modern PyTorch features are not available. llama.cpp is recommended.

Install llama.cpp without CUDA (Maxwell support is limited):
  pip install llama-cpp-python

For CUDA-accelerated inference on Nano (experimental):
  CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=53' \\
      pip install llama-cpp-python

Model recommendations:
  - Phi-2 (2.7B) Q4_K_M: ~2GB, ~8 tok/s
  - TinyLlama (1.1B) Q4_K_M: ~0.7GB, ~15 tok/s
  - Qwen2.5-1.5B Q4_K_M: ~1GB, ~12 tok/s

Power considerations:
  - Max draw: 10W (2-lane) or 5W (fan) mode
  - Use sudo nvpmodel -m 0 for max performance
""",
)
