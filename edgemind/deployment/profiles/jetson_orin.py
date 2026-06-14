"""Hardware profile for NVIDIA Jetson AGX Orin (sm_87, 32GB unified memory)."""

from edgemind.deployment.profiles.base_profile import HardwareProfile

JETSON_AGX_ORIN_PROFILE = HardwareProfile(
    name="NVIDIA Jetson AGX Orin",
    vram_gb=32.0,
    ram_gb=32.0,
    cuda_compute="sm_87",
    architecture="Ampere (embedded)",
    recommended_inference="llama_cpp",
    suitable_model_sizes=["1B-13B (INT4)", "1B-7B (INT8)"],
    max_recommended_params_billions=13.0,
    quantization_recommendations={
        "1B-3B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "50-90",
            "vram_required_gb": 2.0,
            "notes": (
                "Q4_K_M fits easily. Excellent for real-time edge inference. "
                "Consider Q5_K_M for slightly better quality."
            ),
        },
        "7B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "20-40",
            "vram_required_gb": 4.5,
            "notes": (
                "Q4_K_M fits well within 32GB unified memory. "
                "Good balance of quality and speed for production edge use."
            ),
        },
        "13B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "10-20",
            "vram_required_gb": 8.0,
            "notes": "Q4_K_M for 13B requires ~8GB — fits with headroom on Orin 32GB.",
        },
        "32B": {
            "method": "gguf_q2_k",
            "expected_tps": "3-7",
            "vram_required_gb": 11.0,
            "notes": "Q2_K marginally fits but quality is poor. Not recommended.",
        },
    },
    setup_notes="""Jetson AGX Orin — JetPack 6.x setup:

Install llama.cpp with CUDA support (JetPack provides CUDA 11.4+):
  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python

Or use the prebuilt llama.cpp for Jetson:
  https://github.com/ggerganov/llama.cpp/releases

For Ollama on Jetson:
  curl -fsSL https://ollama.com/install.sh | sh

JetPack 6 ships with CUDA 12.x. Set compute to sm_87:
  CMAKE_CUDA_ARCHITECTURES=87

Power mode for max performance:
  sudo nvpmodel -m 0  # 60W MAXN mode
  sudo jetson_clocks
""",
)
