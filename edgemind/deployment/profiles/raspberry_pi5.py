"""Hardware profile for Raspberry Pi 5 (8GB RAM, CPU-only inference)."""

from edgemind.deployment.profiles.base_profile import HardwareProfile

RASPBERRY_PI5_PROFILE = HardwareProfile(
    name="Raspberry Pi 5 (8GB)",
    vram_gb=0.0,
    ram_gb=8.0,
    cuda_compute=None,
    architecture="ARM Cortex-A76 (CPU only)",
    recommended_inference="llama_cpp",
    suitable_model_sizes=["1B-3B (Q4 GGUF, CPU only)"],
    max_recommended_params_billions=3.0,
    quantization_recommendations={
        "1B-3B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "2-8",
            "vram_required_gb": 0.0,
            "notes": (
                "CPU-only inference via llama.cpp. Q4_K_M provides best quality/speed balance. "
                "Pi 5 has significantly better CPU than Pi 4 (~2-3x faster inference). "
                "Phi-2 (2.7B) or TinyLlama (1.1B) are optimal choices."
            ),
        },
        "7B": {
            "method": "gguf_q2_k",
            "expected_tps": "0.5-2",
            "vram_required_gb": 0.0,
            "notes": (
                "Q2_K 7B requires ~2.8GB RAM but inference is extremely slow (<2 tok/s). "
                "Only suitable for offline batch tasks. Pi 5 is not designed for 7B models."
            ),
        },
    },
    setup_notes="""Raspberry Pi 5 (64-bit Raspberry Pi OS):

Install llama.cpp with CPU optimizations:
  # Enable NEON and SVE optimizations for Cortex-A76
  CMAKE_ARGS='-DGGML_NATIVE=on' pip install llama-cpp-python

Or build from source with Pi 5 optimizations:
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && make -j4 LLAMA_NATIVE=1

Recommended models for Pi 5:
  - TinyLlama-1.1B Q4_K_M: ~0.7GB, ~8 tok/s
  - Phi-2 2.7B Q4_K_M: ~1.7GB, ~4 tok/s
  - Qwen2.5-1.5B Q4_K_M: ~1.0GB, ~6 tok/s

Use Ollama for easiest deployment:
  curl -fsSL https://ollama.com/install.sh | sh
  ollama run tinyllama

Memory tip: Disable the desktop (use Lite OS) to free ~1GB RAM.
""",
)
