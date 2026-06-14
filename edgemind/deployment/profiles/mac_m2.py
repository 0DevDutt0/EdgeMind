"""Hardware profile for Apple M2 (8-24GB unified memory, MPS backend)."""

from edgemind.deployment.profiles.base_profile import HardwareProfile

MAC_M2_PROFILE = HardwareProfile(
    name="Apple M2 (8-24GB)",
    vram_gb=16.0,
    ram_gb=16.0,
    cuda_compute=None,
    architecture="Apple Silicon M2 (MPS)",
    recommended_inference="llama_cpp",
    suitable_model_sizes=["1B-13B (Q4-Q5 GGUF)"],
    max_recommended_params_billions=13.0,
    quantization_recommendations={
        "1B-3B": {
            "method": "gguf_q5_k_m",
            "expected_tps": "40-80",
            "vram_required_gb": 2.5,
            "notes": "Can afford higher quality Q5_K_M. Excellent speed on M2 Neural Engine.",
        },
        "7B": {
            "method": "gguf_q5_k_m",
            "expected_tps": "20-40",
            "vram_required_gb": 5.5,
            "notes": (
                "Q5_K_M on 16GB M2 provides excellent quality (~94% retention). "
                "8GB M2 should use Q4_K_M instead."
            ),
        },
        "13B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "10-20",
            "vram_required_gb": 8.0,
            "notes": (
                "Q4_K_M for 13B fits on 16GB M2. "
                "24GB M2 Max can run Q5_K_M at better quality."
            ),
        },
        "32B": {
            "method": "gguf_q2_k",
            "expected_tps": "3-6",
            "vram_required_gb": 11.0,
            "notes": (
                "Only possible on 24GB M2 Max with Q2_K (quality severely degraded). "
                "Consider M3 Pro/Max for 32B models."
            ),
        },
    },
    setup_notes="""Apple M2 Mac — Metal GPU acceleration:

llama.cpp with Metal (recommended):
  CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python

Ollama (easiest, full Metal support):
  brew install ollama
  ollama serve

For M2 Pro/Max/Ultra with more RAM:
  - M2 Pro (16-32GB): Can run 13B at Q5_K_M
  - M2 Max (32-96GB): Can run 32B at Q4_K_M
  - M2 Ultra (64-192GB): Can run 70B at Q4_K_M

Performance tip: Use --n-gpu-layers -1 in llama.cpp to offload all layers to GPU.
""",
)
