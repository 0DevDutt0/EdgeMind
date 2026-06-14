"""Hardware profile for Apple M3 Pro (18-36GB unified memory, MPS backend)."""

from edgemind.deployment.profiles.base_profile import HardwareProfile

MAC_M3_PRO_PROFILE = HardwareProfile(
    name="Apple M3 Pro (18-36GB)",
    vram_gb=18.0,
    ram_gb=36.0,
    cuda_compute=None,
    architecture="Apple Silicon M3 (MPS)",
    recommended_inference="llama_cpp",
    suitable_model_sizes=["1B-32B (Q4-Q5 GGUF)"],
    max_recommended_params_billions=32.0,
    quantization_recommendations={
        "1B-3B": {
            "method": "bfloat16",
            "expected_tps": "60-120",
            "vram_required_gb": 3.0,
            "notes": (
                "Run at full BF16 via MPS. M3 Pro has fast unified bandwidth. "
                "No quantization needed for 1-3B models."
            ),
        },
        "7B": {
            "method": "gguf_q5_k_m",
            "expected_tps": "30-55",
            "vram_required_gb": 5.5,
            "notes": (
                "Q5_K_M gives excellent quality (~94% retention). "
                "BF16 (14GB) also fits on 18GB and is fastest."
            ),
        },
        "13B": {
            "method": "gguf_q5_k_m",
            "expected_tps": "15-28",
            "vram_required_gb": 9.0,
            "notes": "Q5_K_M fits well on 18GB. Excellent quality for edge deployment.",
        },
        "32B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "6-12",
            "vram_required_gb": 20.0,
            "notes": (
                "Q4_K_M for 32B requires ~20GB — fits on 36GB M3 Pro Max. "
                "18GB model needs Q3_K_M (fits at ~15GB, quality reduced)."
            ),
        },
        "70B": {
            "method": "gguf_q2_k",
            "expected_tps": "2-5",
            "vram_required_gb": 25.0,
            "notes": (
                "70B Q2_K requires ~25GB — only fits on 36GB M3 Pro Max. "
                "Quality is significantly degraded. M3 Ultra (96-192GB) is preferred for 70B."
            ),
        },
    },
    setup_notes="""Apple M3 Pro Mac — Metal GPU acceleration:

llama.cpp with Metal (highest performance):
  CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python

Ollama (recommended for ease of use):
  brew install ollama
  ollama serve
  ollama run llama3.2:latest  # 3B, excellent quality

HuggingFace transformers via MPS:
  import torch
  model = AutoModelForCausalLM.from_pretrained(
      model_id, torch_dtype=torch.float16
  ).to("mps")

Performance notes:
  - M3 Pro has 150GB/s memory bandwidth (vs M1's 68GB/s)
  - 6 performance + 6 efficiency CPU cores
  - 18-core GPU for Metal/MPS inference
  - M3 Max: 30-40 core GPU, even faster inference

For the best 32B experience, use M3 Max (36-128GB).
""",
)
