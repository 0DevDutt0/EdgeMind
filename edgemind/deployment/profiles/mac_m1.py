"""Hardware profile for Apple M1 (8GB or 16GB unified memory, MPS backend)."""

from edgemind.deployment.profiles.base_profile import HardwareProfile

MAC_M1_PROFILE = HardwareProfile(
    name="Apple M1 (8-16GB)",
    vram_gb=8.0,
    ram_gb=8.0,
    cuda_compute=None,
    architecture="Apple Silicon M1 (MPS)",
    recommended_inference="llama_cpp",
    suitable_model_sizes=["1B-7B (Q4 GGUF)"],
    max_recommended_params_billions=7.0,
    quantization_recommendations={
        "1B-3B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "30-60",
            "vram_required_gb": 2.0,
            "notes": (
                "Excellent fit. Metal GPU acceleration via llama.cpp. "
                "TinyLlama and Phi-2 run extremely well on M1."
            ),
        },
        "7B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "15-30",
            "vram_required_gb": 4.5,
            "notes": (
                "Q4_K_M fits on 8GB M1. For 16GB M1, Q5_K_M gives better quality. "
                "Mistral 7B and Llama 3.2 7B both work well."
            ),
        },
        "13B": {
            "method": "gguf_q4_k_m",
            "expected_tps": "8-15",
            "vram_required_gb": 8.0,
            "notes": (
                "Requires 16GB M1. Q4_K_M is tight at ~8GB — monitor memory pressure. "
                "Q3_K_M is safer on 8GB M1 (tight fit)."
            ),
        },
    },
    setup_notes="""Apple M1 Mac — Metal GPU acceleration:

llama.cpp with Metal (recommended, fastest):
  CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python

Or via Ollama (easiest):
  brew install ollama
  ollama serve &
  ollama pull mistral

HuggingFace transformers with MPS backend:
  pip install torch  # MPS support included in standard torch
  model = AutoModelForCausalLM.from_pretrained(
      model_id, torch_dtype=torch.float16
  ).to("mps")

Memory tip: Close other apps — unified memory is shared with the OS.
  Streaming performance is excellent on M1 due to high memory bandwidth.
""",
)
