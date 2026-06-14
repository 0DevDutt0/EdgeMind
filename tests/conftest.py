"""Shared pytest fixtures for EdgeMind tests. All fixtures work without a GPU."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session")
def tiny_model():
    """Load GPT-2 (124M) for CPU-only benchmark testing.

    Session-scoped so the model is loaded only once per test session.
    Uses FP32 to work on any hardware including CPU-only CI environments.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Set pad token to eos token for generation
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.eval()
        return model, tokenizer
    except Exception as exc:
        pytest.skip(f"Could not load gpt2 model (requires internet): {exc}")


@pytest.fixture
def mock_gpu_info() -> dict:
    """Return a fake RTX 5090 GPU info dict for testing gpu_utils logic."""
    return {
        "device_name": "NVIDIA RTX 5090",
        "compute_capability": "sm_120",
        "vram_gb": 24.0,
        "cuda_version": "13.0",
        "pytorch_version": "2.12.0+cu130",
        "is_blackwell": True,
        "full_support": True,
        "backend": "cuda",
        "system_ram_gb": 32.0,
        "torch_available": True,
    }


@pytest.fixture
def mock_cpu_gpu_info() -> dict:
    """Return a CPU-only GPU info dict."""
    return {
        "device_name": "CPU",
        "compute_capability": None,
        "vram_gb": 0.0,
        "cuda_version": None,
        "pytorch_version": "2.12.0",
        "is_blackwell": False,
        "full_support": False,
        "backend": "cpu",
        "system_ram_gb": 16.0,
        "torch_available": True,
    }


@pytest.fixture
def sample_wikitext() -> list[str]:
    """Return 50 sample sentences for perplexity testing."""
    return [
        "The transformer architecture uses self-attention mechanisms to process sequences of tokens efficiently.",
        "Machine learning models can be compressed through quantization, pruning, and knowledge distillation.",
        "Neural networks learn hierarchical representations of data through multiple layers of computation.",
        "The backpropagation algorithm computes gradients by applying the chain rule of calculus backwards through layers.",
        "Attention mechanisms allow models to focus on relevant parts of the input when generating each output token.",
        "Large language models are trained on massive datasets of text from the internet and books.",
        "Quantization reduces model precision from 32-bit floats to 8-bit or 4-bit integers for efficient inference.",
        "The perplexity metric measures how well a language model predicts a held-out test set of text.",
        "Edge AI refers to artificial intelligence computation performed locally on edge devices rather than in the cloud.",
        "The RTX 5090 graphics card features 24GB of GDDR7 VRAM and the new Blackwell architecture.",
        "Python is the dominant programming language for machine learning research and production deployment.",
        "Gradient descent optimizes neural network weights by following the negative gradient of the loss function.",
        "Transfer learning enables models trained on large datasets to be fine-tuned efficiently on smaller datasets.",
        "The GGUF file format is used by llama.cpp for efficient cross-platform language model inference.",
        "Regularization techniques such as dropout prevent overfitting by adding noise during training.",
        "Batch normalization normalizes layer inputs to accelerate training and improve stability of deep networks.",
        "The embedding layer maps discrete tokens to continuous vector representations in a learned semantic space.",
        "Convolutional neural networks are particularly effective for image processing and computer vision tasks.",
        "Recurrent neural networks process sequences step by step, maintaining a hidden state across time steps.",
        "Model parallelism distributes a large model across multiple GPUs when it does not fit on a single device.",
        "The softmax function converts a vector of raw scores into a probability distribution over classes.",
        "Residual connections allow gradients to flow more easily through very deep neural networks during training.",
        "The encoder-decoder architecture is commonly used for sequence-to-sequence tasks like machine translation.",
        "Sparse attention reduces the computational complexity of the attention operation for very long sequences.",
        "Flash attention computes the attention operation in tiles to minimize memory bandwidth requirements.",
        "Knowledge distillation transfers knowledge from a large teacher model to a compact student model.",
        "The CUDA programming model enables massively parallel computation on NVIDIA graphics processing units.",
        "Tokenization converts raw text into sequences of integer token IDs for processing by language models.",
        "The validation loss is monitored during training to detect overfitting and determine when to stop training.",
        "Zero-shot learning allows models to perform tasks they were not explicitly trained on using natural language.",
        "Reinforcement learning from human feedback is used to align language model outputs with human preferences.",
        "The context window of a language model determines the maximum number of tokens it can process at once.",
        "Low-rank adaptation enables efficient fine-tuning of large models by updating only small additional matrices.",
        "The inference latency of a language model depends on model size, hardware, and the length of the output.",
        "Mixture of experts models activate only a subset of their parameters for each input, reducing compute cost.",
        "Continuous batching allows inference servers to add new requests to running batches without waiting.",
        "The KV cache stores computed key and value matrices to avoid recomputation during autoregressive generation.",
        "Vector databases enable efficient semantic search over large collections of embedding vectors.",
        "Retrieval-augmented generation combines a language model with a document retrieval system for factual accuracy.",
        "Model evaluation benchmarks measure different aspects of language model capability and alignment.",
        "The temperature parameter controls the randomness of the probability distribution during text generation.",
        "Top-p sampling selects tokens from the smallest set whose cumulative probability exceeds the threshold p.",
        "System prompts provide instructions and context to language models before the user conversation begins.",
        "Fine-tuning adapts a pre-trained foundation model to a specific task using a smaller labeled dataset.",
        "Instruction tuning trains models to follow natural language instructions across diverse tasks.",
        "Constitutional AI is a method for training models to be helpful, harmless, and honest simultaneously.",
        "The scaling laws for neural language models describe how performance improves with model and data scale.",
        "Compute-optimal training balances the allocation of a fixed compute budget between model size and data.",
        "Post-training quantization converts the weights of a fully trained model to lower precision without retraining.",
        "Quantization-aware training simulates quantization effects during training for better low-precision performance.",
    ]


@pytest.fixture
def sample_eval_prompts() -> list[dict]:
    """Return 5 sample quality evaluation prompts for testing."""
    return [
        {"id": 1, "category": "explanation",
         "prompt": "Explain how attention mechanisms work:"},
        {"id": 2, "category": "coding",
         "prompt": "Write a Python function to compute the factorial of n:"},
        {"id": 3, "category": "reasoning",
         "prompt": "If 2 + 2 = 4 and 4 + 4 = 8, what is 8 + 8?"},
        {"id": 4, "category": "factual",
         "prompt": "What is the capital of France?"},
        {"id": 5, "category": "creative",
         "prompt": "Write one sentence about a robot:"},
    ]


@pytest.fixture
def mock_groq_client() -> MagicMock:
    """Return a mock Groq client that returns score=8.5 for any judge call."""
    client = MagicMock()
    client.complete.return_value = "8"
    client.is_configured = True
    client.estimate_tokens_per_second.return_value = 520.0
    return client
