"""Tokens-per-second and time-to-first-token benchmarking."""

from __future__ import annotations

import time
from typing import Any

from tqdm import tqdm

from edgemind.core.logging import get_logger
from edgemind.models.benchmark_models import SpeedResult

logger = get_logger(__name__)

TEST_PROMPTS: list[str] = [
    "Explain the concept of quantum entanglement in simple terms:",
    "Write a Python function that sorts a list of dictionaries by a key:",
    "What are the main differences between supervised and unsupervised learning?",
    "Describe the history of artificial intelligence in three paragraphs:",
    "How does gradient descent work in neural network training?",
]


class InferenceSpeedBenchmark:
    """Measure tokens-per-second throughput and time-to-first-token latency.

    Two metrics serve different use cases:
    - TTFT (time-to-first-token): latency — critical for voice and interactive apps.
    - TPS (tokens per second): throughput — critical for batch processing and APIs.
    """

    def measure(
        self,
        model: Any,
        tokenizer: Any,
        num_runs: int = 10,
        warmup_runs: int = 3,
        max_new_tokens: int = 100,
    ) -> SpeedResult:
        """Benchmark inference speed with warmup runs to ensure stable measurements.

        Selects prompts round-robin across TEST_PROMPTS. After warmup, records
        TPS and TTFT for each run and returns summary statistics.

        Args:
            model: Loaded model (HF transformer or llama_cpp.Llama).
            tokenizer: HuggingFace tokenizer (None for GGUF models).
            num_runs: Number of recorded benchmark runs.
            warmup_runs: Number of unrecorded warmup runs.
            max_new_tokens: Output tokens to generate per run.

        Returns:
            SpeedResult with mean/std/min/max TPS, TTFT statistics, and metadata.
        """
        is_gguf = tokenizer is None or _is_llama_cpp(model)
        if is_gguf:
            return self._measure_gguf(model, num_runs, warmup_runs, max_new_tokens)
        return self._measure_hf(model, tokenizer, num_runs, warmup_runs, max_new_tokens)

    def _measure_hf(
        self,
        model: Any,
        tokenizer: Any,
        num_runs: int,
        warmup_runs: int,
        max_new_tokens: int,
    ) -> SpeedResult:
        """Speed measurement for HuggingFace transformer models."""
        import torch

        device = next(model.parameters()).device
        device_str = str(device).split(":")[0]

        # Determine prompt token count from first prompt
        sample_enc = tokenizer(TEST_PROMPTS[0], return_tensors="pt")
        prompt_tokens = sample_enc["input_ids"].shape[1]

        total_runs = warmup_runs + num_runs
        tps_list: list[float] = []
        ttft_list: list[float] = []

        start_total = time.perf_counter()

        for i in tqdm(range(total_runs), desc=f"Benchmarking speed (0/{num_runs})"):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = enc["input_ids"].shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            t_start = time.perf_counter()

            # TTFT approximation: time for first token
            with torch.no_grad():
                t_first = None
                outputs = None
                streamer = _FirstTokenTimer(lambda: None)
                try:
                    outputs = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        streamer=streamer if i >= warmup_runs else None,
                    )
                    t_first = streamer.first_token_time
                except Exception:
                    try:
                        outputs = model.generate(
                            **enc,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    except Exception as gen_exc:
                        logger.warning(f"generate() failed on run {i}: {gen_exc}")

            t_end = time.perf_counter()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if i < warmup_runs:
                continue

            if outputs is None:
                continue

            output_len = outputs.shape[1] - input_len
            elapsed = t_end - t_start
            tps = output_len / elapsed if elapsed > 0 else 0.0
            ttft_ms = (t_first - t_start) * 1000 if t_first else (elapsed / output_len * 1000)

            tps_list.append(tps)
            ttft_list.append(ttft_ms)

        import numpy as np
        tps_arr = np.array(tps_list)
        ttft_arr = np.array(ttft_list)
        duration = time.perf_counter() - start_total

        return SpeedResult(
            mean_tps=round(float(tps_arr.mean()), 2),
            std_tps=round(float(tps_arr.std()), 2),
            min_tps=round(float(tps_arr.min()), 2),
            max_tps=round(float(tps_arr.max()), 2),
            mean_ttft_ms=round(float(ttft_arr.mean()), 1),
            std_ttft_ms=round(float(ttft_arr.std()), 1),
            num_runs=num_runs,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            device=device_str,
            duration_seconds=round(duration, 2),
        )

    def _measure_gguf(
        self,
        model: Any,
        num_runs: int,
        warmup_runs: int,
        max_new_tokens: int,
    ) -> SpeedResult:
        """Speed measurement for llama_cpp (GGUF) models."""
        total_runs = warmup_runs + num_runs
        tps_list: list[float] = []
        ttft_list: list[float] = []
        prompt_tokens = 0
        start_total = time.perf_counter()

        for i in tqdm(range(total_runs), desc=f"Benchmarking speed (0/{num_runs})"):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            t_start = time.perf_counter()
            result = model.create_completion(
                prompt,
                max_tokens=max_new_tokens,
                stream=False,
                temperature=0.0,
            )
            t_end = time.perf_counter()

            if i < warmup_runs:
                if prompt_tokens == 0:
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 10)
                continue

            elapsed = t_end - t_start
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", max_new_tokens)
            tps = completion_tokens / elapsed if elapsed > 0 else 0.0
            # GGUF timing approximation: first token ≈ prefill time
            ttft_ms = (elapsed / max(completion_tokens, 1)) * 1000

            tps_list.append(tps)
            ttft_list.append(ttft_ms)

        import numpy as np
        tps_arr = np.array(tps_list) if tps_list else np.array([0.0])
        ttft_arr = np.array(ttft_list) if ttft_list else np.array([0.0])
        duration = time.perf_counter() - start_total

        return SpeedResult(
            mean_tps=round(float(tps_arr.mean()), 2),
            std_tps=round(float(tps_arr.std()), 2),
            min_tps=round(float(tps_arr.min()), 2),
            max_tps=round(float(tps_arr.max()), 2),
            mean_ttft_ms=round(float(ttft_arr.mean()), 1),
            std_ttft_ms=round(float(ttft_arr.std()), 1),
            num_runs=num_runs,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            device="cpu",
            duration_seconds=round(duration, 2),
        )


class _FirstTokenTimer:
    """Minimal streamer that records when the first token arrives."""

    def __init__(self, put_fn: Any) -> None:
        self.first_token_time: float | None = None
        self._put = put_fn

    def put(self, value: Any) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self._put(value)

    def end(self) -> None:
        pass


def _is_llama_cpp(model: Any) -> bool:
    """Detect if model is a llama_cpp.Llama instance."""
    return type(model).__name__ == "Llama" and hasattr(model, "create_completion")
