"""Integration tests for the full benchmark pipeline using GPT-2 on CPU."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from edgemind.models.benchmark_models import BenchmarkResult, QuantizationMethod


class TestBenchmarkRunner:
    """Integration tests for BenchmarkRunner.run_all()."""

    def test_results_saved_to_json(
        self, tiny_model: tuple, tmp_path: Path, sample_wikitext: list[str]
    ) -> None:
        """Benchmark runner must save JSON results before returning."""
        from edgemind.benchmarks.runner import BenchmarkRunner

        model, tokenizer = tiny_model

        # Write eval data
        wiki_path = tmp_path / "wiki.txt"
        wiki_path.write_text("\n".join(sample_wikitext[:20]), encoding="utf-8")
        results_dir = tmp_path / "results"

        with patch("edgemind.benchmarks.runner.get_config") as mock_config:
            cfg = MagicMock()
            cfg.results_dir = results_dir
            cfg.benchmark_num_runs = 2
            cfg.benchmark_warmup_runs = 1
            cfg.benchmark_max_new_tokens = 20
            cfg.perplexity_batch_size = 1
            cfg.groq_api_key = ""
            cfg.ensure_dirs = lambda: results_dir.mkdir(parents=True, exist_ok=True)
            mock_config.return_value = cfg

            with patch("edgemind.benchmarks.runner.load_model_and_tokenizer") as mock_load:
                mock_load.return_value = (model, tokenizer)

                with patch("edgemind.benchmarks.perplexity.PerplexityBenchmark._load_wikitext2_subset") as mock_wiki:
                    mock_wiki.return_value = sample_wikitext[:10]

                    runner = BenchmarkRunner()
                    result = runner.run_all(
                        model_path="gpt2",
                        quantization_method=QuantizationMethod.BF16,
                        tests=["memory", "perplexity"],
                        device="cpu",
                    )

        # Verify JSON was saved
        json_files = list(results_dir.rglob("*.json"))
        assert len(json_files) >= 1, f"Expected JSON output in {results_dir}"

        # Verify the result is a valid BenchmarkResult
        assert isinstance(result, BenchmarkResult)
        assert result.model_id == "gpt2"
        assert result.quantization_method == QuantizationMethod.BF16

    def test_json_schema_complete(self, tmp_path: Path) -> None:
        """Saved JSON must contain all required top-level fields."""
        # Use a sample result JSON to validate the schema
        sample_path = Path("sample_results/qwen2.5-7b/bf16_benchmark.json")
        if not sample_path.exists():
            pytest.skip("Sample results not available")

        result = BenchmarkResult.from_json(str(sample_path))

        required_fields = [
            "model_id", "quantization_method", "model_size_gb",
            "benchmarked_at", "device", "compression_ratio",
        ]
        data = result.to_dict()
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_benchmark_continues_on_test_failure(
        self, tiny_model: tuple, tmp_path: Path, sample_wikitext: list[str]
    ) -> None:
        """A failing test should be logged and skipped, not crash the runner."""
        from edgemind.benchmarks.runner import BenchmarkRunner

        model, tokenizer = tiny_model
        results_dir = tmp_path / "results"

        with patch("edgemind.benchmarks.runner.get_config") as mock_config:
            cfg = MagicMock()
            cfg.results_dir = results_dir
            cfg.benchmark_num_runs = 2
            cfg.benchmark_warmup_runs = 1
            cfg.benchmark_max_new_tokens = 10
            cfg.perplexity_batch_size = 1
            cfg.groq_api_key = ""
            cfg.ensure_dirs = lambda: results_dir.mkdir(parents=True, exist_ok=True)
            mock_config.return_value = cfg

            with patch("edgemind.benchmarks.runner.load_model_and_tokenizer") as mock_load:
                mock_load.return_value = (model, tokenizer)

                # Make perplexity raise to test error handling
                with patch("edgemind.benchmarks.runner.PerplexityBenchmark") as mock_ppl:
                    mock_ppl.return_value.measure.side_effect = RuntimeError("Simulated failure")

                    runner = BenchmarkRunner()
                    result = runner.run_all(
                        model_path="gpt2",
                        quantization_method=QuantizationMethod.BF16,
                        tests=["perplexity"],
                        device="cpu",
                    )

        # Runner should complete and return result even with failed perplexity
        assert isinstance(result, BenchmarkResult)
        assert result.perplexity is None  # Failed test returns None, not crash

    def test_full_pipeline_memory_and_speed(
        self, tiny_model: tuple, tmp_path: Path
    ) -> None:
        """Memory profiler and speed benchmark should complete on CPU with gpt2."""
        from edgemind.benchmarks.memory_profiler import MemoryProfiler
        from edgemind.benchmarks.inference_speed import InferenceSpeedBenchmark

        model, tokenizer = tiny_model

        # Memory profiler
        profiler = MemoryProfiler()
        mem_result = profiler.profile(model, tokenizer, device="cpu")
        assert mem_result.system_ram_gb >= 0
        assert mem_result.device_name in ("CPU", "Apple Silicon (MPS)")

        # Speed benchmark (minimal)
        speed_bench = InferenceSpeedBenchmark()
        speed_result = speed_bench.measure(
            model, tokenizer,
            num_runs=2,
            warmup_runs=1,
            max_new_tokens=10,
        )
        assert speed_result.mean_tps > 0
        assert speed_result.num_runs == 2
        assert speed_result.duration_seconds > 0
