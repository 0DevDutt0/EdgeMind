"""Tests for benchmark models serialization and deserialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgemind.models.benchmark_models import (
    BenchmarkResult,
    MemoryResult,
    ModelSummary,
    PerplexityResult,
    QuantizationMethod,
    QualityResult,
    SpeedResult,
)


def _make_benchmark_result(method: QuantizationMethod = QuantizationMethod.BF16) -> BenchmarkResult:
    """Build a complete BenchmarkResult for testing."""
    return BenchmarkResult(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quantization_method=method,
        model_size_gb=14.5,
        benchmarked_at="2026-06-14T10:00:00+00:00",
        device="cuda (NVIDIA RTX 5090)",
        perplexity=PerplexityResult(
            mean_perplexity=8.42,
            std_perplexity=0.38,
            min_perplexity=7.21,
            max_perplexity=9.85,
            num_samples=1000,
            dataset="wikitext2_subset",
            duration_seconds=142.3,
        ),
        speed=SpeedResult(
            mean_tps=178.3,
            std_tps=4.2,
            min_tps=168.1,
            max_tps=186.7,
            mean_ttft_ms=85.2,
            std_ttft_ms=6.1,
            num_runs=10,
            prompt_tokens=24,
            max_new_tokens=100,
            device="cuda",
            duration_seconds=68.4,
        ),
        memory=MemoryResult(
            model_vram_gb=14.2,
            peak_vram_gb=15.1,
            available_vram_gb=24.0,
            system_ram_gb=8.3,
            fits_on_device=True,
            device_name="NVIDIA RTX 5090",
        ),
        quality=QualityResult(
            mean_quality_score=8.9,
            std_quality_score=0.6,
            num_prompts=20,
            prompt_categories={"explanation": 9.1, "coding": 8.8},
            groq_comparison_score=8.9,
            quality_retention_pct=100.0,
            duration_seconds=184.2,
        ),
        compression_ratio=1.0,
        quality_retention_pct=100.0,
        overall_score=89.3,
        recommended_for=["RTX 4090", "RTX 5090"],
    )


class TestBenchmarkResultSerialization:
    """Test JSON round-trip for BenchmarkResult."""

    def test_to_dict_is_json_serializable(self) -> None:
        """BenchmarkResult.to_dict() must produce a JSON-serializable dict."""
        result = _make_benchmark_result()
        data = result.to_dict()
        json_str = json.dumps(data)
        assert len(json_str) > 100

    def test_to_json_creates_file(self, tmp_path: Path) -> None:
        """to_json() should create a file at the given path."""
        result = _make_benchmark_result()
        path = tmp_path / "results" / "test.json"
        result.to_json(str(path))
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["model_id"] == "Qwen/Qwen2.5-7B-Instruct"

    def test_from_json_roundtrip(self, tmp_path: Path) -> None:
        """from_json(to_json()) should reconstruct the original object."""
        original = _make_benchmark_result()
        path = tmp_path / "roundtrip.json"
        original.to_json(str(path))

        loaded = BenchmarkResult.from_json(str(path))

        assert loaded.model_id == original.model_id
        assert loaded.quantization_method == original.quantization_method
        assert loaded.model_size_gb == original.model_size_gb
        assert loaded.perplexity is not None
        assert abs(loaded.perplexity.mean_perplexity - 8.42) < 0.001
        assert loaded.speed is not None
        assert abs(loaded.speed.mean_tps - 178.3) < 0.001
        assert loaded.memory is not None
        assert loaded.memory.fits_on_device is True

    def test_from_dict_with_none_fields(self) -> None:
        """from_dict should handle None perplexity/speed/memory/quality."""
        data = {
            "model_id": "test/model",
            "quantization_method": "bf16",
            "model_size_gb": 7.0,
            "benchmarked_at": "2026-06-14T10:00:00",
            "device": "cpu",
            "perplexity": None,
            "speed": None,
            "memory": None,
            "quality": None,
            "compression_ratio": 1.0,
            "quality_retention_pct": None,
            "overall_score": None,
            "recommended_for": [],
        }
        result = BenchmarkResult.from_dict(data)
        assert result.model_id == "test/model"
        assert result.perplexity is None
        assert result.speed is None

    def test_all_quantization_methods_valid(self) -> None:
        """All QuantizationMethod values should round-trip through from_dict."""
        for method in QuantizationMethod:
            result = _make_benchmark_result(method)
            data = result.to_dict()
            loaded = BenchmarkResult.from_dict(data)
            assert loaded.quantization_method == method


class TestModelSummary:
    """Tests for ModelSummary serialization."""

    def test_to_json_creates_file(self, tmp_path: Path) -> None:
        """ModelSummary.to_json() should create a valid JSON file."""
        summary = ModelSummary(
            model_id="Qwen/Qwen2.5-7B-Instruct",
            baseline_perplexity=8.42,
            baseline_tps=178.3,
            baseline_size_gb=14.5,
            results=[_make_benchmark_result()],
            recommended_method="gptq_4bit",
            recommended_method_reason="Best quality/size tradeoff",
        )
        path = tmp_path / "summary.json"
        summary.to_json(str(path))
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
        assert data["recommended_method"] == "gptq_4bit"
        assert len(data["results"]) == 1
