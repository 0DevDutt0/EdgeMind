"""Unit tests for perplexity measurement benchmark."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from edgemind.benchmarks.perplexity import PerplexityBenchmark, _is_llama_cpp
from edgemind.models.benchmark_models import PerplexityResult


class TestPerplexityBenchmark:
    """Tests for PerplexityBenchmark.measure()."""

    def test_perplexity_with_tiny_model(
        self, tiny_model: tuple, sample_wikitext: list[str], tmp_path
    ) -> None:
        """Verify perplexity measurement works with real GPT-2 on CPU."""
        model, tokenizer = tiny_model
        bench = PerplexityBenchmark()

        # Write sample sentences to temp file
        dataset_path = tmp_path / "test_wiki.txt"
        dataset_path.write_text("\n".join(sample_wikitext), encoding="utf-8")

        result = bench.measure(
            model, tokenizer,
            dataset_path=str(dataset_path),
            batch_size=2,
            max_samples=10,
        )

        assert isinstance(result, PerplexityResult)
        assert result.mean_perplexity > 0
        assert result.num_samples > 0
        assert result.dataset == "wikitext2_subset"

    def test_perplexity_stats_computed(
        self, tiny_model: tuple, sample_wikitext: list[str], tmp_path
    ) -> None:
        """Verify all statistical fields are computed in the result."""
        model, tokenizer = tiny_model
        bench = PerplexityBenchmark()

        dataset_path = tmp_path / "test_wiki.txt"
        dataset_path.write_text("\n".join(sample_wikitext[:15]), encoding="utf-8")

        result = bench.measure(
            model, tokenizer,
            dataset_path=str(dataset_path),
            batch_size=1,
            max_samples=10,
        )

        assert result.mean_perplexity > 0
        assert result.std_perplexity >= 0
        assert result.min_perplexity <= result.mean_perplexity
        assert result.max_perplexity >= result.mean_perplexity
        assert result.duration_seconds > 0

    def test_perplexity_graceful_oom_handling(
        self, tiny_model: tuple, sample_wikitext: list[str], tmp_path
    ) -> None:
        """Verify OOM error triggers batch_size=1 fallback retry."""
        model, tokenizer = tiny_model
        bench = PerplexityBenchmark()

        dataset_path = tmp_path / "test_wiki.txt"
        dataset_path.write_text("\n".join(sample_wikitext[:10]), encoding="utf-8")

        call_count = 0
        original_run = bench._run_hf_batches

        def mock_run(m, t, sentences, batch_size, device):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and batch_size > 1:
                raise RuntimeError("CUDA out of memory")
            return original_run(m, t, sentences, batch_size, device)

        bench._run_hf_batches = mock_run  # type: ignore[method-assign]

        result = bench.measure(
            model, tokenizer,
            dataset_path=str(dataset_path),
            batch_size=4,
            max_samples=5,
        )

        assert call_count >= 2, "Should retry with batch_size=1 after OOM"
        assert result.mean_perplexity > 0

    def test_load_wikitext_filters_short_lines(self, tmp_path) -> None:
        """Verify short lines (< 50 chars) are filtered from the dataset."""
        bench = PerplexityBenchmark()
        dataset_path = tmp_path / "test.txt"
        dataset_path.write_text(
            "Short.\nThis is a properly long sentence that exceeds the fifty character minimum threshold.\n",
            encoding="utf-8",
        )
        sentences = bench._load_wikitext2_subset(str(dataset_path))
        assert len(sentences) == 1
        assert "properly long" in sentences[0]

    def test_perplexity_fallback_on_missing_file(self) -> None:
        """Verify fallback sentences are returned when file is missing."""
        bench = PerplexityBenchmark()
        sentences = bench._load_wikitext2_subset("/nonexistent/path/data.txt")
        assert len(sentences) > 0


class TestIsLlamaCpp:
    """Tests for _is_llama_cpp detection helper."""

    def test_detects_llama_cpp_model(self) -> None:
        """Mock Llama object should be detected as llama_cpp."""
        mock_llama = MagicMock()
        mock_llama.__class__.__name__ = "Llama"
        mock_llama.create_completion = MagicMock()
        assert _is_llama_cpp(mock_llama) is True

    def test_hf_model_not_llama_cpp(self) -> None:
        """HuggingFace model should not be detected as llama_cpp."""
        mock_hf = MagicMock()
        mock_hf.__class__.__name__ = "LlamaForCausalLM"
        assert _is_llama_cpp(mock_hf) is False
