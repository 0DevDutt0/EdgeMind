"""Unit tests for quality evaluation scoring logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import json

import pytest

from edgemind.benchmarks.quality_eval import QualityEvaluator
from edgemind.models.benchmark_models import QualityResult


class TestQualityEvaluator:
    """Tests for QualityEvaluator.evaluate()."""

    def test_heuristic_score_empty_response(self) -> None:
        """Empty response should receive a low score."""
        evaluator = QualityEvaluator()
        score = evaluator._heuristic_score("")
        assert score <= 3.0

    def test_heuristic_score_short_response(self) -> None:
        """Very short response should receive below-average score."""
        evaluator = QualityEvaluator()
        score = evaluator._heuristic_score("Yes.")
        assert score < 5.0

    def test_heuristic_score_long_response(self) -> None:
        """Long, substantive response should receive higher score."""
        evaluator = QualityEvaluator()
        long_response = (
            "Transformer attention mechanisms work by computing compatibility scores "
            "between query, key, and value vectors. Each token attends to all other tokens "
            "weighted by these compatibility scores. This allows the model to capture "
            "long-range dependencies and focus on relevant context."
        )
        score = evaluator._heuristic_score(long_response)
        assert score >= 6.0

    def test_load_eval_prompts_fallback(self) -> None:
        """Should return fallback prompts if JSON file is not found."""
        evaluator = QualityEvaluator()
        with patch("edgemind.benchmarks.quality_eval.get_config") as mock_config:
            cfg = MagicMock()
            from pathlib import Path
            cfg.custom_eval_path = Path("/nonexistent/path.json")
            mock_config.return_value = cfg
            prompts = evaluator._load_eval_prompts()
        assert len(prompts) > 0
        assert "prompt" in prompts[0]
        assert "category" in prompts[0]

    def test_load_eval_prompts_from_file(self, tmp_path) -> None:
        """Should load prompts from JSON file when it exists."""
        prompts_data = [
            {"id": 1, "category": "explanation", "prompt": "Test prompt 1"},
            {"id": 2, "category": "coding", "prompt": "Test prompt 2"},
        ]
        json_path = tmp_path / "eval.json"
        json_path.write_text(json.dumps(prompts_data), encoding="utf-8")

        evaluator = QualityEvaluator()
        with patch("edgemind.benchmarks.quality_eval.get_config") as mock_config:
            cfg = MagicMock()
            cfg.custom_eval_path = json_path
            mock_config.return_value = cfg
            prompts = evaluator._load_eval_prompts()

        assert len(prompts) == 2
        assert prompts[0]["prompt"] == "Test prompt 1"

    def test_quality_result_structure(self, tiny_model, sample_eval_prompts, tmp_path) -> None:
        """Full evaluation should return a valid QualityResult."""
        model, tokenizer = tiny_model
        evaluator = QualityEvaluator()

        json_path = tmp_path / "eval.json"
        json_path.write_text(json.dumps(sample_eval_prompts[:3]), encoding="utf-8")

        with patch("edgemind.benchmarks.quality_eval.get_config") as mock_config:
            cfg = MagicMock()
            cfg.custom_eval_path = json_path
            cfg.groq_api_key = ""
            mock_config.return_value = cfg

            result = evaluator.evaluate(model, tokenizer, groq_client=None)

        assert isinstance(result, QualityResult)
        assert result.mean_quality_score > 0
        assert result.num_prompts > 0
        assert isinstance(result.prompt_categories, dict)
        assert result.duration_seconds > 0
