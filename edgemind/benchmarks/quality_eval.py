"""LLM-as-Judge quality evaluation for quantized models."""

from __future__ import annotations

import json
import time
from typing import Any

from tqdm import tqdm

from edgemind.core.config import get_config
from edgemind.core.logging import get_logger
from edgemind.models.benchmark_models import QualityResult

logger = get_logger(__name__)

_JUDGE_PROMPT = """Rate this AI response quality from 1-10.

Prompt: {prompt}
Response: {response}

Criteria:
- Accuracy and correctness (1-4 points)
- Clarity and explanation quality (1-3 points)
- Completeness (1-2 points)
- Conciseness (0-1 point)

Output ONLY a number from 1 to 10."""


class QualityEvaluator:
    """Evaluate response quality using LLM-as-Judge methodology.

    Generates responses from a local quantized model across 20 diverse prompts
    in 5 categories (explanation, coding, reasoning, creative, factual), then
    uses Groq's llama-3.1-8b-instant to score each response on a 0-10 scale.

    When a Groq client is provided, scores the Groq baseline response too,
    enabling quality_retention_pct = (local_score / groq_score) * 100.

    Interpretation:
    > 95%: Excellent — quantization barely affects quality
    90-95%: Good — minor quality reduction
    85-90%: Acceptable — noticeable but tolerable
    < 85%: Poor — significant quality loss
    """

    def evaluate(
        self,
        local_model: Any,
        local_tokenizer: Any,
        groq_client: Any | None = None,
        judge_model: str = "llama-3.1-8b-instant",
        comparison_model: str = "llama-3.3-70b-versatile",
    ) -> QualityResult:
        """Run quality evaluation across 20 diverse prompts.

        Args:
            local_model: Loaded local model (HF or llama_cpp).
            local_tokenizer: HuggingFace tokenizer (None for GGUF).
            groq_client: Optional GroqClient for baseline comparison.
            judge_model: Groq model ID for scoring (must be accessible).
            comparison_model: Groq model used as quality baseline.

        Returns:
            QualityResult with mean score, per-category breakdown, and
            optional Groq comparison metrics.
        """
        config = get_config()
        prompts = self._load_eval_prompts()
        is_gguf = local_tokenizer is None or _is_llama_cpp(local_model)

        local_scores: list[float] = []
        groq_scores: list[float] = []
        category_scores: dict[str, list[float]] = {}

        start = time.perf_counter()

        for entry in tqdm(prompts, desc="Quality evaluation"):
            prompt = entry["prompt"]
            category = entry["category"]

            local_response = self._generate_local(
                local_model, local_tokenizer, prompt, is_gguf, max_tokens=200
            )

            if groq_client is not None and config.groq_api_key:
                groq_response = self._generate_groq(groq_client, prompt, model=comparison_model)
                groq_score = self._score_with_judge(
                    groq_client, judge_model, prompt, groq_response
                )
                groq_scores.append(groq_score)
            else:
                groq_response = None

            local_score = self._score_with_judge(
                groq_client, judge_model, prompt, local_response
            ) if groq_client else self._heuristic_score(local_response)

            local_scores.append(local_score)
            category_scores.setdefault(category, []).append(local_score)

        import numpy as np

        local_arr = np.array(local_scores)
        per_category = {
            cat: round(float(np.mean(scores)), 2)
            for cat, scores in category_scores.items()
        }

        groq_mean = float(np.mean(groq_scores)) if groq_scores else None
        retention = (
            round((float(local_arr.mean()) / groq_mean) * 100, 1)
            if groq_mean and groq_mean > 0
            else None
        )

        return QualityResult(
            mean_quality_score=round(float(local_arr.mean()), 2),
            std_quality_score=round(float(local_arr.std()), 2),
            num_prompts=len(local_scores),
            prompt_categories=per_category,
            groq_comparison_score=round(groq_mean, 2) if groq_mean else None,
            quality_retention_pct=retention,
            duration_seconds=round(time.perf_counter() - start, 2),
        )

    def _load_eval_prompts(self) -> list[dict]:
        """Load 20 evaluation prompts from the custom eval set JSON.

        Returns:
            List of dicts with keys: id, category, prompt.
        """
        config = get_config()
        path = config.custom_eval_path

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data
        except FileNotFoundError:
            logger.warning(f"Custom eval set not found at {path}, using built-in prompts")
            return _FALLBACK_PROMPTS

    def _generate_local(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        is_gguf: bool,
        max_tokens: int = 200,
    ) -> str:
        """Generate a response from the local model."""
        if is_gguf:
            result = model.create_completion(prompt, max_tokens=max_tokens, temperature=0.1)
            return result["choices"][0]["text"].strip()

        import torch

        device = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_groq(self, groq_client: Any, prompt: str, model: str | None = None) -> str:
        """Generate a response via Groq API."""
        try:
            kwargs: dict = {"max_tokens": 200}
            if model:
                kwargs["model"] = model
            return groq_client.complete(prompt, **kwargs)
        except Exception as exc:
            logger.warning(f"Groq generation failed: {exc}")
            return ""

    def _score_with_judge(
        self,
        groq_client: Any,
        judge_model: str,
        prompt: str,
        response: str,
    ) -> float:
        """Score a response using the LLM-as-Judge via Groq.

        Returns a float in [1, 10]. Falls back to heuristic scoring on error.
        """
        if not response.strip():
            return 1.0

        judge_prompt = _JUDGE_PROMPT.format(prompt=prompt[:300], response=response[:500])
        try:
            raw = groq_client.complete(
                judge_prompt,
                model=judge_model,
                max_tokens=5,
                temperature=0.0,
            )
            score = float(raw.strip().split()[0])
            return max(1.0, min(10.0, score))
        except Exception as exc:
            logger.debug(f"Judge scoring failed: {exc}")
            return self._heuristic_score(response)

    def _heuristic_score(self, response: str) -> float:
        """Fallback heuristic score when judge API is unavailable.

        Scores based on response length and basic quality signals.
        """
        if not response or len(response) < 20:
            return 2.0
        if len(response) < 50:
            return 4.0
        if len(response) < 100:
            return 5.5
        if len(response) < 200:
            return 6.5
        return 7.0


def _is_llama_cpp(model: Any) -> bool:
    """Detect if model is a llama_cpp.Llama instance."""
    return type(model).__name__ == "Llama" and hasattr(model, "create_completion")


_FALLBACK_PROMPTS: list[dict] = [
    {"id": 1, "category": "explanation",
     "prompt": "Explain how transformer attention mechanisms work in simple terms:"},
    {"id": 2, "category": "coding",
     "prompt": "Write a Python function that implements binary search on a sorted list:"},
    {"id": 3, "category": "reasoning",
     "prompt": "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?"},
    {"id": 4, "category": "factual",
     "prompt": "What are the key differences between BERT and GPT architectures?"},
    {"id": 5, "category": "creative",
     "prompt": "Write the opening paragraph of a science fiction story set on a generation ship:"},
]
