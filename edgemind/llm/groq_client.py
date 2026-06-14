"""Groq API async client for quality comparison benchmarking."""

from __future__ import annotations

import time

from tenacity import retry, stop_after_attempt, wait_exponential

from edgemind.core.config import get_config
from edgemind.core.logging import get_logger

logger = get_logger(__name__)


class GroqClient:
    """Async-compatible Groq API client with retry logic and rate-limit handling.

    Used for:
    - Quality baseline generation (llama-3.3-70b-versatile)
    - LLM-as-Judge scoring (llama-3.1-8b-instant)
    - Side-by-side comparison benchmarks
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Initialize the Groq client.

        Args:
            api_key: Groq API key. Defaults to GROQ_API_KEY from config.
            model: Default model for completions. Defaults to config value.
        """
        config = get_config()
        self._api_key = api_key or config.groq_api_key
        self._default_model = model or config.groq_comparison_model

        if not self._api_key:
            logger.warning(
                "GROQ_API_KEY not set — Groq comparison features will be disabled. "
                "Add it to .env to enable quality comparison benchmarks."
            )

    def _get_client(self) -> object:
        """Lazily initialize the Groq SDK client."""
        try:
            from groq import Groq  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "groq SDK is required for API comparison.\n"
                "Install: pip install groq"
            ) from exc
        return Groq(api_key=self._api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def complete(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> str:
        """Generate a completion via Groq API.

        Args:
            prompt: User prompt text.
            model: Model ID to use. Defaults to default_model.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            system_prompt: Optional system prompt.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If API key is not configured.
        """
        if not self._api_key:
            raise RuntimeError(
                "Groq API key not configured. Set GROQ_API_KEY in .env"
            )

        client = self._get_client()
        target_model = model or self._default_model

        t0 = time.perf_counter()
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=target_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - t0

        text = response.choices[0].message.content or ""
        usage = response.usage
        logger.debug(
            f"Groq [{target_model}] {elapsed:.2f}s — "
            f"in:{usage.prompt_tokens} out:{usage.completion_tokens}"
        )
        return text.strip()

    def estimate_tokens_per_second(
        self,
        prompt: str,
        max_tokens: int = 100,
    ) -> float:
        """Estimate Groq API throughput in tokens/second.

        Args:
            prompt: Test prompt.
            max_tokens: Tokens to generate for timing.

        Returns:
            Approximate tokens per second from the API.
        """
        t0 = time.perf_counter()
        self.complete(prompt, max_tokens=max_tokens)
        elapsed = time.perf_counter() - t0
        return round(max_tokens / elapsed, 1) if elapsed > 0 else 0.0

    @property
    def is_configured(self) -> bool:
        """Return True if an API key is set."""
        return bool(self._api_key)
