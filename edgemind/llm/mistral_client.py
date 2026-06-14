"""Mistral API client for secondary quality comparison."""

from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from edgemind.core.config import get_config
from edgemind.core.logging import get_logger

logger = get_logger(__name__)


class MistralClient:
    """Mistral API client for secondary comparison benchmarks.

    Uses mistral-small (free tier) as an additional quality baseline
    alongside Groq. Useful for cross-provider quality comparison.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        """Initialize the Mistral client.

        Args:
            api_key: Mistral API key. Defaults to MISTRAL_API_KEY from config.
            model: Default model. Defaults to config value (mistral-small-latest).
        """
        config = get_config()
        self._api_key = api_key or config.mistral_api_key
        self._default_model = model or config.mistral_comparison_model

        if not self._api_key:
            logger.warning(
                "MISTRAL_API_KEY not set — Mistral comparison disabled. "
                "Add it to .env to enable secondary comparison benchmarks."
            )

    def _get_client(self) -> object:
        """Lazily initialize the Mistral SDK client."""
        try:
            from mistralai import Mistral  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "mistralai is required for Mistral comparison.\n"
                "Install: pip install mistralai"
            ) from exc
        return Mistral(api_key=self._api_key)

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
    ) -> str:
        """Generate a completion via Mistral API.

        Args:
            prompt: User prompt text.
            model: Model ID to use. Defaults to mistral-small-latest.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If API key is not configured.
        """
        if not self._api_key:
            raise RuntimeError(
                "Mistral API key not configured. Set MISTRAL_API_KEY in .env"
            )

        client = self._get_client()
        target_model = model or self._default_model

        response = client.chat.complete(  # type: ignore[attr-defined]
            model=target_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = response.choices[0].message.content or ""
        logger.debug(f"Mistral [{target_model}] completed")
        return text.strip()

    @property
    def is_configured(self) -> bool:
        """Return True if an API key is set."""
        return bool(self._api_key)
