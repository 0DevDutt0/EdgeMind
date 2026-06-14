"""Pydantic Settings configuration for EdgeMind."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EdgeMindConfig(BaseSettings):
    """Runtime configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM API keys
    groq_api_key: str = Field(default="", description="Groq API key for comparison benchmarks")
    mistral_api_key: str = Field(default="", description="Mistral API key")

    # Comparison models
    groq_comparison_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model for quality comparison baseline",
    )
    mistral_comparison_model: str = Field(
        default="mistral-small-latest",
        description="Mistral model for secondary comparison",
    )

    # Local inference
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )

    # Paths
    results_dir: Path = Field(
        default=Path("./results"),
        description="Directory for benchmark result JSON files",
    )
    eval_data_dir: Path = Field(
        default=Path("./data/eval"),
        description="Directory containing evaluation datasets",
    )
    wikitext2_subset_path: Path = Field(
        default=Path("./data/eval/wikitext2_subset.txt"),
        description="Path to WikiText-2 evaluation subset",
    )
    custom_eval_path: Path = Field(
        default=Path("./data/eval/custom_eval_set.json"),
        description="Path to custom quality evaluation prompts",
    )

    # Benchmark parameters
    benchmark_num_runs: int = Field(default=10, description="Speed benchmark repetitions")
    benchmark_warmup_runs: int = Field(default=3, description="Warmup runs before recording")
    benchmark_max_new_tokens: int = Field(
        default=100, description="Tokens to generate per speed benchmark run"
    )
    perplexity_batch_size: int = Field(default=8, description="Batch size for perplexity eval")
    quality_eval_judge_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model used as LLM-as-Judge for quality scoring",
    )

    def ensure_dirs(self) -> None:
        """Create results and eval directories if they do not exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.eval_data_dir.mkdir(parents=True, exist_ok=True)


_config_instance: EdgeMindConfig | None = None


def get_config() -> EdgeMindConfig:
    """Return the singleton EdgeMind configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = EdgeMindConfig()
    return _config_instance
