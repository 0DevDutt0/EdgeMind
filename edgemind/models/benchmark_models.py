"""Data models for EdgeMind benchmark results and quantization metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path


class QuantizationMethod(StrEnum):
    """All supported quantization methods."""

    BF16 = "bf16"
    FP16 = "fp16"
    INT8 = "int8"
    INT4_NF4 = "int4_nf4"
    INT4_FP4 = "int4_fp4"
    GPTQ_4BIT = "gptq_4bit"
    AWQ_4BIT = "awq_4bit"
    GGUF_Q2K = "gguf_q2_k"
    GGUF_Q3KM = "gguf_q3_k_m"
    GGUF_Q4KM = "gguf_q4_k_m"
    GGUF_Q5KM = "gguf_q5_k_m"
    GGUF_Q8 = "gguf_q8_0"
    GGUF_F16 = "gguf_f16"


DISPLAY_NAMES: dict[str, str] = {
    QuantizationMethod.BF16: "BFloat16 (Full Precision)",
    QuantizationMethod.FP16: "Float16 (Full Precision)",
    QuantizationMethod.INT8: "INT8 (bitsandbytes)",
    QuantizationMethod.INT4_NF4: "INT4 NF4 (bitsandbytes)",
    QuantizationMethod.INT4_FP4: "INT4 FP4 (bitsandbytes)",
    QuantizationMethod.GPTQ_4BIT: "GPTQ 4-bit (AutoGPTQ)",
    QuantizationMethod.AWQ_4BIT: "AWQ 4-bit (AutoAWQ)",
    QuantizationMethod.GGUF_Q2K: "GGUF Q2_K",
    QuantizationMethod.GGUF_Q3KM: "GGUF Q3_K_M",
    QuantizationMethod.GGUF_Q4KM: "GGUF Q4_K_M (Recommended)",
    QuantizationMethod.GGUF_Q5KM: "GGUF Q5_K_M",
    QuantizationMethod.GGUF_Q8: "GGUF Q8_0",
    QuantizationMethod.GGUF_F16: "GGUF F16",
}


@dataclass
class PerplexityResult:
    """WikiText-2 perplexity measurement results."""

    mean_perplexity: float
    std_perplexity: float
    min_perplexity: float
    max_perplexity: float
    num_samples: int
    dataset: str
    duration_seconds: float


@dataclass
class SpeedResult:
    """Inference speed measurement results."""

    mean_tps: float
    std_tps: float
    min_tps: float
    max_tps: float
    mean_ttft_ms: float
    std_ttft_ms: float
    num_runs: int
    prompt_tokens: int
    max_new_tokens: int
    device: str
    duration_seconds: float


@dataclass
class MemoryResult:
    """VRAM and system RAM usage profile."""

    model_vram_gb: float
    peak_vram_gb: float
    available_vram_gb: float
    system_ram_gb: float
    fits_on_device: bool
    device_name: str


@dataclass
class QualityResult:
    """LLM-as-Judge quality evaluation results."""

    mean_quality_score: float
    std_quality_score: float
    num_prompts: int
    prompt_categories: dict[str, float]
    groq_comparison_score: float | None
    quality_retention_pct: float | None
    duration_seconds: float


@dataclass
class QuantizationResult:
    """Result from a quantization operation."""

    model_id: str
    method: QuantizationMethod
    output_dir: str
    size_gb: float
    vram_at_load_gb: float = 0.0
    quantization_config: dict = field(default_factory=dict)
    calibration_samples_used: int = 0
    duration_seconds: float = 0.0

    def to_json(self, path: str) -> None:
        """Save result to JSON file."""
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one model/quantization combination."""

    model_id: str
    quantization_method: QuantizationMethod
    model_size_gb: float
    benchmarked_at: str
    device: str
    perplexity: PerplexityResult | None
    speed: SpeedResult | None
    memory: MemoryResult | None
    quality: QualityResult | None
    compression_ratio: float
    quality_retention_pct: float | None
    overall_score: float | None
    recommended_for: list[str]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save benchmark result to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkResult:
        """Reconstruct from a loaded JSON dict."""
        perplexity = PerplexityResult(**data["perplexity"]) if data.get("perplexity") else None
        speed = SpeedResult(**data["speed"]) if data.get("speed") else None
        memory = MemoryResult(**data["memory"]) if data.get("memory") else None
        quality = QualityResult(**data["quality"]) if data.get("quality") else None
        return cls(
            model_id=data["model_id"],
            quantization_method=QuantizationMethod(data["quantization_method"]),
            model_size_gb=data["model_size_gb"],
            benchmarked_at=data["benchmarked_at"],
            device=data["device"],
            perplexity=perplexity,
            speed=speed,
            memory=memory,
            quality=quality,
            compression_ratio=data.get("compression_ratio", 1.0),
            quality_retention_pct=data.get("quality_retention_pct"),
            overall_score=data.get("overall_score"),
            recommended_for=data.get("recommended_for", []),
        )

    @classmethod
    def from_json(cls, path: str) -> BenchmarkResult:
        """Load benchmark result from JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class ModelSummary:
    """Aggregated summary across all quantization methods for one model."""

    model_id: str
    baseline_perplexity: float
    baseline_tps: float
    baseline_size_gb: float
    results: list[BenchmarkResult]
    recommended_method: str
    recommended_method_reason: str

    def to_json(self, path: str) -> None:
        """Save summary to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": self.model_id,
            "baseline_perplexity": self.baseline_perplexity,
            "baseline_tps": self.baseline_tps,
            "baseline_size_gb": self.baseline_size_gb,
            "recommended_method": self.recommended_method,
            "recommended_method_reason": self.recommended_method_reason,
            "results": [r.to_dict() for r in self.results],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
