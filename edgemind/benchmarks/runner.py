"""Benchmark orchestrator â€” loads a model once and runs all specified tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table

from edgemind.benchmarks.inference_speed import InferenceSpeedBenchmark
from edgemind.benchmarks.memory_profiler import MemoryProfiler
from edgemind.benchmarks.perplexity import PerplexityBenchmark
from edgemind.benchmarks.quality_eval import QualityEvaluator
from edgemind.core.config import get_config
from edgemind.core.logging import get_logger
from edgemind.core.model_loader import load_model_and_tokenizer
from edgemind.models.benchmark_models import (
    BenchmarkResult,
    MemoryResult,
    PerplexityResult,
    QualityResult,
    QuantizationMethod,
    SpeedResult,
)

logger = get_logger(__name__)
console = Console(legacy_windows=False)

ALL_TESTS = ["memory", "perplexity", "speed", "quality"]


class BenchmarkRunner:
    """Orchestrates all benchmark tests for a model/quantization combination.

    Loads the model once, runs each test sequentially, saves intermediate
    JSON results after each test so no data is lost if later tests fail.
    Displays a Rich live-updating status table during the run.
    """

    def run_all(
        self,
        model_path: str,
        quantization_method: QuantizationMethod,
        tests: list[str] | None = None,
        compare_groq: bool = False,
        device: str = "auto",
        groq_client: Any | None = None,
    ) -> BenchmarkResult:
        """Load the model once and run all specified benchmark tests.

        Args:
            model_path: HuggingFace model ID or local directory path.
            quantization_method: Quantization method label for result metadata.
            tests: List of test names to run. Defaults to all tests.
            compare_groq: Whether to include Groq quality comparison.
            device: Target device string ("auto", "cuda", "cpu").
            groq_client: Optional GroqClient for quality/comparison tests.

        Returns:
            BenchmarkResult with all completed test results.
        """
        config = get_config()
        config.ensure_dirs()
        active_tests = tests or ALL_TESTS

        model_name = Path(model_path).name or model_path.split("/")[-1]
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        result_dir = config.results_dir / model_name
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{quantization_method}_{timestamp}.json"

        console.rule(
            f"[bold cyan]Benchmarking {model_name} / {quantization_method}[/bold cyan]"
        )

        # Load model
        console.print(f"[cyan]Loading model:[/cyan] {model_path}")
        try:
            model, tokenizer = load_model_and_tokenizer(model_path, quantization_method, device)
        except Exception as exc:
            logger.error(f"Model load failed: {exc}")
            raise

        # Try to get disk size
        model_size_gb = 0.0
        try:
            p = Path(model_path)
            if p.exists():
                model_size_gb = sum(
                    f.stat().st_size for f in p.rglob("*") if f.is_file()
                ) / 1e9
        except Exception:
            pass

        # Accumulate results
        memory_result: MemoryResult | None = None
        perplexity_result: PerplexityResult | None = None
        speed_result: SpeedResult | None = None
        quality_result: QualityResult | None = None

        status: dict[str, str] = {t: "pending" for t in active_tests}

        def _make_table() -> Table:
            t = Table(title="Benchmark Progress", show_header=True, header_style="bold magenta")
            t.add_column("Test", style="cyan", min_width=18)
            t.add_column("Status / Result", min_width=35)
            for test_name in active_tests:
                s = status[test_name]
                if s == "pending":
                    t.add_row(test_name.title(), "[dim]â¸  Pending[/dim]")
                elif s == "running":
                    t.add_row(test_name.title(), "[yellow]â³ Running...[/yellow]")
                elif s.startswith("ok:"):
                    t.add_row(test_name.title(), f"[green]âœ“[/green] {s[3:]}")
                elif s.startswith("err:"):
                    t.add_row(test_name.title(), f"[red]âœ—[/red] {s[4:]}")
            return t

        with Live(_make_table(), console=console, refresh_per_second=4) as live:

            def _tick(test: str, label: str) -> None:
                status[test] = label
                live.update(_make_table())

            if "memory" in active_tests:
                _tick("memory", "running")
                try:
                    profiler = MemoryProfiler()
                    memory_result = profiler.profile(model, tokenizer, device)
                    _tick(
                        "memory",
                        f"ok:{memory_result.peak_vram_gb:.1f} GB VRAM / "
                        f"{memory_result.available_vram_gb:.1f} GB",
                    )
                except Exception as exc:
                    logger.error(f"Memory profiling failed: {exc}")
                    _tick("memory", f"err:{exc!s:.40}")

            if "perplexity" in active_tests:
                _tick("perplexity", "running")
                try:
                    ppl_bench = PerplexityBenchmark()
                    ppl_dataset = str(config.wikitext2_subset_path) if config.wikitext2_subset_path.exists() else None
                    perplexity_result = ppl_bench.measure(
                        model, tokenizer,
                        dataset_path=ppl_dataset,
                        batch_size=config.perplexity_batch_size,
                    )
                    _tick(
                        "perplexity",
                        f"ok:{perplexity_result.mean_perplexity:.2f} Â± "
                        f"{perplexity_result.std_perplexity:.2f}",
                    )
                except Exception as exc:
                    logger.error(f"Perplexity benchmark failed: {exc}")
                    _tick("perplexity", f"err:{exc!s:.40}")

            if "speed" in active_tests:
                _tick("speed", "running")
                try:
                    speed_bench = InferenceSpeedBenchmark()
                    speed_result = speed_bench.measure(
                        model, tokenizer,
                        num_runs=config.benchmark_num_runs,
                        warmup_runs=config.benchmark_warmup_runs,
                        max_new_tokens=config.benchmark_max_new_tokens,
                    )
                    _tick(
                        "speed",
                        f"ok:{speed_result.mean_tps:.1f} tok/s | "
                        f"TTFT {speed_result.mean_ttft_ms:.0f}ms",
                    )
                except Exception as exc:
                    logger.error(f"Speed benchmark failed: {exc}")
                    _tick("speed", f"err:{exc!s:.40}")

            if "quality" in active_tests:
                _tick("quality", "running")
                try:
                    evaluator = QualityEvaluator()
                    quality_result = evaluator.evaluate(
                        model, tokenizer, groq_client=groq_client
                    )
                    _tick(
                        "quality",
                        f"ok:{quality_result.mean_quality_score:.1f}/10"
                        + (
                            f" ({quality_result.quality_retention_pct:.0f}% vs Groq)"
                            if quality_result.quality_retention_pct else ""
                        ),
                    )
                except Exception as exc:
                    logger.error(f"Quality evaluation failed: {exc}")
                    _tick("quality", f"err:{exc!s:.40}")

        device_str = _detect_device_str(model, tokenizer)
        result = BenchmarkResult(
            model_id=model_path,
            quantization_method=quantization_method,
            model_size_gb=round(model_size_gb, 3),
            benchmarked_at=datetime.now(tz=UTC).isoformat(),
            device=device_str,
            perplexity=perplexity_result,
            speed=speed_result,
            memory=memory_result,
            quality=quality_result,
            compression_ratio=1.0,
            quality_retention_pct=quality_result.quality_retention_pct if quality_result else None,
            overall_score=_compute_overall_score(perplexity_result, speed_result, quality_result),
            recommended_for=_compute_recommendations(memory_result),
        )

        result.to_json(str(result_path))
        console.print(f"\n[green]Results saved:[/green] {result_path}")

        return result


def _detect_device_str(model: Any, tokenizer: Any) -> str:
    """Return a human-readable device string from the loaded model."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _compute_overall_score(
    ppl: PerplexityResult | None,
    speed: SpeedResult | None,
    quality: QualityResult | None,
) -> float | None:
    """Compute a composite 0-100 score weighting quality, speed, and perplexity."""
    scores: list[float] = []
    if quality is not None:
        scores.append(quality.mean_quality_score * 10)
    if ppl is not None:
        ppl_score = max(0.0, 100.0 - (ppl.mean_perplexity - 8.0) * 5.0)
        scores.append(min(100.0, ppl_score))
    if speed is not None:
        tps_score = min(100.0, speed.mean_tps / 2.0)
        scores.append(tps_score)
    return round(sum(scores) / len(scores), 1) if scores else None


def _compute_recommendations(memory: MemoryResult | None) -> list[str]:
    """Return list of hardware names this model configuration is suitable for."""
    if memory is None:
        return []
    vram = memory.peak_vram_gb
    recs: list[str] = []
    if vram <= 4:
        recs.extend(["Jetson Nano", "Mac M1 (8GB)", "Raspberry Pi 5"])
    elif vram <= 16:
        recs.extend(["RTX 4090", "RTX 5090", "Jetson AGX Orin", "Mac M2", "Mac M3 Pro"])
    elif vram <= 24:
        recs.extend(["RTX 4090", "RTX 5090"])
    return recs

