"""EdgeMind CLI — LLM quantization, benchmarking, and edge deployment toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from edgemind import __version__
from edgemind.core.gpu_utils import GPUInfo

app = typer.Typer(
    name="edgemind",
    help=(
        "[bold cyan]EdgeMind[/bold cyan] — Take any LLM. Quantize it, benchmark every "
        "compression level. Deploy to any edge device. One toolkit, full pipeline."
    ),
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console(legacy_windows=False)


@app.command("info")
def info() -> None:
    """Show GPU information, PyTorch version, and EdgeMind compatibility status."""
    gpu_utils = GPUInfo()
    gpu_info = gpu_utils.detect()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan", min_width=20)
    table.add_column("Value", style="white")

    table.add_row("EdgeMind version", f"v{__version__}")
    table.add_row("─" * 18, "─" * 30)

    backend = gpu_info.get("backend", "cpu")
    if backend == "cuda":
        cc = gpu_info.get("compute_capability", "unknown")
        is_blackwell = gpu_info.get("is_blackwell", False)
        full_support = gpu_info.get("full_support", False)
        arch = "Blackwell" if is_blackwell else "Ada/Ampere/Other"
        status = (
            "[bold green]✓ Full sm_120 support[/bold green]"
            if full_support
            else (
                "[bold yellow]⚠ sm_120 needs CUDA 13[/bold yellow]"
                if is_blackwell
                else "[bold green]✓ Supported[/bold green]"
            )
        )
        table.add_row("GPU", gpu_info.get("device_name", "unknown"))
        table.add_row("Architecture", arch)
        table.add_row("Compute capability", f"{cc} ({arch})")
        table.add_row("VRAM", f"{gpu_info.get('vram_gb', 0):.1f} GB")
        table.add_row("CUDA version", gpu_info.get("cuda_version", "unknown"))
        table.add_row("PyTorch version", gpu_info.get("pytorch_version", "unknown"))
        table.add_row("Status", status)
    elif backend == "mps":
        table.add_row("Backend", "Apple MPS (Metal)")
        table.add_row("Device", "Apple Silicon")
        table.add_row("Unified RAM", f"{gpu_info.get('system_ram_gb', 0):.1f} GB")
        table.add_row("PyTorch version", gpu_info.get("pytorch_version", "unknown"))
        table.add_row("Status", "[bold green]✓ MPS supported[/bold green]")
    else:
        table.add_row("Backend", "CPU only")
        table.add_row("RAM", f"{gpu_info.get('system_ram_gb', 0):.1f} GB")
        table.add_row("PyTorch version", gpu_info.get("pytorch_version", "Not installed"))
        table.add_row("Status", "[yellow]Limited — no GPU detected[/yellow]")

    panel = Panel(table, title="[bold cyan]EdgeMind System Info[/bold cyan]", padding=(1, 2))
    console.print(panel)

    if gpu_info.get("is_blackwell") and not gpu_info.get("full_support"):
        gpu_utils.print_setup_guide(gpu_info)


@app.command("quantize")
def quantize(
    model: Annotated[str, typer.Argument(help="HuggingFace model ID or local path")],
    method: Annotated[
        str,
        typer.Option("--method", help="Quantization method: bitsandbytes|gptq|awq|gguf"),
    ] = "bitsandbytes",
    bits: Annotated[int, typer.Option("--bits", help="Target bit width: 4 or 8")] = 4,
    output: Annotated[
        str | None,
        typer.Option("--output", help="Output directory. Default: ./quantized/{model}_{method}"),
    ] = None,
    quant_type: Annotated[
        str,
        typer.Option("--quant-type", help="Quant type for bitsandbytes: nf4|fp4"),
    ] = "nf4",
    gguf_quant: Annotated[
        str,
        typer.Option("--gguf-quant", help="GGUF quantization level: q4_k_m|q5_k_m|q8_0|q2_k"),
    ] = "q4_k_m",
) -> None:
    """Quantize a model using the specified method and save to disk.

    Examples:

      edgemind quantize Qwen/Qwen2.5-7B-Instruct --method bitsandbytes --bits 4

      edgemind quantize Qwen/Qwen2.5-7B-Instruct --method gguf --gguf-quant q4_k_m

      edgemind quantize Qwen/Qwen2.5-7B-Instruct --method gptq --bits 4
    """
    from edgemind.core.gpu_utils import GPUInfo

    gpu_info = GPUInfo().detect()
    if gpu_info.get("is_blackwell") and not gpu_info.get("full_support"):
        GPUInfo().print_setup_guide(gpu_info)
        raise typer.Exit(1)

    model_name = model.split("/")[-1].lower()
    default_output = f"./quantized/{model_name}_{method}_{'gguf_' + gguf_quant if method == 'gguf' else str(bits) + 'bit'}"
    out_dir = output or default_output

    console.print(f"\n[bold cyan]Quantizing:[/bold cyan] {model}")
    console.print(f"[dim]Method: {method} | Output: {out_dir}[/dim]\n")

    try:
        if method == "bitsandbytes":
            from edgemind.quantization.bitsandbytes_quant import BitsAndBytesQuantizer

            quantizer = BitsAndBytesQuantizer()
            result = quantizer.quantize(
                model, out_dir, bits=bits, quant_type=quant_type
            )
        elif method == "gptq":
            from edgemind.quantization.gptq_quant import GPTQQuantizer

            quantizer = GPTQQuantizer()
            result = quantizer.quantize(model, out_dir, bits=bits)
        elif method == "awq":
            from edgemind.quantization.awq_quant import AWQQuantizer

            quantizer = AWQQuantizer()
            result = quantizer.quantize(model, out_dir)
        elif method == "gguf":
            from edgemind.quantization.gguf_converter import GGUFConverter

            converter = GGUFConverter()
            result = converter.convert(model, out_dir, quantization=gguf_quant)
        else:
            console.print(f"[red]Unknown method: {method}[/red]")
            raise typer.Exit(1)

        console.print("\n[bold green]✓ Quantization complete[/bold green]")
        console.print(f"  Method:    {result.method}")
        console.print(f"  Output:    {result.output_dir}")
        console.print(f"  Size:      {result.size_gb:.2f} GB")
        if result.vram_at_load_gb > 0:
            console.print(f"  VRAM load: {result.vram_at_load_gb:.2f} GB")
        console.print(f"  Time:      {result.duration_seconds:.0f}s")

    except RuntimeError as exc:
        console.print(f"\n[bold red]✗ Quantization failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc


@app.command("benchmark")
def benchmark(
    model_path: Annotated[str, typer.Argument(help="Model path or HuggingFace ID to benchmark")],
    tests: Annotated[
        str,
        typer.Option("--tests", help="Comma-separated tests: perplexity,speed,memory,quality"),
    ] = "memory,perplexity,speed,quality",
    compare_groq: Annotated[
        bool,
        typer.Option("--compare-groq/--no-compare-groq", help="Compare vs Groq API baseline"),
    ] = False,
    method: Annotated[
        str,
        typer.Option("--method", help="Quantization method label for result metadata"),
    ] = "bf16",
    runs: Annotated[
        int, typer.Option("--runs", help="Number of speed benchmark runs")
    ] = 10,
    device: Annotated[
        str,
        typer.Option("--device", help="Inference device: auto|cuda|cpu|mps"),
    ] = "auto",
    output_format: Annotated[
        str,
        typer.Option("--output-format", help="Output format: table|json|markdown"),
    ] = "table",
) -> None:
    """Run comprehensive benchmarks on a model (perplexity, speed, memory, quality).

    Examples:

      edgemind benchmark ./quantized/qwen2.5-7b-int4 --method int4_nf4

      edgemind benchmark Qwen/Qwen2.5-7B-Instruct --tests speed,memory

      edgemind benchmark ./model --compare-groq --method bf16
    """
    from edgemind.benchmarks.runner import BenchmarkRunner
    from edgemind.models.benchmark_models import QuantizationMethod

    try:
        qm = QuantizationMethod(method)
    except ValueError:
        valid = ", ".join(m.value for m in QuantizationMethod)
        console.print(f"[red]Unknown method '{method}'. Valid values: {valid}[/red]")
        raise typer.Exit(1)

    test_list = [t.strip() for t in tests.split(",")]

    groq_client = None
    if compare_groq:
        from edgemind.llm.groq_client import GroqClient

        groq_client = GroqClient()
        if not groq_client.is_configured:
            console.print(
                "[yellow]⚠ GROQ_API_KEY not set — skipping Groq comparison[/yellow]"
            )
            groq_client = None

    runner = BenchmarkRunner()
    result = runner.run_all(
        model_path=model_path,
        quantization_method=qm,
        tests=test_list,
        compare_groq=compare_groq,
        device=device,
        groq_client=groq_client,
    )

    if output_format == "json":
        import json

        console.print(json.dumps(result.to_dict(), indent=2))
    elif output_format == "markdown":
        _print_result_markdown(result)
    else:
        _print_result_table(result)


@app.command("profile")
def profile(
    model: Annotated[str, typer.Argument(help="HuggingFace model ID or model name")],
    hardware: Annotated[
        str,
        typer.Option(
            "--hardware",
            help=(
                "Target hardware: rtx_5090|rtx_4090|jetson_orin|jetson_nano|"
                "raspberry_pi5|mac_m1|mac_m2|mac_m3_pro"
            ),
        ),
    ] = "auto",
    params: Annotated[
        float,
        typer.Option("--params", help="Model parameter count in billions (e.g. 7 for 7B)"),
    ] = 7.0,
) -> None:
    """Show hardware-specific quantization recommendations for a model.

    Examples:

      edgemind profile Qwen/Qwen2.5-7B-Instruct --params 7 --hardware rtx_5090

      edgemind profile llama3.2:3b --params 3 --hardware raspberry_pi5

      edgemind profile Qwen/Qwen2.5-32B-Instruct --params 32 --hardware mac_m3_pro
    """
    from edgemind.core.gpu_utils import GPUInfo
    from edgemind.deployment.profiles import PROFILE_DISPLAY_NAMES, get_profile
    from edgemind.models.benchmark_models import QuantizationMethod

    if hardware == "auto":
        gpu_info = GPUInfo().detect()
        backend = gpu_info.get("backend", "cpu")
        if backend == "cuda":
            is_blackwell = gpu_info.get("is_blackwell", False)
            hardware = "rtx_5090" if is_blackwell else "rtx_4090"
        elif backend == "mps":
            hardware = "mac_m2"
        else:
            hardware = "raspberry_pi5"
        console.print(f"[dim]Auto-detected hardware: {hardware}[/dim]")

    try:
        hw_profile = get_profile(hardware)
    except KeyError:
        console.print(
            f"[red]Unknown hardware: {hardware}[/red]\n"
            f"Available: {list(PROFILE_DISPLAY_NAMES.keys())}"
        )
        raise typer.Exit(1)

    rec = hw_profile.get_recommendation(params)
    gpu_info_obj = GPUInfo()
    method_str = rec.get("method", "int4_nf4")

    try:
        qm = QuantizationMethod(method_str)
        required_gb = gpu_info_obj.estimate_vram_requirement(params, qm)
    except ValueError:
        required_gb = params * 0.5

    available = hw_profile.vram_gb if hw_profile.vram_gb > 0 else hw_profile.ram_gb
    fits = required_gb <= available
    fits_label = f"[green]✓ fits in {available:.0f} GB[/green]" if fits else f"[red]✗ exceeds {available:.0f} GB[/red]"

    console.print(f"\n[bold cyan]Hardware Profile:[/bold cyan] {hw_profile.name} ({available:.0f}GB)")
    console.print(f"[bold cyan]Model:[/bold cyan] {model} ({params}B parameters)\n")

    console.print(f"[bold]Recommendation:[/bold] [green]{method_str.upper()}[/green]")
    console.print(f"[bold]Memory Required:[/bold] ~{required_gb:.1f} GB — {fits_label}")
    console.print(f"[bold]Expected Speed:[/bold] {rec.get('expected_tps', 'N/A')} tokens/second")
    console.print(f"[bold]Notes:[/bold] {rec.get('notes', '')}")

    # Show why this quantization was chosen
    console.print("\n[bold]Why this quantization?[/bold]")
    all_methods = [
        (QuantizationMethod.BF16, "BF16"),
        (QuantizationMethod.INT8, "INT8"),
        (QuantizationMethod.INT4_NF4, "INT4 NF4"),
        (QuantizationMethod.GGUF_Q4KM, "GGUF Q4_K_M"),
    ]
    for qm_check, label in all_methods:
        req = gpu_info_obj.estimate_vram_requirement(params, qm_check)
        f_label = "✓ fits" if req <= available else "✗ exceeds"
        color = "green" if req <= available else "red"
        console.print(f"  [{color}]{f_label}[/{color}]  {label:<14} ~{req:.1f} GB required")

    model_name = model.split("/")[-1]
    console.print(
        f"\n[bold]Next step:[/bold]\n"
        f"  [cyan]edgemind quantize {model} \\\n"
        f"      --method {'bitsandbytes' if 'int' in method_str else 'gguf' if 'gguf' in method_str else method_str} "
        f"--output ./quantized/{model_name.lower()}-{method_str}[/cyan]"
    )

    console.print("\n[dim]Setup notes:[/dim]")
    console.print(f"[dim]{hw_profile.setup_notes}[/dim]")


@app.command("deploy")
def deploy(
    model_path: Annotated[str, typer.Argument(help="Path to quantized model or GGUF file")],
    target: Annotated[
        str,
        typer.Option("--target", help="Deployment target: ollama|onnx"),
    ] = "ollama",
    model_name: Annotated[
        str | None,
        typer.Option("--name", help="Model name to register (Ollama only)"),
    ] = None,
) -> None:
    """Deploy a quantized model to Ollama or export to ONNX.

    Examples:

      edgemind deploy ./quantized/qwen2.5-7b-int4 --target ollama

      edgemind deploy ./quantized/qwen2.5-7b-int4 --target onnx
    """
    if target == "ollama":
        from edgemind.deployment.ollama_deployer import OllamaDeployer

        deployer = OllamaDeployer()
        result = deployer.deploy(model_path, model_name=model_name)
        console.print(
            f"\n[bold green]✓ Deployed to Ollama as '{result['model_name']}'[/bold green]"
        )
    elif target == "onnx":
        from edgemind.deployment.onnx_exporter import ONNXExporter

        out_dir = str(Path(model_path).parent / (Path(model_path).name + "_onnx"))
        exporter = ONNXExporter()
        result = exporter.export(model_path, out_dir)
        console.print(
            f"\n[bold green]✓ ONNX export complete:[/bold green] {result['output_path']}"
        )
    else:
        console.print(f"[red]Unknown target: {target}. Use 'ollama' or 'onnx'.[/red]")
        raise typer.Exit(1)


@app.command("compare")
def compare(
    model: Annotated[str, typer.Argument(help="Local model path or HuggingFace ID")],
    method: Annotated[
        str,
        typer.Option("--method", help="Quantization method label for the local model"),
    ] = "int4_nf4",
    num_prompts: Annotated[
        int,
        typer.Option("--prompts", help="Number of comparison prompts"),
    ] = 5,
) -> None:
    """Side-by-side quality comparison: local model vs Groq llama-3.3-70b.

    Requires GROQ_API_KEY to be set in .env

    Examples:

      edgemind compare ./quantized/qwen2.5-7b-int4 --method int4_nf4

      edgemind compare Qwen/Qwen2.5-7B-Instruct --method bf16 --prompts 10
    """
    import asyncio

    from edgemind.benchmarks.groq_comparison import GroqComparisonBenchmark
    from edgemind.core.model_loader import load_model_and_tokenizer
    from edgemind.llm.groq_client import GroqClient
    from edgemind.models.benchmark_models import QuantizationMethod

    groq = GroqClient()
    if not groq.is_configured:
        console.print(
            "[red]✗ GROQ_API_KEY not set. Add it to .env and try again.[/red]"
        )
        raise typer.Exit(1)

    try:
        qm = QuantizationMethod(method)
    except ValueError:
        qm = QuantizationMethod.BF16

    console.print(f"[cyan]Loading local model:[/cyan] {model}")
    local_model, local_tokenizer = load_model_and_tokenizer(model, qm)

    bench = GroqComparisonBenchmark()
    results = asyncio.run(
        bench.compare(local_model, local_tokenizer, groq, num_prompts=num_prompts)
    )

    _print_comparison_table(results)


def _print_result_table(result: object) -> None:
    """Print a Rich table summarizing benchmark results."""
    from edgemind.models.benchmark_models import BenchmarkResult

    assert isinstance(result, BenchmarkResult)

    table = Table(
        title=f"Benchmark Results — {result.model_id.split('/')[-1]} / {result.quantization_method}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Benchmark", style="cyan", min_width=18)
    table.add_column("Result", min_width=25)
    table.add_column("Status", min_width=16)

    if result.perplexity:
        ppl = result.perplexity
        status = "[green]✓ Good[/green]" if ppl.mean_perplexity < 12 else "[yellow]⚠ High[/yellow]"
        table.add_row(
            "Perplexity", f"{ppl.mean_perplexity:.2f} ± {ppl.std_perplexity:.2f}", status
        )

    if result.speed:
        s = result.speed
        status = (
            "[green]✓ Fast[/green]" if s.mean_tps > 50
            else "[yellow]⚠ Moderate[/yellow]" if s.mean_tps > 15
            else "[red]✗ Slow[/red]"
        )
        table.add_row("Speed (TPS)", f"{s.mean_tps:.1f} ± {s.std_tps:.1f} tok/s", status)
        ttft_status = "[green]✓ <200ms[/green]" if s.mean_ttft_ms < 200 else "[yellow]⚠ >200ms[/yellow]"
        table.add_row("TTFT", f"{s.mean_ttft_ms:.0f}ms", ttft_status)

    if result.memory:
        m = result.memory
        fits_label = "[green]✓ Fits[/green]" if m.fits_on_device else "[red]✗ OOM Risk[/red]"
        table.add_row(
            "VRAM Usage",
            f"{m.peak_vram_gb:.1f} GB / {m.available_vram_gb:.1f} GB",
            fits_label,
        )

    if result.quality:
        q = result.quality
        score_status = (
            "[green]✓ Excellent[/green]" if q.mean_quality_score >= 8
            else "[green]✓ Good[/green]" if q.mean_quality_score >= 7
            else "[yellow]⚠ Marginal[/yellow]"
        )
        table.add_row("Quality Score", f"{q.mean_quality_score:.1f}/10", score_status)
        if q.quality_retention_pct is not None:
            ret_status = (
                "[green]✓ Excellent[/green]" if q.quality_retention_pct >= 95
                else "[green]✓ Good[/green]" if q.quality_retention_pct >= 90
                else "[yellow]⚠ Marginal[/yellow]" if q.quality_retention_pct >= 85
                else "[red]✗ Poor[/red]"
            )
            table.add_row(
                "Quality vs Groq", f"{q.quality_retention_pct:.0f}%", ret_status
            )

    console.print(table)


def _print_result_markdown(result: object) -> None:
    """Print benchmark results as a Markdown table."""
    from edgemind.models.benchmark_models import BenchmarkResult

    assert isinstance(result, BenchmarkResult)
    console.print(f"## {result.model_id} / {result.quantization_method}\n")
    console.print("| Metric | Value |")
    console.print("|--------|-------|")
    if result.perplexity:
        console.print(
            f"| Perplexity | {result.perplexity.mean_perplexity:.2f} ± {result.perplexity.std_perplexity:.2f} |"
        )
    if result.speed:
        console.print(f"| TPS | {result.speed.mean_tps:.1f} ± {result.speed.std_tps:.1f} |")
        console.print(f"| TTFT | {result.speed.mean_ttft_ms:.0f}ms |")
    if result.memory:
        console.print(
            f"| VRAM | {result.memory.peak_vram_gb:.1f} / {result.memory.available_vram_gb:.1f} GB |"
        )
    if result.quality:
        console.print(f"| Quality | {result.quality.mean_quality_score:.1f}/10 |")


def _print_comparison_table(results: dict) -> None:
    """Print a side-by-side comparison table of local vs Groq results."""
    table = Table(
        title="Local Model vs Groq llama-3.3-70b",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", min_width=22)
    table.add_column("Local Model", min_width=20)
    table.add_column("Groq Baseline", min_width=20)

    table.add_row(
        "Mean Quality Score",
        f"{results['local_mean_score']:.1f}/10",
        f"{results['groq_mean_score']:.1f}/10",
    )
    table.add_row(
        "Tokens/Second",
        f"{results['local_tps']:.0f}",
        f"{results['groq_tps']:.0f}",
    )
    table.add_row("Cost per Query", "$0.00", "~$0.001")
    ret = results.get("quality_retention_pct")
    if ret:
        table.add_row("Quality Retention", f"{ret:.1f}%", "100% (baseline)")

    console.print(table)
    console.print(f"\n[bold]Verdict:[/bold] {results['verdict']}")
    console.print(f"[dim]{results['cost_advantage']}[/dim]")
