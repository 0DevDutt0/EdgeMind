"""Deploy quantized models to Ollama for local serving."""

from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console

from edgemind.core.logging import get_logger

logger = get_logger(__name__)
console = Console(legacy_windows=False)


class OllamaDeployer:
    """Deploy a quantized HuggingFace or GGUF model to Ollama.

    Creates a Modelfile and registers it with the local Ollama instance.
    After deployment, the model is accessible via 'ollama run {name}'.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        """Initialize the deployer.

        Args:
            base_url: Ollama server base URL.
        """
        self._base_url = base_url

    def deploy(
        self,
        model_path: str,
        model_name: str | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        context_length: int = 4096,
    ) -> dict:
        """Deploy a model to Ollama.

        Detects whether the model is GGUF (single file) or a HuggingFace
        directory and creates the appropriate Modelfile.

        Args:
            model_path: Path to GGUF file or HF model directory.
            model_name: Name to register in Ollama. Defaults to path basename.
            system_prompt: System prompt embedded in the Modelfile.
            context_length: Context window size (num_ctx).

        Returns:
            Dict with model_name, modelfile_path, and test_output.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        name = model_name or path.stem.lower().replace(".", "-").replace("_", "-")

        modelfile_path = path.parent / f"Modelfile.{name}"
        modelfile_content = self._build_modelfile(model_path, system_prompt, context_length)
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        console.print(f"[cyan]Creating Ollama model:[/cyan] {name}")
        console.print(f"[dim]Modelfile: {modelfile_path}[/dim]")

        result = subprocess.run(
            ["ollama", "create", name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ollama create failed:\n{result.stderr}\n"
                "Ensure Ollama is installed and running: https://ollama.com"
            )

        console.print(f"[green]âœ“ Model registered as '{name}'[/green]")

        test_output = self._test_model(name)

        console.print("\n[bold]Usage:[/bold]")
        console.print(f"  [cyan]ollama run {name}[/cyan]")
        console.print(
            f"  [cyan]curl {self._base_url}/api/generate -d "
            f'{{"model": "{name}", "prompt": "Hello"}}[/cyan]'
        )

        return {
            "model_name": name,
            "modelfile_path": str(modelfile_path),
            "test_output": test_output,
        }

    def _build_modelfile(
        self, model_path: str, system_prompt: str, context_length: int
    ) -> str:
        """Build an Ollama Modelfile for the given model.

        Args:
            model_path: Path to GGUF file or HF directory.
            system_prompt: System prompt for the model.
            context_length: Context window size.

        Returns:
            Modelfile content string.
        """
        path = Path(model_path)
        if path.is_file() and path.suffix == ".gguf":
            from_line = f"FROM {path.resolve()}"
        else:
            gguf_files = list(path.glob("*.gguf"))
            if gguf_files:
                from_line = f"FROM {gguf_files[0].resolve()}"
            else:
                from_line = f"FROM {path.resolve()}"

        return (
            f"{from_line}\n\n"
            f'SYSTEM """{system_prompt}"""\n\n'
            f"PARAMETER num_ctx {context_length}\n"
            f"PARAMETER temperature 0.7\n"
            f"PARAMETER top_p 0.9\n"
        )

    def _test_model(self, model_name: str) -> str:
        """Run a quick test inference via Ollama.

        Args:
            model_name: Registered Ollama model name.

        Returns:
            Generated response text (first 100 chars).
        """
        try:
            result = subprocess.run(
                ["ollama", "run", model_name, "Say hello in one sentence."],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout.strip()[:100]
            console.print(f"[green]âœ“ Test inference:[/green] {output!r}")
            return output
        except Exception as exc:
            logger.warning(f"Test inference failed: {exc}")
            return ""

