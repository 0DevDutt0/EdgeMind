"""Structured logging setup with Rich for EdgeMind."""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

console = Console(legacy_windows=False)
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a Rich-formatted logger for the given module name.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    _loggers[name] = logger
    return logger


def log_section(title: str, **kwargs: Any) -> None:
    """Print a prominent section header to the console.

    Args:
        title: Section title text.
        **kwargs: Additional keyword arguments passed to console.rule.
    """
    console.rule(f"[bold cyan]{title}[/bold cyan]", **kwargs)


def log_success(message: str) -> None:
    """Print a success message with green styling.

    Args:
        message: The success message text.
    """
    console.print(f"[bold green]✓[/bold green] {message}")


def log_warning(message: str) -> None:
    """Print a warning message with yellow styling.

    Args:
        message: The warning message text.
    """
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def log_error(message: str) -> None:
    """Print an error message with red styling.

    Args:
        message: The error message text.
    """
    console.print(f"[bold red]✗[/bold red] {message}")
