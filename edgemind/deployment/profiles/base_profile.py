"""Abstract hardware profile base class for EdgeMind deployment advisor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareProfile:
    """Defines hardware constraints and recommended quantization strategies.

    Each profile encodes the capabilities of a specific edge device and
    maps model parameter ranges to the optimal quantization method,
    expected throughput, and VRAM/RAM requirements.
    """

    name: str
    vram_gb: float
    ram_gb: float
    cuda_compute: str | None
    recommended_inference: str
    suitable_model_sizes: list[str]
    max_recommended_params_billions: float
    quantization_recommendations: dict[str, dict]
    setup_notes: str
    architecture: str = "unknown"

    def get_recommendation(self, param_billions: float) -> dict:
        """Return the best quantization recommendation for a given model size.

        Selects the recommendation whose param range bracket best matches
        the requested size. Falls back to the largest bracket if none match.

        Args:
            param_billions: Model parameter count in billions.

        Returns:
            Dict with keys: method, expected_tps, vram_required_gb, notes.
        """
        # Find matching bracket
        for bracket, rec in self.quantization_recommendations.items():
            low, high = _parse_bracket(bracket)
            if low <= param_billions <= high:
                return dict(rec)

        # No exact match: use the bracket with the highest lower bound
        best_bracket = max(
            self.quantization_recommendations.keys(),
            key=lambda b: _parse_bracket(b)[0],
        )
        return dict(self.quantization_recommendations[best_bracket])

    def can_run(self, param_billions: float, method: str) -> bool:
        """Check if this hardware can run the given model/method combination.

        Args:
            param_billions: Model parameter count in billions.
            method: Quantization method string.

        Returns:
            True if estimated VRAM/RAM fits within hardware limits.
        """
        from edgemind.core.gpu_utils import GPUInfo
        from edgemind.models.benchmark_models import QuantizationMethod

        try:
            qm = QuantizationMethod(method)
        except ValueError:
            return False

        gpu_info = GPUInfo()
        available = self.vram_gb if self.vram_gb > 0 else self.ram_gb
        fits, _ = gpu_info.check_fits_in_vram(param_billions, qm, available)
        return fits


def _parse_bracket(bracket: str) -> tuple[float, float]:
    """Parse a param range bracket string like '7B' or '1B-3B' into (low, high).

    Args:
        bracket: String like '7B', '1B-3B', or '32B'.

    Returns:
        Tuple of (low_billions, high_billions).
    """
    bracket = bracket.replace("B", "").strip()
    if "-" in bracket:
        parts = bracket.split("-")
        return float(parts[0]), float(parts[1])
    val = float(bracket)
    # Use ±25% of the stated size for a tighter, more accurate bracket
    return val * 0.75, val * 1.25
