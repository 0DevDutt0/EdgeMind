"""Unit tests for GPU detection and VRAM estimation utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from edgemind.core.gpu_utils import GPUInfo
from edgemind.models.benchmark_models import QuantizationMethod


class TestGPUDetect:
    """Tests for GPUInfo.detect()."""

    def test_cpu_mode_when_no_cuda(self) -> None:
        """Verify graceful CPU fallback when CUDA is unavailable."""
        gpu_info = GPUInfo()

        with patch("edgemind.core.gpu_utils._TORCH_AVAILABLE", True):
            with patch("edgemind.core.gpu_utils.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = False
                mock_torch.__version__ = "2.12.0"

                with patch("edgemind.core.gpu_utils._PSUTIL_AVAILABLE", False):
                    info = gpu_info.detect()

        assert info["backend"] == "cpu"
        assert info["is_blackwell"] is False
        assert info["full_support"] is False
        assert info["vram_gb"] == 0.0

    def test_blackwell_detection_sm120(self, mock_gpu_info: dict) -> None:
        """Verify sm_120 is correctly identified as Blackwell."""
        gpu_info = GPUInfo()

        with patch("edgemind.core.gpu_utils._TORCH_AVAILABLE", True):
            with patch("edgemind.core.gpu_utils.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 5090"
                mock_torch.cuda.get_device_capability.return_value = (12, 0)
                mock_props = MagicMock()
                mock_props.total_memory = 24 * 1_000_000_000
                mock_torch.cuda.get_device_properties.return_value = mock_props
                mock_torch.version.cuda = "13.0"
                mock_torch.__version__ = "2.12.0+cu130"
                mock_torch.backends.mps.is_available.return_value = False

                with patch("edgemind.core.gpu_utils._PSUTIL_AVAILABLE", False):
                    info = gpu_info.detect()

        assert info["is_blackwell"] is True
        assert info["compute_capability"] == "sm_120"
        assert info["full_support"] is True

    def test_non_blackwell_cuda_not_flagged(self) -> None:
        """RTX 4090 (sm_89) should not be flagged as Blackwell."""
        gpu_info = GPUInfo()

        with patch("edgemind.core.gpu_utils._TORCH_AVAILABLE", True):
            with patch("edgemind.core.gpu_utils.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
                mock_torch.cuda.get_device_capability.return_value = (8, 9)
                mock_props = MagicMock()
                mock_props.total_memory = 24 * 1_000_000_000
                mock_torch.cuda.get_device_properties.return_value = mock_props
                mock_torch.version.cuda = "12.1"
                mock_torch.__version__ = "2.3.0+cu121"
                mock_torch.backends.mps.is_available.return_value = False

                with patch("edgemind.core.gpu_utils._PSUTIL_AVAILABLE", False):
                    info = gpu_info.detect()

        assert info["is_blackwell"] is False
        assert info["compute_capability"] == "sm_89"


class TestVRAMEstimation:
    """Tests for GPUInfo.estimate_vram_requirement()."""

    def test_vram_estimate_bfloat16(self) -> None:
        """BF16: 7B model should require approximately 16.1 GB (with overhead)."""
        gpu_info = GPUInfo()
        result = gpu_info.estimate_vram_requirement(7.0, QuantizationMethod.BF16)

        # 7B * 2.0 bytes = 14GB base, +20% overhead = ~16.8GB
        assert 14.0 < result < 18.0, f"Expected ~16.1 GB, got {result}"

    def test_vram_estimate_int4(self) -> None:
        """INT4 NF4: 7B model should require approximately 4 GB."""
        gpu_info = GPUInfo()
        result = gpu_info.estimate_vram_requirement(7.0, QuantizationMethod.INT4_NF4)

        # 7B * 0.5 bytes = 3.5GB base, +20% overhead = ~4.2GB
        assert 3.0 < result < 5.5, f"Expected ~4.0 GB, got {result}"

    def test_vram_estimate_int8(self) -> None:
        """INT8: 7B model should require approximately 8 GB."""
        gpu_info = GPUInfo()
        result = gpu_info.estimate_vram_requirement(7.0, QuantizationMethod.INT8)

        # 7B * 1.0 bytes = 7GB base, +20% overhead = ~8.4GB
        assert 6.5 < result < 10.0, f"Expected ~8.1 GB, got {result}"

    def test_vram_estimate_gguf_q4(self) -> None:
        """GGUF Q4_K_M: overhead > INT4_NF4 due to K-quant format."""
        gpu_info = GPUInfo()
        q4 = gpu_info.estimate_vram_requirement(7.0, QuantizationMethod.GGUF_Q4KM)
        nf4 = gpu_info.estimate_vram_requirement(7.0, QuantizationMethod.INT4_NF4)

        # Q4_K_M uses 0.55 bytes/param vs 0.5 for NF4
        assert q4 > nf4

    def test_vram_estimate_scaling(self) -> None:
        """VRAM requirement should scale linearly with parameter count."""
        gpu_info = GPUInfo()
        r7b = gpu_info.estimate_vram_requirement(7.0, QuantizationMethod.BF16)
        r14b = gpu_info.estimate_vram_requirement(14.0, QuantizationMethod.BF16)

        assert abs(r14b - 2 * r7b) < 0.5, "VRAM should scale linearly with params"

    def test_check_fits_in_vram_true(self) -> None:
        """7B INT4 should fit in 24GB RTX 5090 VRAM."""
        gpu_info = GPUInfo()
        fits, required = gpu_info.check_fits_in_vram(7.0, QuantizationMethod.INT4_NF4, 24.0)
        assert fits is True
        assert required < 24.0

    def test_check_fits_in_vram_false(self) -> None:
        """70B BF16 should NOT fit in 24GB VRAM."""
        gpu_info = GPUInfo()
        fits, required = gpu_info.check_fits_in_vram(70.0, QuantizationMethod.BF16, 24.0)
        assert fits is False
        assert required > 24.0


class TestSetupGuide:
    """Tests for GPUInfo.print_setup_guide()."""

    def test_print_setup_guide_runs_without_error(
        self, mock_gpu_info: dict, capsys: pytest.CaptureFixture
    ) -> None:
        """Setup guide should print without raising exceptions."""
        gpu_info = GPUInfo()
        # Mark as needing setup (is_blackwell but no full_support)
        mock_gpu_info["full_support"] = False
        gpu_info.print_setup_guide(mock_gpu_info)
        # Just verify it doesn't raise
