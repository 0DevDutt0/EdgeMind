#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# EdgeMind — CPU-Only / CI Setup Script
# ═══════════════════════════════════════════════════════════════════════════════
# Installs standard PyTorch (CPU-only) and EdgeMind for non-GPU environments.
# Use this for: CI/CD pipelines, development without GPU, Raspberry Pi, Mac.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  EdgeMind — CPU-Only Setup                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Note: GPU-dependent features (quantization with bitsandbytes, GPTQ, AWQ)"
echo "      are not available in CPU mode. GGUF inference via llama.cpp works."
echo ""

# Step 1: Install CPU PyTorch
echo "[1/3] Installing PyTorch (CPU only)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install EdgeMind
echo "[2/3] Installing EdgeMind..."
pip install -e ".[dev]"

# Step 3: Verify
echo "[3/3] Verifying installation..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('  ✓ CPU-only mode confirmed')
"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ CPU setup complete!                                       ║"
echo "║  Run:  edgemind info                                         ║"
echo "║  Test: pytest tests/ -v --tb=short                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
