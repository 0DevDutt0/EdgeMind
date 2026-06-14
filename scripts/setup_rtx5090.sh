#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# EdgeMind — RTX 5090 (Blackwell, sm_120) Setup Script
# ═══════════════════════════════════════════════════════════════════════════════
# The RTX 5090 uses Compute Capability 12.0 (sm_120).
# Standard pip install torch does NOT support sm_120.
# This script installs the correct nightly build with CUDA 13 support.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  EdgeMind — RTX 5090 (Blackwell sm_120) Setup               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Remove any existing PyTorch installation
echo "[1/4] Removing existing PyTorch installation..."
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Step 2: Install PyTorch nightly with CUDA 13 (required for sm_120)
echo "[2/4] Installing PyTorch nightly with CUDA 13.0 (sm_120 support)..."
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Step 3: Verify GPU detection
echo "[3/4] Verifying RTX 5090 detection..."
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    cuda = torch.version.cuda
    print(f'  GPU: {name}')
    print(f'  Compute Capability: sm_{cc[0]}{cc[1]}')
    print(f'  CUDA: {cuda}')
    if cc == (12, 0):
        print('  ✓ RTX 5090 (Blackwell sm_120) detected and supported!')
    else:
        print(f'  ⚠ Detected sm_{cc[0]}{cc[1]}, expected sm_120 for RTX 5090')
else:
    print('  ✗ No CUDA GPU detected')
"

# Step 4: Install EdgeMind
echo "[4/4] Installing EdgeMind..."
pip install -e .

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✓ Setup complete! Run: edgemind info                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
