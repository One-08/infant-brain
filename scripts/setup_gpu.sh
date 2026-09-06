#!/bin/bash
# Setup script for cloud GPU (Lambda Labs, Vast.ai, RunPod, etc.)
#
# Usage:
#   git clone <your-repo> infant-brain && cd infant-brain
#   bash scripts/setup_gpu.sh

set -e

echo "=== Setting up Infant Brain on GPU ==="

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project deps
pip install -e ".[all]"

# Install Atari ROMs
AutoROM --accept-license 2>/dev/null || echo "ROMs may already be installed"

# Verify
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import gymnasium as gym
import ale_py
env = gym.make('ALE/Pong-v5')
print(f'Atari: OK ({env.action_space.n} actions)')
env.close()
print('\\n=== Setup complete! ===')
"
