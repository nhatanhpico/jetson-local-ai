#!/usr/bin/env bash
# Build stable-diffusion.cpp with CUDA for Jetson Orin (compute capability 8.7).
set -euo pipefail
AI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SD_COMMIT=79526930040ba361c78cb42e79610cc5bfdc301e   # commit this project was tested with
if [ ! -d "$AI_ROOT/stable-diffusion.cpp/.git" ]; then
  git clone --recursive https://github.com/leejet/stable-diffusion.cpp "$AI_ROOT/stable-diffusion.cpp"
  git -C "$AI_ROOT/stable-diffusion.cpp" checkout "$SD_COMMIT"
  git -C "$AI_ROOT/stable-diffusion.cpp" submodule update --init --recursive
fi
cmake -S "$AI_ROOT/stable-diffusion.cpp" -B "$AI_ROOT/stable-diffusion.cpp/build" \
  -DSD_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSD_CUDA_CUTLASS_PATH:PATH=
cmake --build "$AI_ROOT/stable-diffusion.cpp/build" --target sd-cli -j 6
echo "Built: $AI_ROOT/stable-diffusion.cpp/build/bin/sd-cli"
