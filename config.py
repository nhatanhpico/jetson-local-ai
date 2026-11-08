"""Central settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CHAT_MODEL = os.environ.get('LOCAL_AI_CHAT_MODEL', 'huihui_ai/qwen3-vl-abliterated:4b-instruct')
OLLAMA_API = os.environ.get('LOCAL_AI_OLLAMA_API', 'http://127.0.0.1:11434/api')
NUM_CTX = int(os.environ.get('LOCAL_AI_NUM_CTX', 8192))
NUM_PREDICT = int(os.environ.get('LOCAL_AI_NUM_PREDICT', 1024))
KEEP_ALIVE = os.environ.get('LOCAL_AI_KEEP_ALIVE', '3m')

# Safe default: only this device. Set LOCAL_AI_HOST=0.0.0.0 to allow the LAN (no login is provided!).
HOST = os.environ.get('LOCAL_AI_HOST', '127.0.0.1')
PORT = int(os.environ.get('LOCAL_AI_PORT', 7860))

# Optional login for LAN use: LOCAL_AI_AUTH="user:password" (plain HTTP, so this only deters casual use).
_auth = os.environ.get('LOCAL_AI_AUTH', '')
AUTH = tuple(_auth.split(':', 1)) if ':' in _auth else None
# How many requests may wait in line for the GPU before new ones are refused (friendly "busy" message).
MAX_QUEUE = int(os.environ.get('LOCAL_AI_MAX_QUEUE', 20))

SD_BIN = Path(os.environ.get('LOCAL_AI_SD_BIN', ROOT / 'stable-diffusion.cpp/build/bin/sd-cli'))
SD_MODEL = Path(os.environ.get('LOCAL_AI_SD_MODEL', ROOT / 'models/sd15.safetensors'))
OUTPUT_DIR = ROOT / 'outputs'
GEN_LOG = ROOT / 'generation.log'
SD_TIMEOUT = int(os.environ.get('LOCAL_AI_SD_TIMEOUT', 1200))


def sd_command(prompt, negative, steps, seed, output):
    """Command line for stable-diffusion.cpp: 512x512, f16 weights, CLIP on CPU, diffusion+VAE on CUDA."""
    return [str(SD_BIN), '-m', str(SD_MODEL), '-p', prompt, '-n', negative, '-W', '512', '-H', '512',
            '--steps', str(int(steps)), '-s', str(int(seed)), '-o', str(output), '--clip-on-cpu', '--type', 'f16']
