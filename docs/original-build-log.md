> **Historical log.** This is the original step-by-step record of the first install on the author's device (2026-11-05). Paths and the LAN IP were anonymized. The source code embedded below is the *old* version; the current code is in the repository root (`app.py`, `config.py`, `scripts/`). Start with `README.md` and `docs/blog.md`.

# Jetson Local AI: build, operation, scripts, and performance

Built and documented on 2026-11-05 for the author.

## Start here: after shutting down and turning on again

The **Jetson is the AI server**. A laptop on the network is just a browser client.

- If you restart only your laptop: reconnect to the same network and open **http://<JETSON-IP>:7860**. No installation or model startup is needed on that laptop.
- If you restart the Jetson: turn it on and **log in as your user**. Ollama starts at boot, and the web interface starts automatically with the user's login session. Then open the same URL from your laptop.
- On the Jetson itself, use **http://127.0.0.1:7860** or the **Jetson Local AI** desktop/application shortcut.
- The Jetson must remain powered on and awake. Shutting it down or suspending it makes the website unavailable.
- The models remain installed on disk; rebooting does not require downloading or compiling them again. The first request loads the selected model into memory.

Verified configuration: `ollama` is enabled as a system service; `local-ai` is enabled as a user service; `Linger=no`. Therefore the web interface is **not currently configured for unattended startup before user login**.

If automatic startup does not work, run these commands on the Jetson as your user:

```bash
bash $HOME/local-ai/scripts/start.sh
systemctl --user status local-ai --no-pager
systemctl status ollama --no-pager
```

Equivalent manual commands:

```bash
sudo systemctl start ollama
systemctl --user start local-ai
```

To restart the interface after editing its Python file:

```bash
systemctl --user restart local-ai
```

To stop the interface and unload chat:

```bash
bash $HOME/local-ai/scripts/stop.sh
```

Optional, **not performed during this installation**: to start the enabled user service at boot without logging in:

```bash
sudo loginctl enable-linger $USER
```

Undo that option with `sudo loginctl disable-linger $USER`.

### If the IP address changes

`<JETSON-IP>` is the address observed during setup, not a guaranteed permanent assignment. On the Jetson:

```bash
ip -4 -brief address
ip route
```

Use the address on the same subnet as your laptop, followed by `:7860`. The current LAN is `<your-subnet>`; ignore the loopback address `127.0.0.1` and the secondary link-local address `<link-local>` for normal LAN access. A DHCP reservation in your router can keep the address stable; none was configured here.

## Hardware and installed stack

| Component | Observed configuration |
| --- | --- |
| Device | NVIDIA Jetson Orin NX 16GB / Seeed reComputer image |
| CPU | 8 ARM Cortex-A78AE cores, aarch64 |
| Memory | Approximately 15 GiB usable shared CPU/GPU memory |
| Storage | 468 GB root filesystem; about 428 GB free at initial inspection |
| OS | Ubuntu 22.04.5 LTS |
| Jetson software | L4T R36.4.3, JetPack 6.2 image |
| CUDA | 12.6; GPU compute capability 8.7 / SM87 |
| Ollama | 0.40.14, already installed |
| Chat / vision | `huihui_ai/qwen3-vl-abliterated:4b-instruct`, about 3.3 GB download |
| Image generation | Stable Diffusion 1.5, `v1-5-pruned-emaonly.safetensors`, about 4.0 GiB |
| Image runtime | CUDA-enabled stable-diffusion.cpp |
| Runtime source commit | `79526930040ba361c78cb42e79610cc5bfdc301e` |
| Web UI | Gradio 6.31.0, Python virtual environment |
| HTTP client | requests 2.34.2 |
| Browser | Firefox ARM64 installed under `~/local-ai/firefox` |

Docker was not available on the shell PATH when checked. This installation uses Python, Ollama, systemd, and a native CUDA executable; it does not require Docker.

## How it works

```text
Laptop browser -> Jetson LAN IP:7860 -> Gradio app.py
                                      |-- Chat / uploaded images -> localhost:11434 -> Ollama -> Qwen3-VL
                                      `-- Create images -> unload Qwen -> sd-cli -> SD1.5 -> PNG
```

- The interface listens on `0.0.0.0:7860`; Ollama remains on `127.0.0.1:11434`.
- There is no login on the web interface. Devices able to reach port 7860 can use it. No router port forwarding or internet tunnel was configured.
- Chat requests stream their responses. Context is now set to 8192 tokens (increased from 4096 by the image-upload fix) and each answer is capped at 1024 generated tokens.
- Uploaded images are encoded and sent to the local vision model.
- The UI maintains conversation history in session state; there is no persistent conversation database.
- Chat is kept in memory for three minutes after use unless explicitly unloaded.
- Before image generation, the app asks Ollama to unload the chat model.
- Image generation runs as a separate process, then exits and releases its memory.
- The Gradio queue and a Python lock serialize chat/image work rather than running multiple GPU jobs at once. Multiple laptops share this queue.
- Images are 512×512, with 20 sampling steps by default and adjustable steps from 10 to 40. The seed defaults to 42.
- Image weights are converted to `f16` for inference. CLIP text conditioning is on CPU; diffusion and VAE run on CUDA.
- The app adds no content/prompt filter. Abliteration reduces chat refusals but does not guarantee zero refusals or correct answers.
- Gradio analytics are disabled in the service. Inference is local; model downloads and initial dependency installation required internet access.

## Files and locations

| Path | Purpose |
| --- | --- |
| `~/local-ai/app.py` | Actual web application |
| `~/local-ai/venv/` | Isolated Python packages |
| `~/local-ai/requirements.lock.txt` | Installed Python package versions captured during setup |
| `~/local-ai/models/sd15.safetensors` | Image model |
| `~/local-ai/stable-diffusion.cpp/` | Source code and CUDA build |
| `~/local-ai/stable-diffusion.cpp/build/bin/sd-cli` | Image-generation executable |
| `~/local-ai/outputs/` | Generated PNGs and a red vision test image |
| `~/local-ai/generation.log` | Most recent image-generation log; overwritten on each generation |
| `~/local-ai/evidence/` | Preserved original chat response and final f16 image-test log |
| `~/.config/systemd/user/local-ai.service` | Active user service |
| `~/.local/share/applications/jetson-local-ai.desktop` | Application shortcut |
| `~/Desktop/jetson-local-ai.desktop` | Desktop shortcut |
| `~/local-ai/scripts/` | Convenience scripts and copies of service/launcher configuration |

Ollama manages its model storage separately; use `ollama list` to inspect it. The `~/local-ai` directory occupied about 6.2 GB when documented, excluding Ollama's model storage.

## Build command record

This is a readable reconstruction of the meaningful commands executed during setup, grouped by purpose. Repeated progress polling is consolidated; it is **not a byte-for-byte terminal transcript**. Password entry is intentionally omitted. The exact final application and configuration files are embedded below. Do not rerun the installation sequence merely to start the app.

### 1. Inspect the device and existing software

```bash
uname -a
lscpu
free -h
df -h /
command -v ollama
command -v docker
nvidia-smi
cat /proc/device-tree/model /etc/nv_tegra_release /etc/os-release
ls /dev/nvmap /dev/nvhost-gpu /usr/local/cuda/bin/nvcc
ollama --version
ollama list
docker --version
docker ps
/usr/local/cuda/bin/nvcc --version
python3 -m pip --version
curl -sS --max-time 3 http://127.0.0.1:11434/api/version
```

The first restricted-session checks could not access GPU devices or localhost sockets. After session permissions changed, GPU and Ollama access worked. Docker was not found, and Python pip/venv tooling needed installation. The initial `nvidia-smi` error was not treated as proof of a broken driver; subsequent inference verified GPU operation.

### 2. Install build tools and Python environment

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-venv
mkdir -p $HOME/local-ai/models $HOME/local-ai/outputs
python3 -m venv $HOME/local-ai/venv
$HOME/local-ai/venv/bin/pip install gradio requests
$HOME/local-ai/venv/bin/pip freeze > $HOME/local-ai/requirements.lock.txt
```

For reconstructing the same Python environment later, prefer:

```bash
$HOME/local-ai/venv/bin/pip install -r $HOME/local-ai/requirements.lock.txt
```

### 3. Download models

```bash
ollama pull huihui_ai/qwen3-vl-abliterated:4b-instruct
curl -L --fail --retry 3 \
  -o $HOME/local-ai/models/sd15.safetensors \
  https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
```

Model tags and `main` download URLs can change over time. These commands record the sources used; they are not immutable model-version pins.

### 4. Compile the CUDA image engine

```bash
git clone --recursive https://github.com/leejet/stable-diffusion.cpp \
  $HOME/local-ai/stable-diffusion.cpp
cmake -S $HOME/local-ai/stable-diffusion.cpp \
  -B $HOME/local-ai/stable-diffusion.cpp/build \
  -DSD_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSD_CUDA_CUTLASS_PATH:PATH=
cmake --build $HOME/local-ai/stable-diffusion.cpp/build --target sd-cli -j 3
```

The original build was interrupted with SIGINT and resumed incrementally using six workers:

```bash
cmake --build $HOME/local-ai/stable-diffusion.cpp/build --target sd-cli -j 6
```

The initial build took many minutes on the Jetson; this is a one-time setup cost, not a per-image cost. The architecture option limits the build to this Jetson GPU. The empty CUTLASS path disables the optional CUTLASS integration.

For a fresh checkout matching this build, checkout the commit listed above and run `git submodule update --init --recursive` before compiling. The provided `scripts/rebuild-image-engine.sh` uses the existing checkout and does not update source code.

### 5. Create the app and enable its service

`app.py` was written as the Gradio application embedded later in this document. It initially showed the latest answer in Markdown; it was then upgraded to a conversation view and tested for multi-turn memory. Image generation was first tested with default f32 weights and then changed to `--type f16`.

The service file below was written to `~/.config/systemd/user/local-ai.service`, then activated:

```bash
mkdir -p $HOME/.config/systemd/user
systemctl --user daemon-reload
systemctl --user enable --now local-ai
systemctl --user restart local-ai
systemctl --user status local-ai --no-pager
```

### 6. Install and open a local browser

The desktop's default browser setting referred to Firefox, but its executable was absent. An ARM64 Firefox archive was installed locally:

```bash
curl -L --fail -o /tmp/local-ai-firefox.tar.xz \
  'https://download.mozilla.org/?product=firefox-latest-ssl&os=linux64-aarch64&lang=en-US'
tar -xf /tmp/local-ai-firefox.tar.xz -C $HOME/local-ai
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  $HOME/local-ai/firefox/firefox http://127.0.0.1:7860
```

The launcher below was written to `~/.local/share/applications/jetson-local-ai.desktop`, then copied to the desktop:

```bash
cp $HOME/.local/share/applications/jetson-local-ai.desktop $HOME/Desktop/jetson-local-ai.desktop
chmod +x $HOME/Desktop/jetson-local-ai.desktop
gio set $HOME/Desktop/jetson-local-ai.desktop metadata::trusted true
```

A browser rendering check also used a separate temporary Firefox profile:

```bash
mkdir -p /tmp/local-ai-firefox-check
$HOME/local-ai/firefox/firefox --headless --no-remote \
  --profile /tmp/local-ai-firefox-check \
  --screenshot $HOME/local-ai/interface.png --window-size 1280,1000 \
  http://127.0.0.1:7860
```

`interface.png` captures the earlier version before the full conversation-view improvement. A desktop screenshot attempt returned a black image; it was not used as evidence that the interface rendered. The separate browser rendering check succeeded.

### 7. Enable access from other laptops

The final `launch` call was changed from `server_name='127.0.0.1'` to `server_name='0.0.0.0'` in `app.py`.

```bash
systemctl --user restart local-ai
ip -4 -brief address
ip route
ss -ltnp | rg '7860|11434'
curl -s -o /dev/null -w '%{http_code}\n' http://<JETSON-IP>:7860
```

Verified: port 7860 listened on `0.0.0.0`, HTTP returned `200`, and the service was active. This HTTP check was performed on the Jetson using its LAN address; a separate laptop connection was not independently tested. `sudo ufw status` reported that UFW was not installed; no firewall rule was changed.

### 8. Tests and diagnostic commands

The initial text test was:

```bash
curl -sS http://localhost:11434/api/chat -d '{"model":"huihui_ai/qwen3-vl-abliterated:4b-instruct","messages":[{"role":"user","content":"Reply with just: local chat works"}],"stream":false,"options":{"num_ctx":4096,"num_predict":30}}'
ollama ps
ollama stop huihui_ai/qwen3-vl-abliterated:4b-instruct
```

Python tests created a 128×128 red PNG with Pillow, submitted it to Ollama and the Gradio API, checked conversation memory using the word `apple`, and generated the sailboat image. Their reusable equivalent is `scripts/smoke-test.py`, embedded below. That consolidated script was written while documenting; its constituent checks were run during installation.

The executed image-generation arguments, now assembled by the application, are equivalent to:

```bash
$HOME/local-ai/stable-diffusion.cpp/build/bin/sd-cli \
  -m $HOME/local-ai/models/sd15.safetensors \
  -p 'a watercolor painting of a small red sailboat on a calm blue lake, mountains, sunny day' \
  -n 'blurry, low quality' -W 512 -H 512 --steps 20 -s 42 \
  -o $HOME/local-ai/outputs/example.png --clip-on-cpu --type f16
```

The first image test omitted `--type f16`; the final test included it. Actual test outputs used UUID filenames to avoid overwriting existing images.

Progress inspection used `tail` on `/tmp/local-ai-{build,pull,download,pip,clone}.log`, `free -h`, `ps`, `ollama ps`, and service status checks. Temporary logs may disappear after reboot. Durable test evidence was copied to `~/local-ai/evidence/`.

## Performance review

These are installation-time observations, not a controlled benchmark suite. Some chat tests ran while compilation was active. No sustained multi-user load, thermal throttling, energy consumption, or long-context benchmark was performed.

| Test | Observed result | Interpretation |
| --- | --- | --- |
| Cold chat request | 6.73 s total, 6.08 s loading, 0.196 s prompt processing, 0.445 s generation | Most first-request latency was model loading |
| Short chat decode | 4 output tokens / 0.445 s, about 9 tokens/s | Too short a sample to predict sustained speed |
| Chat GPU placement | `ollama ps`: 100% GPU, 3.6 GB, 4096-token context | Confirms GPU offload; 3.6 GB is Ollama's reported loaded size, not whole-system peak RAM |
| Vision test | Red 128×128 image identified correctly; 7.67 s total, 7.49 s prompt processing | Basic image upload/vision path works; not a broad vision-quality evaluation |
| Multi-turn chat | Correctly remembered `apple` in the next turn | Session conversation state works |
| SD1.5 f32 | 512×512, 20 steps, seed 42; 50.2 s through web API, 48.87 s engine time | Functional GPU generation with original weight precision |
| SD1.5 f16 | Same prompt/settings; 45.2 s through web API, 43.99 s engine time | Final default; roughly 10% lower elapsed time in these two individual runs |
| Model switching | `ollama ps` empty after generation; image process exited | Models released their resident runtime allocations after use |
| LAN endpoint | HTTP 200 on `http://<JETSON-IP>:7860` | Verified from host; separate-laptop network path not benchmarked |

### Memory and precision

The Jetson uses shared memory, not separate 16 GB CPU RAM plus 16 GB GPU VRAM. CPU applications, browser, model weights, and GPU working buffers compete for the same pool.

The image engine's allocation plan reported:

| Component parameters | f32 run | f16 run |
| --- | ---: | ---: |
| Diffusion network | 3278 MiB | 1644 MiB |
| CLIP conditioner | 469 MiB | 307 MiB |
| VAE | 319 MiB | 159 MiB |

These are parameter estimates from the engine's plan, **not measured total or peak usage**; compute buffers and other process memory are additional. Source weights remain f32 on disk and are converted for inference. Automatic unloading is useful on this shared-memory device.

### Image quality and practical limits

Both saved samples are valid 512×512 watercolor-like scenes with a red boat, lake, and mountains. The result resembled a motorboat more than the requested small sailboat, showing that prompt adherence is imperfect. Stable Diffusion 1.5 is a compact practical baseline, not a modern high-end image-quality model. No typography, face, anatomy, or editing benchmark was performed.

Measured samples:

- f32: `outputs/5d827e0f5b29408088eee3bbac958405.png`
- f16: `outputs/e8caeacc844f44f195e31e3143a24670.png`

The approximately 10% speed difference is indicative only: one run per precision, different cache state, and no controlled repeated trials. For everyday use, expect roughly a minute for a 20-step image under similar conditions, with variation from other workloads and power/thermal settings. Longer chats, larger uploaded images, and additional clients can increase latency. Jobs queue rather than scale across laptops.

### Overall assessment

The 4B vision model plus SD1.5 is a workable combination for this 16 GB Jetson when run sequentially. Chat, uploaded-image recognition, multi-turn state, GPU image generation, and unloading all passed functional checks. The main tradeoffs are model load latency, limited image quality/prompt accuracy, and serialized multi-user throughput. No power-mode or clock tuning was applied.

## Troubleshooting and maintenance

```bash
# Web service and logs (run as your user, without sudo)
systemctl --user status local-ai --no-pager
journalctl --user -u local-ai -n 100 --no-pager

# Ollama and models
systemctl status ollama --no-pager
ollama list
ollama ps

# Image log, memory, listening port
tail -n 50 $HOME/local-ai/generation.log
free -h
ss -ltnp | rg ':7860'

# Test server locally and over its LAN address
curl -I http://127.0.0.1:7860
curl -I http://<JETSON-IP>:7860
```

If localhost works but your laptop cannot connect, check the Jetson's current IP, whether both devices are on the same LAN, guest-Wi-Fi/client isolation, VPN routing, and network firewall rules. Do not reinstall the models to fix a network issue.

If port 7860 is already occupied, do not launch a second copy of `app.py`; manage the existing service. If the image engine fails, inspect `generation.log`. If resources are tight, close other GPU applications and use the Unload chat model button.

Optional repeat test:

```bash
$HOME/local-ai/venv/bin/python $HOME/local-ai/scripts/smoke-test.py
```

This creates another sample image and uses the shared GPU queue. Chat history is not saved across service restarts. Model files and generated PNGs persist.

## Sources

- Chat model: https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated
- Image model: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- Image runtime/build instructions: https://github.com/leejet/stable-diffusion.cpp
- Firefox installer: https://download.mozilla.org/

## Exact final source and scripts

The following snapshots were copied from the actual installed files when this guide was written. The helper scripts are provided for future operation; writing them did not rerun installation, stop the service, or generate additional images.

### Web application: `app.py`

```python
import base64, io, json, subprocess, uuid, threading
from PIL import Image, ImageOps
from pathlib import Path
import gradio as gr
import requests
ROOT=Path(__file__).parent
MODEL='huihui_ai/qwen3-vl-abliterated:4b-instruct'
API='http://127.0.0.1:11434/api'
lock=threading.Lock()
def unload():
    r=requests.post(API+'/generate',json={'model':MODEL,'keep_alive':0},timeout=120)
    r.raise_for_status()
    return 'Chat model unloaded. Memory is available.'
def encode_image(picture):
    # Normalize phone photos (including MPO), orientation and color mode.
    with Image.open(picture) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
    return base64.b64encode(buffer.getvalue()).decode()

def chat(prompt, picture, history):
    if not prompt.strip():
        raise gr.Error('Enter a message first.')
    history=list(history or [])
    msg={'role':'user','content':prompt}
    if picture:
        try:
            encoded = encode_image(picture)
        except (OSError, ValueError) as exc:
            raise gr.Error('This image could not be read. Try a JPEG or PNG file.') from exc
        # The upload stays selected between messages; do not duplicate its tokens.
        previous = next((m['images'][0] for m in reversed(history) if m.get('images')), None)
        if encoded != previous:
            msg['images']=[encoded]
    messages=history+[msg]
    with lock:
        try:
            while True:
                response=requests.post(API+'/chat',json={'model':MODEL,'messages':messages,'stream':True,'keep_alive':'3m','options':{'num_ctx':8192,'num_predict':1024}},stream=True,timeout=(10,600))
                if response.ok:
                    break
                try:
                    detail=response.json().get('error',response.text)
                except ValueError:
                    detail=response.text
                response.close()
                if response.status_code == 400 and 'context' in detail.lower() and len(messages)>1:
                    # Keep recent complete turns; preserve the most recent image
                    # if it would otherwise disappear from the retained context.
                    last_image=next((m['images'] for m in reversed(messages) if m.get('images')),None)
                    messages=messages[2:] if len(messages)>2 else messages[-1:]
                    if last_image and not any(m.get('images') for m in messages):
                        messages[0]=dict(messages[0],images=last_image)
                    continue
                raise gr.Error('The model could not process this request: '+detail[:400])
            answer=''
            with response:
                for line in response.iter_lines():
                    if line:
                        item=json.loads(line)
                        if 'error' in item: raise gr.Error(item['error'])
                        answer+=item.get('message',{}).get('content','')
                        yield [{'role':m['role'],'content':m['content']} for m in messages]+[{'role':'assistant','content':answer}], history
        except requests.RequestException as exc:
            raise gr.Error('Cannot reach the local model. Please retry; if it persists, check the Ollama service.') from exc
    messages.append({'role':'assistant','content':answer})
    yield [{'role':m['role'],'content':m['content']} for m in messages],messages

def generate(prompt, negative, steps, seed):
    if not prompt.strip(): raise gr.Error('Describe the image first.')
    output=ROOT/'outputs'/f'{uuid.uuid4().hex}.png'
    args=[str(ROOT/'stable-diffusion.cpp/build/bin/sd-cli'),'-m',str(ROOT/'models/sd15.safetensors'),'-p',prompt,'-n',negative,'-W','512','-H','512','--steps',str(int(steps)),'-s',str(int(seed)),'-o',str(output),'--clip-on-cpu','--type','f16']
    with lock:
        unload()
        with (ROOT/'generation.log').open('w') as log:
            result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        if result.returncode: raise gr.Error('Generation failed. See ~/local-ai/generation.log.')
    if not output.exists():
        matches=list(output.parent.glob(output.stem+'*.png'))
        if not matches: raise gr.Error('No output image was produced.')
        output=matches[0]
    return str(output)

with gr.Blocks(title='Jetson Local AI') as app:
    gr.Markdown('# Jetson Local AI\nPrivate chat, image understanding, and image creation on your device.')
    with gr.Tab('Chat & image understanding'):
        history=gr.State([])
        prompt=gr.Textbox(label='Your message',lines=3)
        picture=gr.Image(type='filepath',label='Optional image to analyze',height=180,width=280,sources=['upload'],elem_id='chat-image')
        send=gr.Button('Send',variant='primary')
        answer=gr.Chatbot(label='Conversation',height=400)
        send.click(chat,[prompt,picture,history],[answer,history],concurrency_id='gpu',concurrency_limit=1)
        gr.Button('New conversation').click(lambda: ([],[]),outputs=[answer,history])
        status=gr.Textbox(label='Memory status',interactive=False)
        gr.Button('Unload chat model').click(unload,outputs=status,concurrency_id='gpu')
    with gr.Tab('Create images'):
        gr.Markdown('Stable Diffusion 1.5 · 512 × 512. Chat unloads automatically before generation. The image model unloads when finished.')
        desc=gr.Textbox(label='Describe your image',lines=3)
        neg=gr.Textbox(label='Negative prompt',value='blurry, low quality, distorted')
        steps=gr.Slider(10,40,value=20,step=1,label='Steps')
        seed=gr.Number(value=42,precision=0,label='Seed')
        go=gr.Button('Generate image',variant='primary')
        result=gr.Image(label='Generated image')
        go.click(generate,[desc,neg,steps,seed],result,concurrency_id='gpu',concurrency_limit=1)
app.queue().launch(css='#chat-image {max-width: 280px !important;} #chat-image img {object-fit: contain !important;}',server_name='0.0.0.0',server_port=7860,inbrowser=False)
```

### Enabled user service: `scripts/local-ai.service`

```ini
[Unit]
Description=Jetson Local AI web interface
After=network.target
[Service]
WorkingDirectory=$HOME/local-ai
ExecStart=$HOME/local-ai/venv/bin/python $HOME/local-ai/app.py
Restart=on-failure
RestartSec=5
Environment=GRADIO_ANALYTICS_ENABLED=False
[Install]
WantedBy=default.target
```

### Desktop launcher: `scripts/jetson-local-ai.desktop`

```ini
[Desktop Entry]
Type=Application
Name=Jetson Local AI
Comment=Local chat, vision, and image generation
Exec=$HOME/local-ai/firefox/firefox http://127.0.0.1:7860
Icon=applications-graphics
Terminal=false
Categories=Utility;
```

### Start helper: `scripts/start.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
if ! systemctl is-active --quiet ollama; then
  sudo systemctl start ollama
fi
systemctl --user start local-ai
printf 'Local: http://127.0.0.1:7860\nLAN addresses:\n'
ip -4 -brief address
printf 'Use http://<LAN-IP>:7860 from your laptop.\n'
```

### Stop helper: `scripts/stop.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
systemctl --user stop local-ai
ollama stop huihui_ai/qwen3-vl-abliterated:4b-instruct
```

### Rebuild image engine: `scripts/rebuild-image-engine.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
AI_ROOT=$HOME/local-ai
cmake -S "$AI_ROOT/stable-diffusion.cpp" -B "$AI_ROOT/stable-diffusion.cpp/build" \
  -DSD_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSD_CUDA_CUTLASS_PATH:PATH=
cmake --build "$AI_ROOT/stable-diffusion.cpp/build" --target sd-cli -j 6
```

### Reusable functional checks: `scripts/smoke-test.py`

```python
"""Optional functional test. Runs one image generation; takes about a minute."""
from pathlib import Path
import time
from PIL import Image
from gradio_client import Client, handle_file
root = Path('$HOME/local-ai')
image = root / 'outputs/vision-test.png'
Image.new('RGB', (128, 128), (255, 0, 0)).save(image)
c = Client('http://127.0.0.1:7860', verbose=False)
print(c.predict('What color is this image?', handle_file(str(image)), api_name='/chat'))
print(c.predict('Remember the word apple. Say OK.', None, api_name='/chat'))
print(c.predict('What word did I ask you to remember?', None, api_name='/chat'))
started = time.monotonic()
print(c.predict('a watercolor painting of a small red sailboat on a calm blue lake, mountains, sunny day', 'blurry, low quality', 20, 42, api_name='/generate'))
print('Image generation seconds:', round(time.monotonic() - started, 1))
print(c.predict(api_name='/unload'))
```

## Image-upload fix (2026-11-05)

A real 1704×2560 upload reproduced an Ollama HTTP 400: the request needed 4255 tokens but the previous context limit was 4096. The fix normalizes uploaded images to RGB JPEG, respects EXIF orientation, and limits the analysis copy to 768 pixels on the longest edge. Original files are not changed. Very small details may be lost by this resizing. Phone MPO images are normalized too.

Chat now uses an 8192-token context. An unchanged selected image is not attached repeatedly. If conversation history still exceeds the context, the oldest turns are dropped and the latest image is preserved for follow-up questions. Earlier trimmed text is no longer available to the model. Backend failures now show an explanatory message instead of a generic HTTP error.

The upload component is limited to 280×180 pixels and uses a contained preview. Reload the web page after the service update. The original performance measurements above predate this change; the new loaded model reports approximately 4.3 GB with an 8192-token context, still 100% GPU.
