#!/usr/bin/env python3
"""Benchmark the Jetson Local AI stack and write docs/benchmark-results.json.

Measures, using the same code paths as the web app:
  * chat: cold load time, prompt-eval and generation tokens/s (from Ollama's own counters)
  * vision: time to describe an image
  * image generation: wall time for 512x512 SD1.5 at several step counts
  * system: peak RAM, peak GPU load and peak board power sampled with tegrastats

Usage:  venv/bin/python scripts/benchmark.py
Needs the Ollama service running. Stops the web app's GPU use while it runs, so
do not use the UI at the same time.
"""
import base64, io, json, re, statistics, subprocess, sys, threading, time, uuid
from pathlib import Path
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

API = config.OLLAMA_API
MODEL = config.CHAT_MODEL


class Tegra:
    T0 = time.monotonic()
    """Sample tegrastats in the background and keep peaks."""
    phase = ''
    trace = []  # (seconds since start, phase, ram_mb, gpu_pct, power_w) across all runs

    def __init__(self):
        self.samples = []
        self.proc = None
        self.t0 = time.monotonic()

    def __enter__(self):
        self.proc = subprocess.Popen(['tegrastats', '--interval', '500'], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            ram = re.search(r'RAM (\d+)/(\d+)MB', line)
            gpu = re.search(r'GR3D_FREQ (\d+)%', line)
            pwr = re.search(r'VDD_IN (\d+)mW', line)
            tj = re.search(r'tj@([\d.]+)C', line)
            if ram and gpu and pwr:
                self.samples.append((int(ram.group(1)), int(gpu.group(1)), int(pwr.group(1)),
                                     float(tj.group(1)) if tj else 0.0))
                Tegra.trace.append((round(time.monotonic() - Tegra.T0, 1), Tegra.phase, int(ram.group(1)),
                                    int(gpu.group(1)), round(int(pwr.group(1)) / 1000, 2)))

    def __exit__(self, *a):
        self.proc.terminate()

    def summary(self):
        if not self.samples:
            return {}
        ram, gpu, pwr, tj = zip(*self.samples)
        return {'peak_ram_mb': max(ram), 'peak_gpu_pct': max(gpu), 'avg_gpu_pct': round(statistics.mean(gpu)),
                'peak_power_w': round(max(pwr) / 1000, 1), 'avg_power_w': round(statistics.mean(pwr) / 1000, 1),
                'peak_temp_c': max(tj)}


def chat(prompt, image_b64=None, keep_alive='3m', num_predict=256):
    msg = {'role': 'user', 'content': prompt}
    if image_b64:
        msg['images'] = [image_b64]
    t0 = time.monotonic()
    r = requests.post(API + '/chat', json={'model': MODEL, 'messages': [msg], 'stream': False, 'keep_alive': keep_alive,
                                           'options': {'num_ctx': config.NUM_CTX, 'num_predict': num_predict}}, timeout=600)
    r.raise_for_status()
    d = r.json()
    wall = time.monotonic() - t0
    ns = 1e9
    return {'wall_s': round(wall, 2), 'load_s': round(d.get('load_duration', 0) / ns, 2),
            'prompt_tokens': d.get('prompt_eval_count'),
            'prompt_tok_s': round(d['prompt_eval_count'] / (d['prompt_eval_duration'] / ns), 1) if d.get('prompt_eval_duration') else None,
            'gen_tokens': d.get('eval_count'),
            'gen_tok_s': round(d['eval_count'] / (d['eval_duration'] / ns), 1) if d.get('eval_duration') else None,
            'answer': d['message']['content'][:160]}


def test_image():
    img = Image.new('RGB', (512, 384), (30, 60, 120))
    d = ImageDraw.Draw(img)
    d.ellipse((60, 60, 220, 220), fill=(220, 40, 40))
    d.rectangle((280, 120, 460, 300), fill=(240, 200, 40))
    buf = io.BytesIO()
    img.save(buf, 'JPEG')
    return base64.b64encode(buf.getvalue()).decode()


def sd(steps, seed=42):
    out = ROOT / 'outputs' / f'bench-{uuid.uuid4().hex[:8]}.png'
    cmd = config.sd_command('a watercolor painting of a small red sailboat on a calm blue lake, mountains, sunny day',
                            'blurry, low quality', steps, seed, out)
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    wall = time.monotonic() - t0
    if p.returncode:
        raise SystemExit('sd-cli failed:\n' + p.stdout[-800:] + p.stderr[-800:])
    sampling = re.findall(r'(\d+\.\d+)s/it', p.stdout + p.stderr)
    return {'steps': steps, 'wall_s': round(wall, 1), 'sec_per_step': float(sampling[-1]) if sampling else None,
            'file': out.name}


def main():
    results = {'model': MODEL, 'date': time.strftime('%Y-%m-%d %H:%M:%S')}
    requests.post(API + '/generate', json={'model': MODEL, 'keep_alive': 0}, timeout=120)
    time.sleep(2)
    with Tegra() as t:
        Tegra.phase = 'chat'
        print('chat (cold load)...')
        results['chat_cold'] = chat('Explain in three sentences what a Jetson Orin NX is.')
        print(results['chat_cold'])
        print('chat (warm) x3...')
        warm = [chat('Write a short paragraph about why on-device AI protects privacy.', num_predict=200) for _ in range(3)]
        results['chat_warm'] = warm
        results['chat_warm_avg_gen_tok_s'] = round(statistics.mean(w['gen_tok_s'] for w in warm), 1)
        print('warm avg tok/s', results['chat_warm_avg_gen_tok_s'])
        Tegra.phase = 'vision'
        print('vision...')
        results['vision'] = chat('Describe the shapes and colors in this image.', test_image())
        print(results['vision'])
    results['chat_system'] = t.summary()
    requests.post(API + '/generate', json={'model': MODEL, 'keep_alive': 0}, timeout=120)
    time.sleep(3)
    results['image'] = []
    with Tegra() as t:
        for steps in (10, 20, 30):
            Tegra.phase = f'image-{steps}'
            print(f'image {steps} steps...')
            results['image'].append(sd(steps))
            print(results['image'][-1])
    results['image_system'] = t.summary()
    Tegra.phase = ''
    trace = ROOT / 'docs' / 'tegrastats-trace.csv'
    trace.parent.mkdir(exist_ok=True)
    trace.write_text('seconds,phase,ram_mb,gpu_pct,power_w\n' + '\n'.join(','.join(map(str, r)) for r in Tegra.trace) + '\n')
    out = ROOT / 'docs' / 'benchmark-results.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k.endswith('system')}, indent=2))
    print('wrote', out)


if __name__ == '__main__':
    main()
