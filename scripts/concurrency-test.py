#!/usr/bin/env python3
"""How many devices can use the server at once? Simulates N simultaneous clients.

  1. N simultaneous page loads (HTTP)          -> how many viewers can have the UI open
  2. N simultaneous short chat requests        -> how the GPU queue serializes them

Usage: venv/bin/python scripts/concurrency-test.py [N_PAGES] [N_CHATS]
"""
import sys, time, threading, statistics, requests
from gradio_client import Client
URL = 'http://127.0.0.1:7860'
n_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 50
n_chats = int(sys.argv[2]) if len(sys.argv) > 2 else 6

def load(i, out):
    t = time.monotonic()
    try:
        r = requests.get(URL, timeout=60)
        out[i] = (r.status_code, time.monotonic() - t)
    except Exception as e:
        out[i] = (type(e).__name__, time.monotonic() - t)

out = {}
ts = [threading.Thread(target=load, args=(i, out)) for i in range(n_pages)]
t0 = time.monotonic(); [t.start() for t in ts]; [t.join() for t in ts]
ok = sum(1 for s, _ in out.values() if s == 200)
lat = [d for _, d in out.values()]
print(f'pages: {ok}/{n_pages} OK in {time.monotonic()-t0:.1f}s, median {statistics.median(lat):.2f}s, max {max(lat):.2f}s')

res = {}
def chat(i):
    c = Client(URL, verbose=False)            # a separate session, like a separate device
    t = time.monotonic()
    try:
        c.predict('Reply with exactly one short sentence about the number %d.' % i, None, api_name='/chat')
        res[i] = ('ok', time.monotonic() - t)
    except Exception as e:
        res[i] = (str(e)[:60], time.monotonic() - t)
ts = [threading.Thread(target=chat, args=(i,)) for i in range(n_chats)]
t0 = time.monotonic(); [t.start() for t in ts]; [t.join() for t in ts]
done = sorted(d for s, d in res.values() if s == 'ok')
print(f'chats: {len(done)}/{n_chats} OK, total {time.monotonic()-t0:.1f}s')
print('completion times (s):', [round(d, 1) for d in done])
print('refused/errors:', [s for s, _ in res.values() if s != 'ok'])
