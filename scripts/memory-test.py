#!/usr/bin/env python3
"""Tests long-term memory (Mem0) and conversation compaction against a THROWAWAY instance.

Start one first, with a tiny context window so compaction triggers quickly and a scratch memory folder:

    LOCAL_AI_PORT=7863 LOCAL_AI_NUM_CTX=1024 LOCAL_AI_NUM_PREDICT=220 \
    LOCAL_AI_MEMORY_DIR=/tmp/ai-test-mem venv/bin/python app.py &
    venv/bin/python scripts/memory-test.py http://127.0.0.1:7863

Never point it at your real instance: it deletes all memories at the end.
"""
import re, sys, time
from gradio_client import Client

url = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:7863'
c = Client(url, verbose=False)
text = lambda m: m['content'][0]['text'] if isinstance(m['content'], list) else m['content']

print('1) compaction (memory off)')
notes = 0
for t in ['My name is Linh and I am building an underwater robot called Manta. Write a detailed 120-word paragraph about buoyancy control for it.',
          'Now a detailed 120-word paragraph about thruster placement for Manta.',
          'Now a detailed 120-word paragraph about battery choices for Manta.',
          'Now a detailed 120-word paragraph about waterproofing for Manta.',
          'Without me repeating it: what is the robot called and who am I?']:
    chat, meter = c.predict(t, None, False, api_name='/chat')
    notes = sum('compacted' in text(m) or 'trimmed' in text(m) for m in chat)
    print('   ', re.sub(r'\*', '', meter)[:70], '| notices:', notes)
assert notes >= 1, 'compaction never ran (is the instance started with LOCAL_AI_NUM_CTX=1024?)'
assert 'manta' in text(chat[-1]).lower(), 'the summary lost the robot name'
c.predict(api_name='/new_conversation')

print('2) memory saved, listed, recalled in a NEW conversation')
c.predict("I'm Linh, a robotics engineer in Da Nang. I love hiking on weekends and I'm allergic to peanuts.", None, True, api_name='/chat')
rows = []
for _ in range(24):
    time.sleep(5)
    rows = c.predict(api_name='/memories')[0]['data']
    if rows:
        break
print('   saved:', [r[1] for r in rows])
assert rows, 'nothing was saved within 2 minutes'
c.predict(api_name='/new_conversation')
chat, _ = c.predict('What do you know about me?', None, True, api_name='/chat')
answer = text(chat[-1]).lower()
print('   recall:', answer[:160])
assert 'linh' in answer or 'robot' in answer or 'peanut' in answer, 'memory was not recalled'

print('3) delete')
try:
    c.predict(False, api_name='/forget_everything')
    raise SystemExit('forget-all should refuse without confirmation')
except Exception as exc:
    assert 'Tick the box' in str(exc), exc
res = c.predict(True, api_name='/forget_everything')
assert not res[0]['data'], 'memories remain after forget-all'
print('MEMORY TEST PASSED')
