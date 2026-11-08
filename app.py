import base64, io, json, subprocess, uuid, threading
from datetime import datetime
from PIL import Image, ImageOps
import gradio as gr
import requests
import config
import memory
ROOT=config.ROOT
MODEL=config.CHAT_MODEL
API=config.OLLAMA_API
lock=threading.Lock()
def unload():
    try:
        r=requests.post(API+'/generate',json={'model':MODEL,'keep_alive':0},timeout=120)
        r.raise_for_status()
    except requests.RequestException:
        return 'Could not reach the chat service (Ollama). Nothing to unload, or it is not running.'
    return 'Chat model unloaded. Memory is available.'
def encode_image(picture):
    # Normalize phone photos (including MPO), orientation and color mode.
    with Image.open(picture) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
    return base64.b64encode(buffer.getvalue()).decode()

IMAGE_TOKENS = 1100          # a 768 px image cost 1,055 prompt tokens in testing
COMPACT_AT = int(config.NUM_CTX * config.COMPACT_RATIO)


def guess_tokens(text):
    """Cheap, deliberately pessimistic estimate (about 3 characters per token)."""
    return len(text) // 3 + 1


def context_label(used):
    pct = min(100, round(100 * used / config.NUM_CTX))
    return f'Context window: **{used:,}** / {config.NUM_CTX:,} tokens ({pct}%). Older turns are summarized at {COMPACT_AT:,}.'


def as_text(message):
    text = message['content']
    return text + ' [an image was attached]' if message.get('images') else text


def compact(history):
    """Replace older turns with a short summary so the conversation fits the context window again.

    Keeps the last config.COMPACT_KEEP messages word for word. Returns (new_history, ok).
    Falls back to simply dropping the oldest turn if the summary request fails.
    """
    keep = config.COMPACT_KEEP
    old, recent = list(history[:-keep]), list(history[-keep:])
    while recent and recent[0]['role'] != 'user' and old:      # never start the kept part mid-turn
        old.append(recent.pop(0))
    if not old:
        return history, False
    transcript = '\n'.join(f"{m['role'].upper()}: {as_text(m)}" for m in old)
    ask = ('Summarize this conversation in under 200 words. Keep names, facts, decisions, numbers and '
           'unresolved questions. Write only the summary.\n\n' + transcript)
    try:
        r = requests.post(API + '/chat', json={'model': MODEL, 'stream': False, 'keep_alive': config.KEEP_ALIVE,
                                               'messages': [{'role': 'user', 'content': ask}],
                                               'options': {'num_ctx': config.NUM_CTX, 'num_predict': 400, 'temperature': 0.2}},
                          timeout=300)
        r.raise_for_status()
        summary = r.json()['message']['content'].strip()
        if not summary:
            raise ValueError('empty summary')
    except (requests.RequestException, ValueError, KeyError):
        return recent or history[-2:], False
    return [{'role': 'system', 'content': 'Summary of the earlier conversation:\n' + summary}] + recent, True


def remember_async(user_text, assistant_text):
    """Store lasting facts in the background. Holds the GPU lock so it never overlaps image generation."""
    def work():
        with lock:
            memory.remember(user_text, assistant_text)
    threading.Thread(target=work, daemon=True).start()


def chat(prompt, picture, history, shown, used, use_memory):
    if not prompt.strip():
        raise gr.Error('Enter a message first.')
    history = list(history or [])
    shown = list(shown or [])
    used = used or 0
    msg = {'role': 'user', 'content': prompt}
    if picture:
        try:
            encoded = encode_image(picture)
        except (OSError, ValueError) as exc:
            raise gr.Error('This image could not be read. Try a JPEG or PNG file.') from exc
        # The upload stays selected between messages; do not duplicate its tokens.
        previous = next((m['images'][0] for m in reversed(history) if m.get('images')), None)
        if encoded != previous:
            msg['images'] = [encoded]
    with lock:
        # 1) Long-term memory: facts remembered from earlier conversations.
        recalled = memory.relevant(prompt) if use_memory else []
        # 2) Compaction: summarize older turns when the conversation reaches the context limit.
        projected = used + guess_tokens(prompt) + (IMAGE_TOKENS if msg.get('images') else 0)
        if projected >= COMPACT_AT and len(history) > 2:
            history, ok = compact(history)
            used = sum(guess_tokens(as_text(m)) for m in history)
            shown.append({'role': 'assistant', 'content': '*Conversation compacted: older messages were summarized to stay within the context window.*'
                          if ok else '*Conversation trimmed: the oldest messages were dropped to stay within the context window.*'})
        messages = history + [msg]
        outgoing = messages
        if recalled:
            note = ('Long-term memory about the user (from earlier chats). Use it only when relevant and do not '
                    'mention this list unless asked:\n' + '\n'.join('- ' + r for r in recalled))
            outgoing = [{'role': 'system', 'content': note}] + messages
        display = shown + [{'role': 'user', 'content': prompt}]
        try:
            while True:
                response = requests.post(API + '/chat', json={'model': MODEL, 'messages': outgoing, 'stream': True, 'keep_alive': config.KEEP_ALIVE, 'options': {'num_ctx': config.NUM_CTX, 'num_predict': config.NUM_PREDICT}}, stream=True, timeout=(10, 600))
                if response.ok:
                    break
                try:
                    detail = response.json().get('error', response.text)
                except ValueError:
                    detail = response.text
                response.close()
                if response.status_code == 400 and 'context' in detail.lower() and len(messages) > 2:
                    # Last resort if the estimate was too low: drop the oldest turn and retry.
                    messages = messages[2:]
                    outgoing = messages if not recalled else [outgoing[0]] + messages
                    continue
                raise gr.Error('The model could not process this request: ' + detail[:400])
            answer = ''
            with response:
                for line in response.iter_lines():
                    if line:
                        item = json.loads(line)
                        if 'error' in item:
                            raise gr.Error(item['error'])
                        answer += item.get('message', {}).get('content', '')
                        if item.get('done'):
                            used = (item.get('prompt_eval_count') or 0) + (item.get('eval_count') or 0) or used
                        yield display + [{'role': 'assistant', 'content': answer}], history, shown, used, context_label(used)
        except requests.RequestException as exc:
            raise gr.Error('Cannot reach the local model. Please retry; if it persists, check the Ollama service.') from exc
    messages.append({'role': 'assistant', 'content': answer})
    shown = display + [{'role': 'assistant', 'content': answer}]
    if use_memory and len(prompt.strip()) >= 12:
        remember_async(prompt, answer)
    yield shown, messages, shown, used, context_label(used)


def generate(prompt, negative, steps, seed):
    if not prompt or not prompt.strip(): raise gr.Error('Describe the image first.')
    if not config.SD_BIN.exists() or not config.SD_MODEL.exists():
        raise gr.Error('Image engine or model is missing. Run scripts/install.sh (see README).')
    # A cleared number field arrives as None; fall back to defaults instead of crashing.
    steps=min(40,max(10,int(steps if steps is not None else 20)))
    seed=int(seed if seed is not None else 42)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    output=config.OUTPUT_DIR/f'{uuid.uuid4().hex}.png'
    args=config.sd_command(prompt,negative or '',steps,seed,output)
    with lock:
        unload()  # free GPU memory for the image engine; harmless if chat is not loaded
        with config.GEN_LOG.open('w') as log:
            try:
                result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=config.SD_TIMEOUT)
            except subprocess.TimeoutExpired:
                raise gr.Error('Image generation timed out and was stopped. Try fewer steps.')
        if result.returncode: raise gr.Error('Generation failed. See generation.log in the project folder.')
    if not output.exists():
        matches=list(output.parent.glob(output.stem+'*.png'))
        if not matches: raise gr.Error('No output image was produced.')
        output=matches[0]
    return str(output)

def new_conversation():
    return [], [], [], 0, context_label(0), '', None


def memory_rows():
    """Current long-term memories as (table rows, dropdown choices, status text)."""
    items = memory.all_memories()
    rows, choices = [], []
    for i in items:
        try:
            when = datetime.fromisoformat(i['created_at']).astimezone().strftime('%Y-%m-%d %H:%M')
        except (KeyError, TypeError, ValueError):
            when = ''
        rows.append([when, i['memory']])
        choices.append((i['memory'][:90], i['id']))
    label = f'{len(items)} saved memor{"y" if len(items) == 1 else "ies"}. ' + memory.status()
    return rows, gr.update(choices=choices, value=None), label


def forget_one(memory_id):
    if memory_id:
        try:
            memory.forget(memory_id)
        except Exception as exc:
            raise gr.Error('Could not delete that memory: ' + str(exc)[:200]) from exc
    return memory_rows()


def forget_everything(sure):
    if not sure:
        raise gr.Error('Tick the box first to confirm you want to delete every memory.')
    memory.forget_all()
    return memory_rows() + (False,)


# Material Design 3 look: tonal teal, large rounded shapes, no heavy borders. Details live in ui.css.
THEME = gr.themes.Soft(
    primary_hue='teal', secondary_hue='teal', neutral_hue='slate',
    radius_size=gr.themes.sizes.radius_xxl,
    font=['Roboto', 'system-ui', 'sans-serif'], font_mono=['ui-monospace', 'monospace'],
).set(block_shadow='none', button_secondary_shadow='none')

with gr.Blocks(title='Jetson Local AI') as app:
    gr.HTML('''<header class="app-header">
        <div class="brand"><span class="brand-mark" aria-hidden="true">J</span>
        <div><h1>Jetson <span>Local AI</span></h1><p>Your ideas. Your device.</p></div></div>
        <span class="local-badge">On-device inference</span>
    </header>''')
    with gr.Row(elem_id='workspace'):
        with gr.Column(scale=1, min_width=220, elem_id='sidebar'):
            gr.HTML('''<div class="sidebar-intro"><span class="eyebrow">YOUR WORKSPACE</span>
                <h2>A little help.<br>Endless possibilities.</h2>
                <p>Think out loud, explore an image, or create something new.</p></div>''')
            new_chat = gr.Button('+  New conversation', elem_id='new-chat')
            with gr.Accordion('Image attachment', open=False):
                picture = gr.Image(type='filepath', label='Image to discuss', height=190,
                                   sources=['upload'], elem_id='chat-image')
                gr.Markdown('Upload a photo, then ask about it in chat. It stays attached until you remove it.')
            with gr.Accordion('Device & memory', open=False):
                gr.Markdown('**Runs locally on your Jetson**\n\nChat releases memory after 3 minutes idle. Image creation manages memory automatically.')
                release = gr.Button('Free chat memory', size='sm')
                status = gr.Textbox(label='Memory status', interactive=False, lines=2,
                                    placeholder='Memory updates appear here.')
            gr.HTML('<div class="sidebar-note">Made for focused work.<br>Conversations last for this session.</div>')
        with gr.Column(scale=4, min_width=320, elem_id='main-panel'):
            with gr.Tabs():
                use_memory = gr.Checkbox(value=config.MEMORY_ENABLED, label='Use and update memory in chat',
                                         info='When off, nothing is recalled and nothing new is saved.', render=False)
                with gr.Tab('Chat', id='chat'):
                    history = gr.State([])      # what the model sees (may contain a summary)
                    shown = gr.State([])        # what the user sees (full conversation)
                    used = gr.State(0)          # tokens used in the context window so far
                    answer = gr.Chatbot(
                        label='Conversation', show_label=False, height=460,
                        buttons=['copy'], layout='bubble', elem_id='conversation',
                        placeholder='''<div class="welcome"><span class="welcome-mark" aria-hidden="true">✦</span>
                            <h2>What’s on your mind?</h2><p>Ask a question, work through an idea,<br>or attach an image to explore together.</p></div>''',
                    )
                    with gr.Group(elem_id='composer'):
                        prompt = gr.Textbox(label='Your message', show_label=False, lines=2,
                                            max_lines=8, placeholder='Ask anything, or describe what you’re working on…')
                        with gr.Row(elem_id='composer-actions'):
                            gr.Markdown('Shift + Enter for a new line', elem_id='composer-hint')
                            send = gr.Button('Send message  ↑', variant='primary', scale=0, min_width=150)
                    gr.Examples(
                        examples=['Help me turn a rough idea into a plan.',
                                  'Explain a complex topic in simple terms.',
                                  'Give me three creative writing prompts.'],
                        inputs=prompt, label='Try a starting point',
                    )
                    ctx = gr.Markdown(context_label(0), elem_id='ctx-meter')
                    gr.Markdown('AI can make mistakes. Double-check important details.', elem_id='chat-footnote')
                    gr.on(triggers=[send.click, prompt.submit], fn=chat,
                          inputs=[prompt, picture, history, shown, used, use_memory],
                          outputs=[answer, history, shown, used, ctx],
                          concurrency_id='gpu', concurrency_limit=1, api_name='chat',
                          trigger_mode='once').success(lambda: '', outputs=prompt, api_name=False)
                    new_chat.click(new_conversation, outputs=[answer, history, shown, used, ctx, prompt, picture], api_name='new_conversation')
                    release.click(unload, outputs=status, concurrency_id='gpu', concurrency_limit=1)
                with gr.Tab('Image studio', id='images'):
                    gr.HTML('''<div class="studio-heading"><span class="eyebrow">FROM IDEA TO IMAGE</span>
                        <h2>Imagine it. Create it.</h2><p>Describe a scene and make it your own.</p></div>''')
                    with gr.Row():
                        with gr.Column(scale=1, min_width=260):
                            desc = gr.Textbox(label='Your vision', lines=5,
                                              placeholder='A quiet mountain cabin beside a lake at sunrise, soft watercolor, warm light…')
                            with gr.Accordion('Fine-tune your image', open=False):
                                neg = gr.Textbox(label='What to avoid', value='blurry, low quality, distorted', lines=2)
                                steps = gr.Slider(10, 40, value=20, step=1, label='Detail steps',
                                                  info='More steps take longer.')
                                seed = gr.Number(value=42, precision=0, label='Seed',
                                                 info='Reuse a seed and settings for similar results.')
                            go = gr.Button('Create image  ✦', variant='primary')
                            gr.Markdown('**512 × 512** · Stable Diffusion 1.5\n\nUsually about a minute. Images are saved on your device.', elem_id='generation-note')
                        with gr.Column(scale=1, min_width=260):
                            result = gr.Image(label='Your creation', height=420, interactive=False,
                                              elem_id='generated-image')
                    go.click(generate, [desc, neg, steps, seed], result,
                             concurrency_id='gpu', concurrency_limit=1)
                with gr.Tab('Memory', id='memory') as memory_tab:
                    gr.HTML('''<div class="studio-heading"><span class="eyebrow">LONG-TERM MEMORY</span>
                        <h2>What I remember about you.</h2><p>Lasting facts from your chats, stored only on this device. Delete anything you do not want kept.</p></div>''')
                    use_memory.render()
                    mem_status = gr.Markdown('', elem_id='memory-status')
                    mem_table = gr.Dataframe(headers=['Saved', 'Memory'], datatype=['str', 'str'], value=[],
                                             interactive=False, wrap=True, show_label=False, elem_id='memory-table')
                    with gr.Row():
                        refresh = gr.Button('Refresh', size='sm', scale=0, min_width=120)
                    with gr.Accordion('Forget something', open=False):
                        pick = gr.Dropdown(choices=[], label='Choose a memory to delete')
                        forget_btn = gr.Button('Forget this memory', size='sm')
                        gr.Markdown('---')
                        sure = gr.Checkbox(label='Yes, delete ALL memories')
                        forget_all_btn = gr.Button('Forget everything', size='sm')
                    memory_tab.select(memory_rows, outputs=[mem_table, pick, mem_status], concurrency_limit=None)
                    refresh.click(memory_rows, outputs=[mem_table, pick, mem_status], api_name='memories', concurrency_limit=None)
                    forget_btn.click(forget_one, inputs=pick, outputs=[mem_table, pick, mem_status], concurrency_limit=None)
                    forget_all_btn.click(forget_everything, inputs=sure, outputs=[mem_table, pick, mem_status, sure], concurrency_limit=None)
    gr.HTML('<footer class="app-footer">JETSON LOCAL AI <span>Chat · Understand · Create</span></footer>')

if __name__ == '__main__':
    app.queue(max_size=config.MAX_QUEUE).launch(
        theme=THEME, css=(ROOT / 'ui.css').read_text(),
        server_name=config.HOST, server_port=config.PORT, inbrowser=False,
        auth=config.AUTH,
    )
