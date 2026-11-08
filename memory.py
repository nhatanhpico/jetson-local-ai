"""Long-term memory for the chat, built on Mem0 and kept entirely on this device.

Mem0 uses the local Ollama chat model to extract facts worth remembering from each exchange and a
local embedding model (nomic-embed-text) to store them in an on-disk Qdrant database
(config.MEMORY_DIR). Nothing is sent to the internet; Mem0's telemetry is switched off below.

Every public function degrades gracefully: if Mem0 or the embedding model is unavailable the chat
keeps working without memory.
"""
import os
os.environ.setdefault('MEM0_TELEMETRY', 'False')   # must be set before mem0 is imported
import threading
import config

_lock = threading.Lock()
_mem = None
_error = None
OLLAMA_URL = config.OLLAMA_API.rsplit('/api', 1)[0]
FILTER = {'user_id': config.MEMORY_USER}
INSTRUCTIONS = (
    'Remember only lasting facts about the user: name, location, job, skills, projects, preferences, habits, '
    'relationships, allergies and constraints, and things they ask you to remember. Do NOT store general '
    'knowledge, explanations, the assistant\'s answers, descriptions of images, small talk or one-off questions. '
    'If the message holds nothing worth keeping, store nothing.')


def _get():
    """Create the Mem0 client on first use (opening the database takes a moment)."""
    global _mem, _error
    if _mem is not None or _error is not None:
        return _mem
    with _lock:
        if _mem is None and _error is None:
            try:
                from mem0 import Memory
                config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
                _mem = Memory.from_config({
                    'custom_instructions': INSTRUCTIONS,
                    'llm': {'provider': 'ollama', 'config': {
                        'model': config.CHAT_MODEL, 'temperature': 0, 'max_tokens': 600,
                        'ollama_base_url': OLLAMA_URL}},
                    'embedder': {'provider': 'ollama', 'config': {
                        'model': config.EMBED_MODEL, 'embedding_dims': config.EMBED_DIMS,
                        'ollama_base_url': OLLAMA_URL}},
                    'vector_store': {'provider': 'qdrant', 'config': {
                        'collection_name': 'jetson_local_ai', 'path': str(config.MEMORY_DIR / 'qdrant'),
                        'embedding_model_dims': config.EMBED_DIMS, 'on_disk': True}},
                    'history_db_path': str(config.MEMORY_DIR / 'history.db'),
                })
            except Exception as exc:  # missing package, Ollama down, locked database ...
                _error = f'{type(exc).__name__}: {exc}'
    return _mem


def status():
    """Short human-readable state for the UI."""
    if not config.MEMORY_ENABLED:
        return 'Memory is turned off (LOCAL_AI_MEMORY=0).'
    if _error:
        return 'Memory unavailable: ' + _error[:200]
    return 'Memory is on. Stored only on this device in ' + config.MEMORY_DIR.name + '/'


def _text(item):
    return item.get('memory') or item.get('text') or ''


def relevant(query, limit=None):
    """Memories related to `query`, as plain strings (empty list on any problem)."""
    mem = _get() if config.MEMORY_ENABLED else None
    if mem is None or not query.strip():
        return []
    try:
        res = mem.search(query, top_k=limit or config.MEMORY_TOP_K, filters=FILTER)
        items = res.get('results', res) if isinstance(res, dict) else res
        return [t for t in (_text(i) for i in items) if t]
    except Exception:
        return []


def remember(user_text, assistant_text=None):
    """Let Mem0 extract and store lasting facts from what the USER said. Returns what it saved or changed.

    Only the user's message is sent: when the assistant's own answer was included, the 4B model stored
    its general-knowledge statements as if they were facts about the user.
    """
    mem = _get() if config.MEMORY_ENABLED else None
    if mem is None:
        return []
    try:
        res = mem.add([{'role': 'user', 'content': user_text}], user_id=config.MEMORY_USER)
        items = res.get('results', []) if isinstance(res, dict) else res
        saved = []
        for i in items:
            if i.get('event') == 'ADD' and _is_duplicate(mem, i):
                continue
            saved.append((i.get('event', ''), _text(i)))
        return saved
    except Exception:
        return []


DUPLICATE_SCORE = 0.95   # near-identical wording scored 0.999 in testing; partial overlaps about 0.81


def _is_duplicate(mem, new_item):
    """Delete the new memory again if an older one says (almost) exactly the same thing."""
    try:
        res = mem.search(_text(new_item), top_k=5, filters=FILTER)
        for other in (res.get('results', res) if isinstance(res, dict) else res):
            if other.get('id') != new_item.get('id') and other.get('score', 0) >= DUPLICATE_SCORE:
                mem.delete(new_item['id'])
                return True
    except Exception:
        pass
    return False


def all_memories():
    """Everything currently stored: list of dicts with id, memory, created_at, updated_at."""
    mem = _get() if config.MEMORY_ENABLED else None
    if mem is None:
        return []
    try:
        res = mem.get_all(filters=FILTER, top_k=500)
        items = res.get('results', res) if isinstance(res, dict) else res
        return sorted(items, key=lambda i: i.get('created_at') or '', reverse=True)
    except Exception:
        return []


def forget(memory_id):
    mem = _get()
    if mem is not None:
        mem.delete(memory_id)


def forget_all():
    mem = _get()
    if mem is not None:
        mem.delete_all(user_id=config.MEMORY_USER)
