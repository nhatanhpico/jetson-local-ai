"""Optional functional test (memory is switched off so nothing is written to your real memory). Runs one image generation; takes about a minute."""
from pathlib import Path
import time
from PIL import Image
from gradio_client import Client, handle_file
root = Path(__file__).resolve().parent.parent
image = root / 'outputs/vision-test.png'
Image.new('RGB', (128, 128), (255, 0, 0)).save(image)
c = Client('http://127.0.0.1:7860', verbose=False)
print(c.predict('What color is this image?', handle_file(str(image)), False, api_name='/chat'))
print(c.predict('Remember the word apple. Say OK.', None, False, api_name='/chat'))
print(c.predict('What word did I ask you to remember?', None, False, api_name='/chat'))
started = time.monotonic()
print(c.predict('a watercolor painting of a small red sailboat on a calm blue lake, mountains, sunny day', 'blurry, low quality', 20, 42, api_name='/generate'))
print('Image generation seconds:', round(time.monotonic() - started, 1))
print(c.predict(api_name='/unload'))
