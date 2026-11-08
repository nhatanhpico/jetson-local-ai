#!/usr/bin/env python3
"""End-to-end browser test + screenshots (docs/images). Needs: pip install selenium; Firefox.

    python scripts/ui-test.py [firefox-binary]

Opens a fresh Firefox profile (your own browser session is not touched), uses the real UI
(chat, image upload, image studio, memory button) and saves a screenshot after each step.
"""
import re, sys, time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / 'docs/images'
import os
# Point this at a throwaway instance (see README) so the test chats do not land in your real memory.
URL = os.environ.get('LOCAL_AI_TEST_URL', 'http://127.0.0.1:7860')
SAMPLE = IMG / 'sample-sailboat.png'


def main():
    o = Options()
    o.binary_location = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / '.local/opt/firefox/firefox')
    d = webdriver.Firefox(options=o)
    d.set_window_size(1440, 1300)
    wait = lambda f, t=240: WebDriverWait(d, t).until(f)
    shot = lambda n: (time.sleep(1), d.save_full_page_screenshot(str(IMG / n)), print('saved', n))
    click_text = lambda txt, tag='button': next(e for e in d.find_elements(By.TAG_NAME, tag) if e.text.strip().startswith(txt)).click()
    bubbles = lambda: d.find_elements(By.CSS_SELECTOR, '#conversation .bot, #conversation [data-testid="bot"]')
    try:
        d.get(URL)
        wait(lambda d: d.find_elements(By.CSS_SELECTOR, 'textarea'))
        time.sleep(2)
        shot('01-home.png')

        # 1) plain chat
        box = d.find_elements(By.CSS_SELECTOR, '#composer textarea')[0]
        box.send_keys("Hi, I'm Linh, a robotics engineer in Da Nang and I love hiking. In two short sentences: why is running AI locally on a Jetson useful?")
        click_text('Send message')
        wait(lambda d: len(bubbles()) >= 1)
        t0 = time.time()
        wait(lambda d: d.find_elements(By.CSS_SELECTOR, '#composer textarea')[0].get_attribute('value') == '')
        time.sleep(1)
        # wait until streaming stops (bubble text stable for 3s)
        last, stable = '', 0
        while stable < 3 and time.time() - t0 < 180:
            cur = ' '.join(b.text for b in bubbles())
            stable = stable + 1 if cur == last and cur else 0
            last = cur
            time.sleep(1)
        print('chat answer:', last[:200])
        shot('02-chat.png')

        # 2) vision: upload an image and ask about it
        click_text('Image attachment', 'button')
        time.sleep(1)
        d.find_element(By.CSS_SELECTOR, '#chat-image input[type=file]').send_keys(str(SAMPLE))
        time.sleep(3)
        box = d.find_elements(By.CSS_SELECTOR, '#composer textarea')[0]
        box.send_keys('What is in the attached picture? Answer in one sentence.')
        click_text('Send message')
        time.sleep(2)
        wait(lambda d: d.find_elements(By.CSS_SELECTOR, '#composer textarea')[0].get_attribute('value') == '', 300)
        time.sleep(2)
        print('vision answer:', bubbles()[-1].text[:250])
        shot('03-vision.png')

        # 3) image studio
        next(e for e in d.find_elements(By.CSS_SELECTOR, 'button[role=tab]') if e.text.strip() == 'Image studio').click()
        time.sleep(1)
        d.find_element(By.CSS_SELECTOR, 'textarea[placeholder^="A quiet mountain"]').send_keys(
            'a cozy wooden cabin beside a mountain lake at sunrise, soft watercolor, warm light')
        click_text('Create image')
        t0 = time.time()
        wait(lambda d: d.find_elements(By.CSS_SELECTOR, '#generated-image img'), 300)
        print('image generated in about %.0fs (UI-observed)' % (time.time() - t0))
        shot('04-image-studio.png')

        # 4) long-term memory tab: wait for the background extraction, then show what was saved
        next(e for e in d.find_elements(By.CSS_SELECTOR, 'button[role=tab]') if e.text.strip() == 'Memory').click()
        deadline = time.time() + 240
        cells = []
        while time.time() < deadline:
            click_text('Refresh')
            time.sleep(4)
            m = re.match(r'(\d+) saved', d.find_element(By.CSS_SELECTOR, '#memory-status').text.strip())
            if m and int(m.group(1)) > 0:
                cells = [m.group(0)]
                break
        print('memory status:', d.find_element(By.CSS_SELECTOR, '#memory-status').text.strip()[:80])
        assert cells, 'no memory was saved'
        shot('08-memory-tab.png')

        # 5) memory button
        next(e for e in d.find_elements(By.CSS_SELECTOR, 'button[role=tab]') if e.text.strip() == 'Chat').click()
        time.sleep(1)
        click_text('Device & memory', 'button')
        time.sleep(1)
        click_text('Free chat memory')
        time.sleep(3)
        shot('05-memory.png')
        # 6) dark theme and a phone-sized window (what a laptop/phone on the LAN would see)
        d.get(URL + '/?__theme=dark')
        wait(lambda d: d.find_elements(By.CSS_SELECTOR, 'textarea'))
        time.sleep(3)
        shot('06-dark.png')
        d.get(URL)
        d.set_window_size(480, 1000)
        wait(lambda d: d.find_elements(By.CSS_SELECTOR, 'textarea'))
        time.sleep(3)
        shot('07-mobile.png')
        print('UI TEST PASSED')
    finally:
        try:
            d.quit()
        except Exception:
            pass


if __name__ == '__main__':
    main()
