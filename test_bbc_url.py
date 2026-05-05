import requests
from bs4 import BeautifulSoup
import os
import sys

# allow importing app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.ml import model_manager
    from app.database import init_db
    init_db()

    url = 'https://www.bbc.com/news/articles/czx95rl4ek5o'
    response = requests.get(url, headers={'User-Agent': 'FactFinderBot/1.0'}, timeout=15)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1')
    paragraphs = [p.get_text(' ', strip=True) for p in soup.select('article p')]
    if not paragraphs:
        paragraphs = [p.get_text(' ', strip=True) for p in soup.select('p')]
    text = ' '.join(paragraphs).strip()
    if title:
        text = f"{title.get_text(' ', strip=True)} {text}".strip()

    print('Extracted length:', len(text))
    print('Preview:', text[:200])

    model_manager.load_or_train()
    pred = model_manager.predict(text)
    print("Label:", pred.label, "Confidence:", pred.confidence)
    print("Details:", pred.details)
except Exception as e:
    import traceback
    traceback.print_exc()
