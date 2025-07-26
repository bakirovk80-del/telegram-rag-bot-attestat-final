import openai, numpy as np, json
from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


with open('pravila_detailed_tagged_autofix.json', encoding='utf-8') as f:
    data = json.load(f)

texts = []
for d in data:
    t = d['text']
    if len(t) > 8000:
        print(f"Внимание: пункт {d.get('punkt_num')} слишком длинный: {len(t)} символов! Будет обрезан.")
        t = t[:8000]
    texts.append(t)

# Генерируем embeddings ПОРЦИЯМИ (например, по 100 за раз, чтобы не упереться в лимиты API)
embeddings = []
batch_size = 100
for i in range(0, len(texts), batch_size):
    response = openai.embeddings.create(input=texts[i:i+batch_size], model="text-embedding-ada-002")
    embeddings.extend([e.embedding for e in response.data])

embeddings = np.array(embeddings)
np.save('embeddings.npy', embeddings)
print('OK:', len(data), embeddings.shape)
