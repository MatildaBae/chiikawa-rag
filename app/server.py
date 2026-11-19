"""Web server: Gradio chat UI at /, JSON API at /ask, health check at /healthz."""
import os

import gradio as gr
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import ChiikawaRAG

rag = ChiikawaRAG()
api = FastAPI(title='Chiikawa-pedia RAG')


class AskRequest(BaseModel):
    question: str


@api.get('/health')
@api.get('/healthz')
def healthz():
    return {'status': 'ok', 'chunks': len(rag.chunks), 'llm': bool(rag.client)}


@api.post('/ask')
def ask(req: AskRequest):
    r = rag.answer(req.question)
    return {'question': req.question, 'answer': r['answer'],
            'sources': [{'n': n, 'title': h['title'], 'section': h['section'], 'url': h['url'],
                         'score': round(h['score'], 3)} for n, h in enumerate(r['hits'], 1)]}


# ── Chat UI (same behaviour as the notebook) ──────────────────────────────
def history_to_turns(history, max_turns=3):
    turns = []
    for item in history or []:
        if isinstance(item, dict):
            turns.append((item.get('role'), item.get('content')))
        else:
            turns += [('user', item[0]), ('assistant', item[1])]
    clean = []
    for role, content in turns:
        if isinstance(content, list):
            content = ' '.join(p.get('text', '') for p in content if isinstance(p, dict))
        if isinstance(content, str) and content.strip():
            clean.append((role, content.split('\n\n---\n**Sources**')[0].strip()))
    return clean[-2 * max_turns:]


def chat_fn(message, history):
    question = rag.condense_question(message, history_to_turns(history))
    r = rag.answer(question)
    note = f'*(searched for: {question})*\n\n' if question != message else ''
    if r['sources']:
        return f"{note}{r['answer']}\n\n---\n**Sources**\n{r['sources']}"
    return note + r['answer']


WELCOME = f"""
### 🐹 Welcome to Chiikawa-pedia! I'm your expert on all things *Chiikawa* (ちいかわ).

**My knowledge domain:** Nagano's manga and anime about tiny, cute creatures who weed, hunt monsters,
earn licenses and eat a lot of ramen. I answer from {len(rag.chunks)} chunks of the Chiikawa fan wiki and
Wikipedia, found with a FAISS vector search, and every answer lists its sources.
"""

EXAMPLES = [
    'Why does Hachiware live in a cave?',
    'What weapon does Usagi use?',
    'Who trains Hachiware to get stronger?',
    "What is Shisa's dream?",
    'What happened between Momonga and Dekatsuyo?',
]

demo = gr.ChatInterface(fn=chat_fn, title='Chiikawa-pedia 🐰 Your Chiikawa RAG expert',
                        description=WELCOME, examples=EXAMPLES, cache_examples=False)
app = gr.mount_gradio_app(api, demo, path='/')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '8080')))
