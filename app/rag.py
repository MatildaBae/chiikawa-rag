"""Chiikawa-pedia RAG core: chunking, FAISS retrieval, grounded answers with gpt-4o-mini.

Ported from JiwonBae_BuildMyOwn_ChiikawaRAG.ipynb (Lab 2 extra credit).
Embeddings use fastembed (ONNX build of BAAI/bge-small-en-v1.5) so the image needs no PyTorch.
"""
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

# ── Knobs (same values as the notebook) ───────────────────────────────────
EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '
CHUNK_WORDS = 120
OVERLAP_SENTS = 1
TOP_K = 5
CANDIDATES = 30
NAME_BOOST = 0.10
MIN_SCORE = float(os.environ.get('MIN_SCORE', '0.40'))
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
MAX_TOKENS = 300

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = Path(os.environ.get('INDEX_DIR', ROOT / 'chiikawa_index'))
MODEL_CACHE = os.environ.get('MODEL_CACHE', str(ROOT / 'models'))


# ── Text helpers ──────────────────────────────────────────────────────────
def norm(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()


def clean_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


# ── Chunking ──────────────────────────────────────────────────────────────
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(\[])')


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def window_chunks(sentences: list[str], max_words: int, overlap: int) -> list[str]:
    chunks, cur = [], []
    for sent in sentences:
        if cur and sum(len(s.split()) for s in cur) + len(sent.split()) > max_words:
            chunks.append(' '.join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.append(sent)
    if cur:
        chunks.append(' '.join(cur))
    return chunks


def parse_markdown(md: str):
    title, url, heads, buf = '', '', {}, []

    def emit():
        if buf:
            path = ' > '.join(heads[lvl] for lvl in sorted(heads))
            yield title, url, path or 'Overview', ' '.join(buf)
        buf.clear()

    for line in md.splitlines():
        line = line.strip()
        if line.startswith('# '):
            title = line[2:]
        elif line.startswith('Source: '):
            url = line[8:]
        elif m := re.match(r'^(#{2,4})\s+(.*)', line):
            yield from emit()
            level = len(m.group(1))
            heads = {lvl: h for lvl, h in heads.items() if lvl < level}
            heads[level] = m.group(2)
        elif line:
            buf.append(line if line[-1] in '.!?"”)' else line + '.')
    yield from emit()


def page_key(title: str) -> str:
    return norm(re.sub(r'\(.*?\)', '', title.split('/')[0]))


def is_junk(text: str) -> bool:
    return text.count('Retrieved') >= 2 or text.count('http') >= 2 or len(text.split()) < 6


def make_chunk(title, url, section, body, source_file):
    return {
        'title': title, 'section': section, 'url': url, 'source_file': source_file,
        'page_key': page_key(title), 'text': body,
        'embed_text': f'{title} — {section}: {body}',
        'n_words': len(body.split()),
    }


def chunk_markdown_file(path: Path) -> list[dict]:
    out = []
    for title, url, section, text in parse_markdown(path.read_text(encoding='utf-8')):
        for body in window_chunks(split_sentences(text), CHUNK_WORDS, OVERLAP_SENTS):
            if not is_junk(body):
                out.append(make_chunk(title, url, section, body, path.name))
    return out


def chunk_pdf_file(path: Path) -> list[dict]:
    from pypdf import PdfReader
    pages = [(i + 1, page.extract_text() or '') for i, page in enumerate(PdfReader(path).pages)]
    end_of_article = re.compile(r'^\s*(References|Notes|External links)\b', flags=re.M)
    out = []
    for page_no, raw in pages:
        raw = '\n'.join(l for l in raw.splitlines() if 'wikipedia.org' not in l)
        pages_done = bool(end_of_article.search(raw))
        if pages_done:
            raw = end_of_article.split(raw)[0]
        text = clean_ws(raw.replace('-\n', ''))
        for body in window_chunks(split_sentences(text), CHUNK_WORDS, OVERLAP_SENTS):
            if not is_junk(body):
                out.append(make_chunk('Chiikawa (Wikipedia PDF)', 'https://en.wikipedia.org/wiki/Chiikawa',
                                      f'page {page_no}', body, path.name))
        if pages_done:
            break
    return out


def build_chunks(data_dir: Path) -> list[dict]:
    chunks = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix == '.md':
            chunks += chunk_markdown_file(path)
        elif path.suffix == '.pdf':
            chunks += chunk_pdf_file(path)
    seen, unique = set(), []
    for c in chunks:
        if c['text'] not in seen:
            seen.add(c['text'])
            unique.append(c)
    for i, c in enumerate(unique):
        c['id'] = i
    return unique


# ── Embeddings ────────────────────────────────────────────────────────────
class Embedder:
    """fastembed wrapper returning L2-normalised float32 vectors."""

    def __init__(self, model_name: str = EMBED_MODEL, cache_dir: str = MODEL_CACHE):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(list(self.model.embed(texts)), dtype='float32')
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs


# ── Retrieval + answering ─────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Chiikawa-pedia, a friendly expert on Nagano's manga and anime series Chiikawa. "
    "Answer ONLY with facts from the numbered context passages. Cite passages inline like [1] or [2]. "
    "If the passages do not contain the answer, say you couldn't find it in your Chiikawa notes "
    "instead of guessing. Keep answers under 120 words."
)

REFUSAL = ("I couldn't find anything about that in my Chiikawa notes. "
           "Try asking about a character, place, or event from the series!")


class ChiikawaRAG:
    def __init__(self, index_dir: Path = INDEX_DIR, embedder=None, client=None):
        import faiss
        self.chunks = [json.loads(l) for l in (index_dir / 'chunks.jsonl').read_text(encoding='utf-8').splitlines() if l]
        self.index = faiss.read_index(str(index_dir / 'chiikawa.faiss'))
        self.embedder = embedder or Embedder()
        self.client = client
        if self.client is None and os.environ.get('OPENAI_API_KEY'):
            from openai import OpenAI
            self.client = OpenAI()   # reads OPENAI_API_KEY from the environment (Secret Manager on Cloud Run)
        self.name_keys = {c['page_key'] for c in self.chunks} - {'chiikawa', 'chiikawa wikipedia pdf', ''}

    def mentioned_pages(self, question: str) -> set[str]:
        q = f' {norm(question)} '
        return {k for k in self.name_keys if f' {k} ' in q or f' {k.replace(" ", "")} ' in q}

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        q = self.embedder.encode([QUERY_PREFIX + question])
        scores, idxs = self.index.search(q, min(CANDIDATES, self.index.ntotal))
        named = self.mentioned_pages(question)
        if len(named) > 2:
            named = set()
        hits = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            c = self.chunks[i]
            boost = NAME_BOOST if c['page_key'] in named else 0.0
            hits.append({**c, 'score': float(score), 'final': float(score) + boost})
        hits.sort(key=lambda h: h['final'], reverse=True)
        return hits[:k]

    @staticmethod
    def format_context(hits: list[dict]) -> str:
        return '\n\n'.join(f"[{n}] ({h['title']} > {h['section']}) {h['text']}" for n, h in enumerate(hits, 1))

    @staticmethod
    def format_sources(hits: list[dict]) -> str:
        return '\n'.join(f"- [{n}] [{h['title']} › {h['section']}]({h['url']}) · score {h['score']:.2f}"
                         for n, h in enumerate(hits, 1))

    def answer(self, question: str) -> dict:
        hits = self.retrieve(question)
        if not hits or hits[0]['final'] < MIN_SCORE:
            return {'answer': REFUSAL, 'hits': [], 'sources': ''}
        hits = [h for h in hits if h['final'] >= MIN_SCORE]
        if self.client:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL, temperature=0.2, max_tokens=MAX_TOKENS,
                messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                          {'role': 'user', 'content': f'Context:\n{self.format_context(hits)}\n\nQuestion: {question}'}])
            text = resp.choices[0].message.content.strip()
        else:
            text = 'Here is what my Chiikawa notes say:\n\n' + '\n\n'.join(
                f"• {h['text']} [{n}]" for n, h in enumerate(hits[:2], 1))
        return {'answer': text, 'hits': hits, 'sources': self.format_sources(hits)}

    # Follow-up rewriting for the chat UI (same prompt as the notebook)
    REWRITE_PROMPT = (
        "Rewrite the user's last message as a standalone question about the Chiikawa series.\n"
        "- Replace pronouns ('they', 'he', 'that') with the names they refer to.\n"
        "- If the message is a short follow-up like 'how about X?', 'what about X?' or 'and X?', "
        "apply the previous question to X.\n"
        "- If it is already standalone, return it unchanged.\n"
        "Output only the question."
    )

    def condense_question(self, message: str, turns: list[tuple[str, str]]) -> str:
        if not turns:
            return message
        if self.client:
            convo = '\n'.join(f'{role}: {text[:500]}' for role, text in turns)
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL, temperature=0, max_tokens=60,
                messages=[{'role': 'system', 'content': self.REWRITE_PROMPT},
                          {'role': 'user', 'content': f'Conversation:\n{convo}\n\nLast message: {message}'}])
            return resp.choices[0].message.content.strip()
        last_user = next((t for r, t in reversed(turns) if r == 'user'), '')
        return f'{last_user} {message}'
