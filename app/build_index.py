"""Build the Chiikawa knowledge base + FAISS index. Runs once during `docker build`.

1. Fetch fan-wiki pages (MediaWiki API) and the Wikipedia article -> cleaned .md in data/
   (any .md / .pdf you put in data/ yourself, e.g. your Wikipedia PDF, is used too)
2. Chunk -> embed with bge-small-en-v1.5 -> save chiikawa_index/{chunks.jsonl, chiikawa.faiss}
"""
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Comment, NavigableString

from app.rag import INDEX_DIR, ROOT, Embedder, build_chunks, clean_ws, norm

DATA_DIR = ROOT / 'data'
WIKI = 'https://chiikawa.fandom.com'
HEADERS = {'User-Agent': 'ENGI4502-Assignment3-RAG-student-project/1.0 (educational use)'}

WIKI_PAGES = [
    'Chiikawa (series)', 'The Chiikawa World', 'Chiikawa World', 'Chiikawas (species)',
    'Chiikawa', 'Chiikawa/Relationships', 'Hachiware', 'Hachiware/Relationships',
    'Usagi', 'Usagi/Relationships', 'Momonga', 'Momonga/Relationships',
    'Kuri-Manjū', 'Kuri-Manjū/Relationships', 'Rakko', 'Rakko/Relationships',
    'Shisa', 'Shisa/Relationships', 'Furuhonya', 'Chiikabu', 'Dekatsuyo', 'Ode',
    'Anoko', 'Majo', 'Labor Yoroi-san', 'Pochette Yoroi-san', 'Ramen Yoroi-san',
    'Yatai no Yoroi-san', 'Yoroi (Humans)', 'Secondary Characters', 'Mob Characters',
    'Goblins', 'Chimera (species)', 'Chimera Hachiware', 'Mimic (species)', 'Muchauman',
    'Siren/Island Arc', 'Magical Girl Chiikawa', 'Marketplace', 'Fish Sausage Grove',
    'Mt. Kiri', 'Pajama Parties', 'Chiikawa Ramen', 'Shooting Star', 'Black Star',
    'Sunfish', 'Chiikawa Pocket', 'Chiikawa Collaborations', 'Sanrio x Chiikawa',
    'My Hero Academia x Chiikawa', 'Uniqlo x Chiikawa x Joke Bear and Friends',
]

JUNK = ['table.article-table', 'figure', '.gallery', 'gallery', '.wikia-gallery',
        'sup', '.reference', 'style', 'script', '.mw-editsection', '.navbox',
        '.toc', 'noscript', 'img', '.mw-empty-elt']


def html_to_markdown(title: str, url: str, html: str) -> str:
    root = BeautifulSoup(html, 'html.parser')
    root = root.select_one('.mw-parser-output') or root
    for c in root.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    lines = [f'# {title}', f'Source: {url}']
    box = root.select_one('aside.portable-infobox')
    if box:
        facts = []
        for item in box.select('.pi-data'):
            label, value = item.select_one('.pi-data-label'), item.select_one('.pi-data-value')
            if label and value:
                facts.append(f'{clean_ws(label.get_text())}: {clean_ws(value.get_text(" "))}')
        if facts:
            lines += ['', '## Profile', '. '.join(facts) + '.']
        box.decompose()
    for sel in JUNK:
        for el in root.select(sel):
            el.decompose()
    lines += ['', '## Overview']
    buf = []

    def flush():
        text = clean_ws(''.join(buf))
        buf.clear()
        if text:
            lines.append(text)

    for el in list(root.children):
        if isinstance(el, NavigableString):
            buf.append(str(el))
            continue
        name = el.name
        if name in ('h2', 'h3', 'h4'):
            flush()
            lines += ['', '#' * int(name[1]) + ' ' + clean_ws(el.get_text())]
        elif name in ('p', 'blockquote'):
            flush(); buf.append(el.get_text()); flush()
        elif name in ('ul', 'ol'):
            flush()
            for li in el.find_all('li', recursive=False):
                text = clean_ws(li.get_text())
                if text:
                    lines.append(text)
        elif name == 'table':
            flush()
            for tr in el.find_all('tr'):
                cells = [clean_ws(td.get_text(' ')) for td in tr.find_all(['td', 'th'])]
                cells = [c for c in cells if c]
                if cells:
                    lines.append(' | '.join(cells))
        elif name in ('div', 'section', 'center', 'dl'):
            flush()
            text = clean_ws(el.get_text(' '))
            if text:
                lines.append(text)
        else:
            buf.append(el.get_text())
    flush()
    return '\n'.join(lines) + '\n'


def slug(title: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', norm(title)).strip('_')


def fetch_wiki_page(title: str) -> tuple[str, str]:
    r = requests.get(f'{WIKI}/api.php', headers=HEADERS, timeout=20, params={
        'action': 'parse', 'page': title, 'prop': 'text', 'format': 'json',
        'formatversion': 2, 'redirects': 1, 'disabletoc': 1, 'disableeditsection': 1})
    r.raise_for_status()
    data = r.json()
    if 'error' in data:
        raise ValueError(data['error'].get('info'))
    return data['parse']['title'], data['parse']['text']


def fetch_wikipedia_markdown() -> str:
    r = requests.get('https://en.wikipedia.org/w/api.php', headers=HEADERS, timeout=20, params={
        'action': 'query', 'prop': 'extracts', 'explaintext': 1, 'titles': 'Chiikawa',
        'format': 'json', 'formatversion': 2})
    r.raise_for_status()
    text = r.json()['query']['pages'][0]['extract']
    text = re.split(r'\n==\s*(References|Notes|External links|See also)\s*==', text)[0]
    text = re.sub(r'^===\s*(.*?)\s*===\s*$', r'### \1', text, flags=re.M)
    text = re.sub(r'^==\s*(.*?)\s*==\s*$', r'## \1', text, flags=re.M)
    return '# Chiikawa (Wikipedia)\nSource: https://en.wikipedia.org/wiki/Chiikawa\n\n## Overview\n' + text


def collect_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    saved, failed = 0, []
    for title in WIKI_PAGES:
        out = DATA_DIR / f'fandom__{slug(title)}.md'
        if out.exists():
            saved += 1
            continue
        try:
            real_title, html = fetch_wiki_page(title)
            url = f"{WIKI}/wiki/{real_title.replace(' ', '_')}"
            out.write_text(html_to_markdown(real_title, url, html), encoding='utf-8')
            saved += 1
            time.sleep(0.3)
        except Exception as e:
            failed.append((title, str(e)[:80]))
    print(f'Fan-wiki pages: {saved}/{len(WIKI_PAGES)}')
    for title, err in failed:
        print(f'  skipped {title!r}: {err}')
    if not list(DATA_DIR.glob('*.pdf')):
        wp = DATA_DIR / 'wikipedia__chiikawa.md'
        if not wp.exists():
            try:
                wp.write_text(fetch_wikipedia_markdown(), encoding='utf-8')
            except Exception as e:
                print('  Wikipedia fetch failed:', str(e)[:80])


def main() -> None:
    import faiss
    collect_data()
    chunks = build_chunks(DATA_DIR)
    if len(chunks) < 20:
        sys.exit(f'Only {len(chunks)} chunks built: data download probably failed. Aborting build.')
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_DIR / 'chunks.jsonl', 'w', encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    emb = Embedder().encode([c['embed_text'] for c in chunks])
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(INDEX_DIR / 'chiikawa.faiss'))
    print(f'Index: {index.ntotal} chunks x {emb.shape[1]} dims -> {INDEX_DIR}')


if __name__ == '__main__':
    main()
