"""Read the curated corpus (PDF, HTML, text) into cleaned pages with document metadata."""
import json
import re
from collections import Counter
from .config import CORPUS

DOC_FIELDS = ("id", "title", "issuer", "date", "effective_from", "effective_to", "role", "priority", "subtypes", "url")
BOILERPLATE = re.compile(r"Release ID|Visitor Counter|Read this release in")


def load_manifest(corpus=CORPUS):
    docs = []
    for doc in json.loads((corpus / "manifest.json").read_text(encoding="utf-8")):
        path = corpus / doc["file"]
        if doc.get("priority") == "exclude" or not path.exists():
            continue
        docs.append({**{k: doc.get(k, "") for k in DOC_FIELDS}, "path": path, "type": doc["type"]})
    return docs


def read_pdf(path):
    import pymupdf
    with pymupdf.open(path) as pdf:
        return [page.get_text() for page in pdf]


def read_html(path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript", "select", "button"]):
        tag.decompose()
    merged = []
    for line in (re.sub(r"\s+", " ", l).strip() for l in soup.get_text("\n").split("\n")):
        if not line:
            continue
        # Inline links (e.g. a bold "1930") break sentences into short fragments; glue them back.
        if merged and len(merged[-1].split()) >= 8 and not re.search(r"[.:;?!]$", merged[-1]) and len(line.split()) < 8:
            merged[-1] += " " + line
        else:
            merged.append(line)
    keep = []
    for line in merged:
        if len(keep) > 3 and BOILERPLATE.search(line):
            break
        # Navigation and menus are short fragments; release bodies are full sentences.
        if len(line.split()) >= 8 or (keep and line.endswith(":")):
            keep.append(line)
    return ["\n".join(keep)]


def read_txt(path):
    return [path.read_text(encoding="utf-8", errors="ignore")]


def is_garbled(line):
    # Bilingual RBI PDFs extract Hindi headers as replacement/private-use characters.
    bad = sum(ch == "�" or "" <= ch <= "" for ch in line)
    return bad >= 2 or (line and bad / len(line) > .05)


def clean_pages(pages):
    lines = [[l.strip() for l in p.splitlines()] for p in pages]
    counts = Counter(l for page in lines for l in set(page) if l)
    repeated = {l for l, c in counts.items() if len(pages) >= 4 and c >= .6 * len(pages) and len(l) < 120}
    cleaned = []
    for page in lines:
        kept = [l for l in page if l and l not in repeated and not is_garbled(l) and not re.fullmatch(r"(page\s*)?\d{1,3}(\s*of\s*\d+)?", l, re.I)]
        text = "\n".join(kept)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[ \t ]+", " ", text)
        cleaned.append(text.strip())
    return cleaned


def load_pages(corpus=CORPUS):
    readers = {"pdf": read_pdf, "html": read_html, "txt": read_txt}
    pages = []
    for doc in load_manifest(corpus):
        meta = {k: v for k, v in doc.items() if k not in ("path", "type")}
        for number, text in enumerate(clean_pages(readers[doc["type"]](doc["path"])), 1):
            if len(text.split()) >= 5:
                pages.append({**meta, "page": number, "text": text})
    return pages
