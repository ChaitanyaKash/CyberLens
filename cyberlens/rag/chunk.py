"""Section-aware chunking: legal text is split on paragraph numbers so one rule stays in one chunk."""
import re
from .config import CHUNK_WORDS, OVERLAP_WORDS

SECTION = re.compile(r"^\s*(?:(\d{1,3}[A-Z]{0,2})\.(?:\s|$)|(\d{1,3}\.\d{1,2})\s|(Q(?:uery)?\.?\s*\d+)[.:]?\s|(Section\s+\d+[A-Z]?)|(Chapter\s+[IVXLC\d]+)|(Point\s+\d+))", re.I)


def sections(pages):
    """Yield (page, section_label, text) blocks in document order."""
    label, block, start = "", [], None
    for page in pages:
        for line in page["text"].splitlines():
            m = SECTION.match(line)
            if m and block:
                yield start, label, "\n".join(block)
                block = []
            if m:
                label = next(g for g in m.groups() if g)
            if not block:
                start = page["page"]
            block.append(line)
    if block:
        yield start, label, "\n".join(block)


def header(doc, page, section):
    where = f"para {section}" if section else f"page {page}"
    return f"[{doc['issuer']} | {doc['title']} | {where}]"


def chunk_document(pages, max_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    doc = {k: v for k, v in pages[0].items() if k not in ("page", "text")}
    chunks = []

    def emit(words, where):
        if len(words) < 12:
            return
        page, section = where
        text = " ".join(words)
        chunks.append({**doc, "chunk_id": f"{doc['id']}#{len(chunks)}", "page": page, "section": section,
                       "text": text, "embed_text": f"{header(doc, page, section)} {text}"})

    buffer, first = [], None
    for page, label, text in sections(pages):
        block = text.split()
        if buffer and len(buffer) + len(block) > max_words:
            emit(buffer, first)
            buffer, first = [], None
        if len(block) > max_words:
            # A section longer than one chunk is cut with overlap so no sentence is lost at an edge.
            for i in range(0, len(block), max_words - overlap):
                emit(block[i:i + max_words], (page, label))
                if i + max_words >= len(block):
                    break
            continue
        first = first or (page, label)
        buffer += block
    if buffer:
        emit(buffer, first)
    return chunks


def chunk_pages(pages, **kw):
    by_doc = {}
    for page in pages:
        by_doc.setdefault(page["id"], []).append(page)
    return [c for doc_pages in by_doc.values() for c in chunk_document(doc_pages, **kw)]
