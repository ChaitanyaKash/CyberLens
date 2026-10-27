"""Retrieval evaluation on a hand-written question set: doc-level Recall@k and MRR, per search mode."""
import json
from .config import CORPUS, ROOT
from .retrieve import DocFilter

MODES = [("bm25", False), ("vector", False), ("hybrid", False), ("bm25", True), ("vector", True), ("hybrid", True)]


def evaluate(index, embedder, questions_path=ROOT / "eval" / "rag_questions.jsonl", corpus=CORPUS, k=5, out=None):
    questions = [json.loads(l) for l in open(questions_path, encoding="utf-8") if l.strip()]
    flt = DocFilter(corpus)
    present = set(index.doc_of.tolist())
    results = {}
    for mode, filtered in MODES:
        if mode != "bm25" and embedder is None:
            continue
        hits_at_k, rr, misses = 0, 0., []
        scored = 0
        for q in questions:
            gold = set(q["gold"]) & present
            if not gold:
                continue
            scored += 1
            ids = flt.allowed(q["subtype"], q["date"], include_background=True) if filtered else None
            docs = [h["id"] for h in index.search(q["question"], embedder, ids, k=10, mode=mode, per_doc=1)]
            first = next((r for r, d in enumerate(docs, 1) if d in gold), None)
            hits_at_k += bool(first and first <= k)
            rr += 1 / first if first else 0
            if not first or first > k:
                misses.append(q["question"][:80])
        name = f"{mode}{'+filter' if filtered else ''}"
        results[name] = {f"recall@{k}": round(hits_at_k / scored, 3), "mrr": round(rr / scored, 3),
                         "questions": scored, "misses": misses}
    report = {"embedder": index.meta["embedder"], "chunks": len(index.chunks), "results": results}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
