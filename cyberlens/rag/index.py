"""On-disk index: chunks + float16 vectors + BM25 statistics. Search = vector, BM25 or hybrid (RRF)."""
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
import numpy as np

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
STOP = set("the a an of to in and or for on by is are be with as at that this from it its shall which any such has have under".split())
RRF_K = 60


def tokenize(text):
    return [t for t in (m.lower() for m in TOKEN.findall(text)) if t not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.length = np.array([sum(tf.values()) for tf in self.tf], dtype=np.float32)
        self.avg = float(self.length.mean()) if len(docs) else 0.
        df = Counter(t for tf in self.tf for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + .5) / (f + .5)) for t, f in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.tf), dtype=np.float32)
        terms = [t for t in tokenize(query) if t in self.idf]
        norm = self.k1 * (1 - self.b + self.b * self.length / (self.avg or 1))
        for i, tf in enumerate(self.tf):
            s = 0.
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm[i])
            out[i] = s
        return out


def rank(scores, mask, k):
    scores = np.where(mask, scores, -np.inf)
    k = min(k, int(mask.sum()))
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in top[np.argsort(-scores[top])] if np.isfinite(scores[i])]


class Index:
    def __init__(self, chunks, vectors, meta):
        self.chunks, self.vectors, self.meta = chunks, vectors, meta
        self.bm25 = BM25([c["embed_text"] for c in chunks])
        self.doc_of = np.array([c["id"] for c in chunks])

    @classmethod
    def build(cls, chunks, embedder, out=None, vectors=None):
        if vectors is None:
            vectors = embedder.passages([c["embed_text"] for c in chunks])
        meta = {"embedder": embedder.name, "dim": int(vectors.shape[1]), "chunks": len(chunks),
                "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
        index = cls(chunks, vectors.astype(np.float32), meta)
        if out:
            index.save(out)
        return index

    def save(self, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        np.save(out / "embeddings.npy", self.vectors.astype(np.float16))
        (out / "meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        chunks = [json.loads(l) for l in open(path / "chunks.jsonl", encoding="utf-8")]
        vectors = np.load(path / "embeddings.npy").astype(np.float32)
        return cls(chunks, vectors, json.loads((path / "meta.json").read_text()))

    def search(self, query, embedder=None, doc_ids=None, k=8, mode="hybrid", pool=40, per_doc=3):
        mask = np.ones(len(self.chunks), bool) if doc_ids is None else np.isin(self.doc_of, list(doc_ids))
        lists = []
        if mode in ("bm25", "hybrid"):
            lists.append(rank(self.bm25.scores(query), mask, pool))
        if mode in ("vector", "hybrid"):
            if embedder is None or embedder.name != self.meta["embedder"]:
                raise ValueError(f"Index was built with {self.meta['embedder']}; pass the same embedder")
            lists.append(rank(self.vectors @ embedder.query(query), mask, pool))
        # Reciprocal Rank Fusion: robust to BM25 and cosine scores living on different scales.
        fused = Counter()
        for ranking in lists:
            for r, i in enumerate(ranking):
                fused[i] += 1 / (RRF_K + r + 1)
        hits, seen = [], Counter()
        for i, score in fused.most_common():
            doc = self.chunks[i]["id"]
            if seen[doc] < per_doc:
                seen[doc] += 1
                hits.append({**self.chunks[i], "score": round(score, 5)})
            if len(hits) == k:
                break
        return hits
