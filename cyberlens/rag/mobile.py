"""Export the index as plain files an Android app can load without Python: JSON + raw float16 vectors."""
import json
from pathlib import Path
import numpy as np
from .config import CORPUS
from .generate import SYSTEM
from .playbook import immediate_steps
from .retrieve import DocFilter


def export_mobile(index, out, corpus=CORPUS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keep = ("chunk_id", "id", "title", "issuer", "date", "page", "section", "text", "embed_text")
    (out / "chunks.json").write_text(json.dumps([{k: c.get(k) for k in keep} for c in index.chunks], ensure_ascii=False))
    index.vectors.astype("<f2").tofile(out / "embeddings_f16.bin")
    bm25 = index.bm25
    (out / "bm25.json").write_text(json.dumps({"k1": bm25.k1, "b": bm25.b, "avg_len": bm25.avg, "idf": bm25.idf,
                                               "lengths": bm25.length.tolist(), "tf": [dict(tf) for tf in bm25.tf]},
                                              ensure_ascii=False))
    flt = DocFilter(corpus)
    docs = {i: {k: d.get(k, "") for k in ("title", "issuer", "date", "effective_from", "effective_to", "role", "priority", "url")}
            for i, d in flt.docs.items()}
    playbook = {f"{s}|{r}": immediate_steps(s, r) for s in flt.by_subtype for r in ("2017", "2027")}
    (out / "config.json").write_text(json.dumps({
        "embedder": index.meta["embedder"], "dim": index.meta["dim"], "chunks": len(index.chunks),
        "query_prefix": "query: ", "rrf_k": 60, "always_docs": sorted(flt.always),
        "subtype_to_docs": {k: sorted(v) for k, v in flt.by_subtype.items()}, "docs": docs,
        "playbook": playbook, "system_prompt": SYSTEM}, ensure_ascii=False, indent=1))
    return out
