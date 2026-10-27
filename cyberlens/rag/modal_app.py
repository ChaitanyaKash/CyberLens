"""GPU jobs on Modal, so the laptop stays cool. Run from the repo root:
  modal run -m cyberlens.rag.modal_app::build     # parse + chunk locally, embed on a Modal GPU, save the index
  modal run -m cyberlens.rag.modal_app::answers   # answer the eval questions with Gemma 4 E2B on Modal
  modal run -m cyberlens.rag.modal_app::answers --model qwen2.5:1.5b   # compare a smaller model
"""
import json
import modal
from cyberlens.rag.config import EMBED_MODEL, LLM_MODEL, INDEX, ROOT

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("curl", "zstd")
         .pip_install("sentence-transformers>=3.0", "numpy", "pymupdf", "beautifulsoup4")
         .run_commands(f"python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{EMBED_MODEL}')\"")
         .run_commands("curl -fsSL https://ollama.com/install.sh | sh",
                       f"(ollama serve > /tmp/ollama.log 2>&1 &) && sleep 8 && ollama pull {LLM_MODEL}")
         .add_local_python_source("cyberlens"))
app = modal.App("cyberlens-rag", image=image)


@app.function(gpu="T4", timeout=1200)
def embed(texts):
    from cyberlens.rag.embed import E5Embedder
    return E5Embedder().passages(texts).tobytes()


@app.cls(gpu="L4", timeout=1800, scaledown_window=120)
class Runner:
    model: str = modal.parameter(default=LLM_MODEL)

    @modal.enter()
    def start(self):
        import subprocess, time
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(6)
        subprocess.run(["ollama", "pull", self.model], check=True)  # no-op for the model baked into the image

    @modal.method()
    def run(self, chunks, vectors, meta, corpus_files, questions):
        import numpy as np, pathlib, tempfile
        from cyberlens.rag.agent import GuidanceAgent
        from cyberlens.rag.generate import OllamaLLM
        from cyberlens.rag.index import Index
        tmp = pathlib.Path(tempfile.mkdtemp())
        (tmp / "corpus").mkdir()
        for name, text in corpus_files.items():
            (tmp / "corpus" / name).write_text(text)
        Index(chunks, np.frombuffer(vectors, dtype=np.float16).reshape(len(chunks), -1).astype(np.float32), meta).save(tmp / "index")
        agent = GuidanceAgent(tmp / "index", tmp / "corpus", llm=OllamaLLM(self.model, timeout=300))
        return [{"question": q["question"], **agent.answer(q["question"], q["subtype"], q["date"])} for q in questions]


@app.local_entrypoint()
def build():
    import numpy as np
    from cyberlens.rag.chunk import chunk_pages
    from cyberlens.rag.index import Index
    from cyberlens.rag.load import load_pages

    class Remote:
        name = EMBED_MODEL

    chunks = chunk_pages(load_pages())
    vectors = np.frombuffer(embed.remote([c["embed_text"] for c in chunks]), dtype=np.float32).reshape(len(chunks), -1)
    Index.build(chunks, Remote(), out=INDEX, vectors=vectors)
    print(f"Saved {len(chunks)} chunks x {vectors.shape[1]} dims to {INDEX}")


@app.local_entrypoint()
def answers(limit: int = 12, model: str = LLM_MODEL):
    from cyberlens.rag.config import CORPUS
    from cyberlens.rag.index import Index
    index = Index.load(INDEX)
    questions = [json.loads(l) for l in open(ROOT / "eval" / "rag_questions.jsonl", encoding="utf-8")][:limit]
    corpus_files = {n: (CORPUS / n).read_text(encoding="utf-8") for n in ("manifest.json", "subtype_to_docs.json")}
    out = Runner(model=model).run.remote(index.chunks, index.vectors.astype("float16").tobytes(), index.meta, corpus_files, questions)
    path = ROOT / "reports" / f"rag_answers_{model.replace(':', '_')}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {len(out)} answers to {path}")
