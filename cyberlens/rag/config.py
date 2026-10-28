"""Paths and model choices. Both models are small enough to run on an Android phone."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "rag_corpus"
INDEX = ROOT / "models" / "rag_index"
# multilingual-e5-small: 118M params, 384-dim, Hindi/Hinglish queries, exports to ONNX/LiteRT for phones.
EMBED_MODEL = "intfloat/multilingual-e5-small"
# Gemma 4 E2B: edge model with LiteRT-LM builds for Android; same weights via Ollama on the laptop.
LLM_MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434"
CHUNK_WORDS = 280
OVERLAP_WORDS = 40
