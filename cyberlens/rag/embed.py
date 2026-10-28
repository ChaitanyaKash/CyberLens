"""Text encoders. E5 needs "query: " / "passage: " prefixes because it was trained with them."""
import hashlib
import re
import numpy as np
from .config import EMBED_MODEL


class E5Embedder:
    def __init__(self, model_name=EMBED_MODEL, device=None, batch_size=32):
        from sentence_transformers import SentenceTransformer
        self.name, self.batch_size = model_name, batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def _encode(self, texts):
        return self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def passages(self, texts):
        return self._encode("passage: " + t for t in texts)

    def query(self, text):
        return self._encode(["query: " + text])[0]


class HashEmbedder:
    """Model-free character n-gram hashing. Used in tests and when no encoder is installed."""
    name = "hash-char-3-5"

    def __init__(self, dim=512):
        self.dim = dim

    def _vector(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        text = " " + re.sub(r"\s+", " ", text.lower()) + " "
        for n in (3, 4, 5):
            for i in range(len(text) - n + 1):
                h = int.from_bytes(hashlib.blake2b(text[i:i + n].encode(), digest_size=4).digest(), "little")
                v[h % self.dim] += 1 if h & 1 << 31 else -1
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def passages(self, texts):
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def query(self, text):
        return self._vector(text)


def get_embedder(name):
    return HashEmbedder() if name in ("hash", HashEmbedder.name) else E5Embedder(name if "/" in name else EMBED_MODEL)
