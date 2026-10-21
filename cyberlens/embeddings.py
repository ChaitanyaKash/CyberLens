"""Skip-gram negative sampling and a character-subword extension, written in PyTorch.

No gensim or fastText training call. Autograd computes gradients; sampling,
vocabulary, n-grams, objective, training and document pooling are implemented here.
"""
import hashlib
import time
from collections import Counter
from functools import lru_cache
import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import ROOT, TASKS, load
from .metrics import evaluate, save_report
from .preprocessing import tokens


@lru_cache(maxsize=120000)
def ngrams(word, buckets=50000, max_grams=24):
    if word.startswith("<") and word.endswith(">"):
        return ()  # PII placeholders are atomic, not spelling variants.
    padded = "<" + word + ">"
    values = []
    for n in range(3, 6):
        for i in range(len(padded)-n+1):
            part = padded[i:i+n]
            h = int.from_bytes(hashlib.blake2b(part.encode(), digest_size=8).digest(), "little")
            values.append(1+h % buckets)
    # Stable truncation across processes, unlike Python's randomized hash().
    return tuple(dict.fromkeys(values))[:max_grams]


class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim, ngram_table=None, buckets=50000):
        super().__init__()
        self.input = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.output = nn.Embedding(vocab_size, dim, padding_idx=0)
        nn.init.uniform_(self.input.weight, -.5/dim, .5/dim)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.input.weight[0].zero_()
        self.subwords = None
        if ngram_table is not None:
            self.subwords = nn.Embedding(buckets+1, dim, padding_idx=0)
            nn.init.uniform_(self.subwords.weight, -.5/dim, .5/dim)
            with torch.no_grad():
                self.subwords.weight[0].zero_()
            self.register_buffer("ngram_table", torch.tensor(ngram_table, dtype=torch.long))

    def center_vectors(self, centers):
        x = self.input(centers)
        if self.subwords is not None:
            grams = self.ngram_table[centers]
            denom = (grams != 0).sum(1).clamp_min(1).unsqueeze(1)
            x = x + self.subwords(grams).sum(1) / denom
        return x

    def forward(self, centers, contexts, negatives):
        v = self.center_vectors(centers)
        positive = (v*self.output(contexts)).sum(-1)
        negative = torch.bmm(self.output(negatives), v.unsqueeze(-1)).squeeze(-1)
        return -(F.logsigmoid(positive) + F.logsigmoid(-negative).sum(-1)).mean()


def make_pairs(documents, vocab, max_pairs, seed=42, window=4):
    rng = np.random.default_rng(seed)
    docs = [[vocab.get(t, 0) for t in doc] for doc in documents if len(doc) >= 2]
    lengths = np.array([len(d) for d in docs])
    ends = np.cumsum(lengths)
    if not len(ends):
        raise ValueError("No usable context pairs")
    starts = np.r_[0, ends[:-1]]
    flat = np.array([t for d in docs for t in d], dtype=np.int64)
    # Uniform center-position sampling covers the corpus instead of stopping at its first rows.
    center_pos = rng.integers(0, len(flat), size=max_pairs*2)
    doc_ids = np.searchsorted(ends, center_pos, side="right")
    offsets = rng.choice(np.r_[np.arange(-window, 0), np.arange(1, window+1)], size=len(center_pos))
    context_pos = center_pos + offsets
    valid = (context_pos >= starts[doc_ids]) & (context_pos < ends[doc_ids])
    center_pos, context_pos = center_pos[valid], context_pos[valid]
    centers, contexts = flat[center_pos], flat[context_pos]
    keep = (centers != 0) & (contexts != 0)
    return centers[keep][:max_pairs], contexts[keep][:max_pairs]


def document_vectors(texts, vocab, word_vectors, subword_vectors=None, config=None):
    config = config or {}
    vectors = np.zeros((len(texts), word_vectors.shape[1]), dtype=np.float32)
    @lru_cache(maxsize=60000)
    def representation(token):
        wid = vocab.get(token)
        vector = word_vectors[wid].copy() if wid is not None else np.zeros(word_vectors.shape[1], np.float32)
        known = wid is not None
        if subword_vectors is not None:
            ids = ngrams(token, config["buckets"], config["max_grams"])
            if ids:
                part = subword_vectors[list(ids)]
                vector += part.mean(0)
                known |= bool(np.any(part))
        return vector if known else None
    for i, text in enumerate(texts):
        word_representations = []
        for token in tokens(text):
            vector = representation(token)
            if vector is not None:
                word_representations.append(vector)
        if word_representations:
            vectors[i] = np.mean(word_representations, axis=0)
    return vectors


def train(tasks=TASKS, root=ROOT, mode="word2vec", epochs=3, max_pairs=300000):
    if epochs < 1 or max_pairs < 1:
        raise ValueError("epochs and max_pairs must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for task in tasks:
        start = time.perf_counter()
        df = load(task, root)
        tr = df[df.split == "train"]
        documents = [tokens(t) for t in tr.text]
        counts = Counter(t for doc in documents for t in doc)
        vocab = {word: i+1 for i, (word, count) in enumerate(
            [(w, c) for w, c in counts.most_common(40000) if c >= 2])}
        config = {"dim": 96, "buckets": 50000, "max_grams": 24, "negative_samples": 5,
                  "window": 4, "seed": 42, "max_pairs": max_pairs, "epochs": epochs,
                  "min_count": 2, "word_subsampling": False}
        table = None
        if mode == "subword":
            table = np.zeros((len(vocab)+1, config["max_grams"]), dtype=np.int64)
            for word, wid in vocab.items():
                ids = ngrams(word, config["buckets"], config["max_grams"])
                table[wid, :len(ids)] = ids
        centers, contexts = make_pairs(documents, vocab, max_pairs)
        if not len(centers):
            raise ValueError("Vocabulary leaves no training pairs")
        model = SkipGram(len(vocab)+1, config["dim"], table, config["buckets"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        weights = torch.zeros(len(vocab)+1, device=device)
        for word, wid in vocab.items():
            weights[wid] = counts[word] ** .75
        weights /= weights.sum()
        centers = torch.tensor(centers, device=device)
        contexts = torch.tensor(contexts, device=device)
        history = []
        print(f"{task}/{mode}: {len(vocab)} words, {len(centers)} pairs, {device}", flush=True)
        for epoch in range(epochs):
            order = torch.randperm(len(centers), device=device)
            total, seen = 0., 0
            for batch in order.split(2048):
                negatives = torch.multinomial(weights, len(batch)*5, replacement=True).reshape(-1, 5)
                optimizer.zero_grad(set_to_none=True)
                loss = model(centers[batch], contexts[batch], negatives)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite skip-gram loss")
                loss.backward()
                optimizer.step()
                total += loss.item()*len(batch)
                seen += len(batch)
            history.append(total/seen)
            print(f"Epoch {epoch+1}: SGNS loss={history[-1]:.4f}", flush=True)
        word_vectors = model.input.weight.detach().cpu().numpy().copy()
        subword_vectors = None
        if model.subwords is not None:
            subword_vectors = model.subwords.weight.detach().cpu().numpy().copy()
            # Untrained hash buckets must not become random evidence for OOV words.
            used = np.zeros(len(subword_vectors), dtype=bool)
            used[np.unique(table)] = True
            subword_vectors[~used] = 0
            subword_vectors[0] = 0
        print("Pooling document vectors and fitting classifier", flush=True)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(
            C=1, max_iter=1000, tol=.005, class_weight="balanced", random_state=42))
        def embed(texts):
            return document_vectors(list(texts), vocab, word_vectors, subword_vectors, config)
        classifier.fit(embed(tr.text), tr.label)
        labels = sorted(df.label.unique())
        results = evaluate(lambda texts: classifier.predict(embed(texts)), df, labels)
        results.update({"mode": mode, "config": config, "sgns_loss": history, "device": str(device),
                        "vocabulary": len(vocab), "sampled_pairs": len(centers),
                        "seconds": time.perf_counter()-start,
                        "cuda_peak_mb": torch.cuda.max_memory_allocated()/1024**2 if device.type == "cuda" else 0})
        bundle = {"vocab": vocab, "word_vectors": word_vectors, "subword_vectors": subword_vectors,
                  "config": config, "classifier": classifier, "mode": mode, "task": task}
        joblib.dump(bundle, root / "models" / f"{task}_{mode}.joblib", compress=3)
        save_report(root / "reports" / f"{task}_{mode}.json", results)
        print("Test macro F1:", results["test"]["macro_f1"], flush=True)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
