"""Load trusted local artifacts only: joblib/pickle files can execute code."""
import joblib
import numpy as np
from .data import ROOT, TASKS
from .preprocessing import normalize


class Predictor:
    def __init__(self, root=ROOT, backend="best"):
        self.root = root
        self.backend = backend
        self.models = {}

    def predict(self, text, task="both", top_k=3, threshold=.55):
        if not str(text).strip():
            raise ValueError("Please enter a non-empty message.")
        cleaned = normalize(text)
        result = {"masked_text": cleaned,
                  "privacy_notice": "Rule-based masking only; names and addresses may remain. No cloud calls.",
                  "predictions": {}}
        for name in TASKS if task == "both" else (task,):
            if name not in TASKS:
                raise ValueError("Unknown task")
            backend = self.backend
            if backend == "best":
                import json
                registry = self.root / "models/champions.json"
                backend = json.loads(registry.read_text())[name]["backend"] if registry.exists() else "baseline"
            if name not in self.models:
                if backend == "transformer":
                    from .transformer import TransformerPredictor
                    self.models[name] = {"model": TransformerPredictor(self.root/"models"/f"{name}_transformer"), "mode":"transformer"}
                else:
                    path = self.root / "models" / f"{name}_{backend}.joblib"
                    self.models[name] = joblib.load(path)
            bundle = self.models[name]
            if backend == "baseline":
                model = bundle["model"]
                features = model.named_steps["features"].transform([cleaned])
                probs = model.named_steps["classifier"].predict_proba(features)[0]
                labels = model.classes_
                coverage = features.nnz > 0
            elif backend == "transformer":
                probs = bundle["model"].predict_proba([cleaned])[0]
                labels = bundle["model"].classes_
                coverage = True
            else:
                from .embeddings import document_vectors
                x = document_vectors([cleaned], bundle["vocab"], bundle["word_vectors"],
                                     bundle.get("subword_vectors"), bundle["config"])
                probs = bundle["classifier"].predict_proba(x)[0]
                labels = bundle["classifier"].classes_
                coverage = bool(np.any(x))
            order = np.argsort(probs)[::-1][:top_k]
            top = [{"label": str(labels[i]), "score": float(probs[i])} for i in order]
            result["predictions"][name] = {
                "model": bundle.get("mode", backend), "top": top,
                "needs_review": bool(not coverage or top[0]["score"] < threshold),
                "has_known_features": bool(coverage),
                "score_notice": "Uncalibrated model scores; threshold is a heuristic, not a safety guarantee.",
                "scope": "Hinglish call labels; English transfer unvalidated" if name == "call_scam" else
                         "English complaint taxonomy; Hinglish transfer unvalidated; no benign class",
            }
            if name == "complaint":
                parents = {}
                for label, probability in zip(labels, probs):
                    parent = str(label).split(" :: ")[0]
                    parents[parent] = parents.get(parent, 0.) + float(probability)
                result["predictions"][name]["category_top"] = [
                    {"label":label,"score":value} for label,value in
                    sorted(parents.items(), key=lambda x:x[1], reverse=True)[:top_k]]
        return result
