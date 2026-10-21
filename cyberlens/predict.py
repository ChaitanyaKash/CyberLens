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
        self.quality = {}

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
            if backend == "hierarchical" and name == "call_scam":
                backend = "baseline"
            if (backend == "robust" and name == "complaint") or (backend in ("regularized","margin","semantic","ensemble","continued","improved","supervised","adjusted") and name == "call_scam"):
                backend = "baseline"
            cache_key = (name,backend)
            if cache_key not in self.models:
                if backend == "supervised":
                    from .supervised import SupervisedPredictor
                    folder = self.root/"models/complaint_supervised"
                    if not (folder/"metadata.json").exists():
                        raise ValueError("No supervised deployment model is available. Rejected internal experiments do not create one; use best or baseline.")
                    self.models[cache_key] = {"model":SupervisedPredictor(folder),"mode":"supervised_subwords"}
                elif backend in ("transformer","continued"):
                    from .transformer import TransformerPredictor
                    self.models[cache_key] = {"model": TransformerPredictor(self.root/"models"/f"{name}_{backend}"), "mode":backend}
                else:
                    path = self.root / "models" / f"{name}_{backend}.joblib"
                    if backend == "improved" and not path.exists():
                        raise ValueError("No improved deployment model is available. Rejected internal experiments do not create one; use best or baseline.")
                    self.models[cache_key] = joblib.load(path)
            bundle = self.models[cache_key]
            if backend == "adjusted":
                model = bundle["model"]
                features = model.named_steps["features"].transform([cleaned])
                probs = model.predict_proba_features(features)[0]
                labels = model.classes_
                coverage = features.nnz > 0
            elif backend in ("baseline","regularized","margin","improved"):
                model = bundle["model"]
                features = model.named_steps["features"].transform([cleaned])
                probs = model.named_steps["classifier"].predict_proba(features)[0]
                labels = model.classes_
                coverage = features.nnz > 0
            elif backend in ("robust","semantic","ensemble","supervised"):
                probs = bundle["model"].predict_proba([cleaned])[0]
                labels = bundle["model"].classes_
                coverage = True
            elif backend == "hierarchical":
                model = bundle["model"]
                probs = model.predict_proba([cleaned])[0]
                labels = model.classes_
                coverage = model.base.named_steps["features"].transform([cleaned]).nnz > 0
            elif backend in ("transformer","continued"):
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
            if cache_key not in self.quality:
                import json
                experimental = False
                report_path = self.root/"reports"/f"{name}_{backend}.json"
                family_path = self.root/"reports/call_family_evaluation.json"
                if name == "call_scam" and backend == "baseline" and family_path.exists():
                    quality = json.loads(family_path.read_text())["oof"]["per_class"]
                    source = "template-family cross-validation"
                elif backend == "robust" and report_path.exists():
                    report = json.loads(report_path.read_text())
                    quality = report["nested"]["per_class"]
                    experimental = not report.get("adopt",False)
                    source = "nested template-family cross-validation"
                elif report_path.exists():
                    report = json.loads(report_path.read_text())
                    if backend == "baseline":
                        report = report[report["selected"]]
                    quality,source = report.get("validation",{}).get("per_class",{}),"validation"
                    experimental = backend in ("regularized","margin","semantic","ensemble","continued","improved","supervised","adjusted") and not report.get("adopt",False)
                else:
                    quality,source = {},"unavailable"
                self.quality[cache_key] = quality,source,experimental
            quality,source,experimental = self.quality[cache_key]
            evidence = quality.get(top[0]["label"],{})
            weak_class = evidence.get("f1-score",0) < .5 or evidence.get("support",0) < 20
            result["predictions"][name] = {
                "model": bundle.get("mode", backend), "top": top,
                "needs_review": bool(not coverage or top[0]["score"] < threshold or weak_class or experimental),
                "experimental": experimental,
                "has_known_features": bool(coverage),
                "class_evidence": {"source":source,"f1":evidence.get("f1-score"),"support":evidence.get("support"),
                                   "notice":"Class-level evaluation summary, not this message's probability of correctness"},
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
                broad = result["predictions"][name]["category_top"][0]["label"]
                parent_mismatch = not top[0]["label"].startswith(broad+" :: ")
                result["predictions"][name]["needs_review"] |= parent_mismatch
                result["predictions"][name]["suggested_type"] = broad+" (subtype needs review)" if weak_class or parent_mismatch else top[0]["label"]
            else:
                result["predictions"][name]["suggested_type"] = top[0]["label"]
        return result
