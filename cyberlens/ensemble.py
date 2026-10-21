"""One predeclared equal-weight sparse/semantic ensemble; no weight search."""
import json
import hashlib
import time
import joblib
import numpy as np
from .data import ROOT, load
from .metrics import evaluate, score, save_report
from .preprocessing import corrupt


class FixedBlend:
    def __init__(self, sparse, semantic):
        self.sparse,self.semantic = sparse,semantic
        self.classes_ = sparse.classes_
        if set(self.classes_) != set(semantic.classes_):
            raise ValueError("Ensemble class vocabularies differ")
        self.semantic_order = np.array([list(semantic.classes_).index(label) for label in self.classes_])

    def predict_proba(self, texts):
        return .5*self.sparse.predict_proba(texts)+.5*self.semantic.predict_proba(texts)[:,self.semantic_order]

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def train(root=ROOT):
    start = time.perf_counter()
    df = load("complaint",root)
    va = df[df.split == "validation"]
    baseline = joblib.load(root/"models/complaint_baseline.joblib")["model"]
    semantic = joblib.load(root/"models/complaint_semantic.joblib")["model"]
    model = FixedBlend(baseline,semantic)
    labels = sorted(set(df.loc[df.split == "train","label"]))
    def predict_validation(values,tag):
        digest = hashlib.sha256((json.dumps(values)+str(semantic.encoder_path)).encode()).hexdigest()
        cache = root/".cache/experiments"/f"complaint_{tag}_{digest}.npy"
        if not cache.exists():
            return model.predict(values)
        semantic_probs = semantic.model.predict_proba(np.load(cache))[:,model.semantic_order]
        probs = .5*baseline.predict_proba(values)+.5*semantic_probs
        return model.classes_[probs.argmax(axis=1)]
    clean = score(va.label,predict_validation(va.text.tolist(),"validation"),labels)
    noisy = score(va.label,predict_validation([corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(va.text)],"validation_typos"),labels)
    selection = (clean["macro_f1"]+noisy["macro_f1"])/2
    base = json.loads((root/"reports/complaint_baseline.json").read_text())
    base = base[base["selected"]]
    reference = (base["validation"]["macro_f1"]+base["validation_typos"]["macro_f1"])/2
    accepted = selection >= reference+.02
    result = {"validation":clean,"validation_typos":noisy,"selection_score":selection,
              "reference_selection_score":reference,"adopt":bool(accepted),
              "protocol":"One fixed equal-weight sparse/semantic ensemble declared before component validation outcomes; no blend-weight search",
              "adoption_rule":"Original clean/noisy validation mean macro-F1 gain >=0.02",
              "limits":"Component scores are uncalibrated; original validation reused in development; no independent external confirmation."}
    joblib.dump({"model":model,"mode":"equal_sparse_semantic","task":"complaint"},root/"models/complaint_ensemble.joblib",compress=3)
    if accepted:
        print("Validation criterion passed; evaluating fixed ensemble on official test",flush=True)
        result.update(evaluate(model.predict,df,sorted(df.label.unique())))
    else:
        result["test_policy"] = "Not evaluated: validation improvement rule failed."
    result["seconds"] = time.perf_counter()-start
    save_report(root/"reports/complaint_ensemble.json",result)
    print("Fixed ensemble validation mean",selection,"gain",selection-reference,"adopt",accepted,flush=True)


if __name__ == "__main__":
    from .ensemble import train as train_module
    train_module()
