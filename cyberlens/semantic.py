"""Frozen multilingual complaint representations, selected inside training only."""
import hashlib
import json
import time
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .robust import FrozenEncoder, RobustClassifier
from .data import ROOT, load
from .preprocessing import corrupt
from .metrics import score, save_report


@threadpool_limits.wrap(limits=1)
def train(root=ROOT):
    start = time.perf_counter()
    df = load("complaint",root)
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "validation"].reset_index(drop=True)
    labels = sorted(tr.label.unique())
    a,b = train_test_split(np.arange(len(tr)),test_size=.2,stratify=tr.label,random_state=2026)
    texts = tr.text.tolist()
    encoder_path = root/".cache/pretrained/distilmbert"
    cache_dir = root/".cache/experiments"
    cache_dir.mkdir(parents=True,exist_ok=True)
    encoder = None

    def encode(values,tag):
        nonlocal encoder
        digest = hashlib.sha256((json.dumps(values)+str(encoder_path)).encode()).hexdigest()
        cache = cache_dir/f"complaint_{tag}_{digest}.npy"
        if cache.exists():
            return np.load(cache)
        if encoder is None:
            encoder = FrozenEncoder(encoder_path)
        chunks = []
        for offset in range(0,len(values),2400):
            chunks.append(encoder.encode(values[offset:offset+2400]))
            print(tag,"encoded",min(offset+2400,len(values)),"/",len(values),flush=True)
        vectors = np.concatenate(chunks)
        np.save(cache,vectors)
        return vectors

    x = encode(texts,"train")
    inner_noisy = encode([corrupt(texts[i],seed=42+j,rate=.25) for j,i in enumerate(b)],"inner_typos")
    search = {}
    for c in (.01,.1):
        model = make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight="balanced",max_iter=500,tol=.001,random_state=42))
        model.fit(x[a],tr.label.iloc[a])
        clean = score(tr.label.iloc[b],model.predict(x[b]),labels)
        noisy = score(tr.label.iloc[b],model.predict(inner_noisy),labels)
        search[str(c)] = {"inner_validation":clean,"inner_validation_typos":noisy,
                          "selection_score":(clean["macro_f1"]+noisy["macro_f1"])/2}
        print("C",c,"inner selection",search[str(c)]["selection_score"],flush=True)
    chosen = max(search,key=lambda k:search[k]["selection_score"])
    model = make_pipeline(StandardScaler(),LogisticRegression(C=float(chosen),class_weight="balanced",max_iter=500,tol=.001,random_state=42)).fit(x,tr.label)
    clean = score(va.label,model.predict(encode(va.text.tolist(),"validation")),labels)
    noisy = score(va.label,model.predict(encode([corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(va.text)],"validation_typos")),labels)
    selection = (clean["macro_f1"]+noisy["macro_f1"])/2
    reference = json.loads((root/"reports/complaint_baseline.json").read_text())
    reference = reference[reference["selected"]]
    reference = (reference["validation"]["macro_f1"]+reference["validation_typos"]["macro_f1"])/2
    accepted = selection >= reference+.02
    wrapper = RobustClassifier("frozen_complaint",model,encoder_path)
    joblib.dump({"model":wrapper,"mode":f"frozen_c{chosen}","task":"complaint"},root/"models/complaint_semantic.joblib",compress=3)
    result = {"search":search,"selected_C":float(chosen),"validation":clean,"validation_typos":noisy,
              "selection_score":selection,"reference_selection_score":reference,"adopt":bool(accepted),
              "protocol":"Frozen public multilingual encoder; C selected on original-training inner split; one original-validation check",
              "adoption_rule":"Mean clean/noisy original-validation macro-F1 gain >=0.02",
              "inner_train_rows":len(a),"inner_validation_rows":len(b),
              "limits":"Previously used validation; frozen pretraining is not task fitting; fold scaler trained only on inner training. No independent external confirmation."}
    if accepted:
        print("Validation criterion passed; evaluating selected semantic candidate on official test",flush=True)
        te = df[df.split == "test"]
        test_labels = sorted(set(labels)|set(te.label))
        result["test"] = score(te.label,model.predict(encode(te.text.tolist(),"test")),test_labels)
        result["test_typos"] = score(te.label,model.predict(encode([corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(te.text)],"test_typos")),test_labels)
    else:
        result["test_policy"] = "Not evaluated: validation improvement rule failed."
    result["seconds"] = time.perf_counter()-start
    save_report(root/"reports/complaint_semantic.json",result)
    print("Semantic validation mean",selection,"gain",selection-reference,"adopt",accepted,flush=True)


if __name__ == "__main__":
    train()
