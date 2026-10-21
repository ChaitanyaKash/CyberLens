"""Bounded call-model search with nested, template-family-grouped evaluation."""
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from .baseline import make_model
from .data import ROOT, load
from .metrics import score, save_report
from .preprocessing import corrupt
from .template_evaluation import families

CANDIDATES = ("reference_word", "word_l2", "char_l2", "hybrid_l2", "frozen_c001", "frozen_c01")


class FrozenEncoder:
    """Public pretrained encoder; no corpus-dependent parameters are updated."""
    def __init__(self, path):
        import torch
        from transformers import AutoTokenizer, AutoModel
        torch.set_num_threads(4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # This is DistilBERT/WordPiece. Do not apply the unrelated Mistral regex patch.
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, fix_mistral_regex=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, use_safetensors=True).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, texts):
        import torch
        output = []
        with torch.inference_mode():
            for start in range(0,len(texts),24):
                batch = self.tokenizer(list(texts[start:start+24]),padding=True,truncation=True,max_length=192,return_tensors="pt").to(self.device)
                with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.device.type == "cuda"):
                    hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden.float()*mask).sum(1)/mask.sum(1).clamp_min(1)
                pooled = torch.nn.functional.normalize(pooled,p=2,dim=1)
                output.append(pooled.cpu().numpy())
        return np.concatenate(output).astype(np.float32)


def build(name):
    if name == "reference_word":
        return make_model("word")
    if name.startswith("frozen_"):
        c = .01 if name == "frozen_c001" else .1
        return make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight="balanced",max_iter=500,random_state=42))
    mode = name.replace("_l2","")
    features = make_model(mode).named_steps["features"]
    return Pipeline([("features",features),("classifier",LogisticRegression(
        C=1,class_weight="balanced",max_iter=500,tol=.001,random_state=42))])


def inputs(name,texts,vectors,idx):
    return vectors[idx] if name.startswith("frozen_") else [texts[i] for i in idx]


def select_inner(texts,noisy,y,groups,vectors,noisy_vectors,indices,seed):
    cv = StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed)
    results = {name:[] for name in CANDIDATES}
    for tr_local,va_local in cv.split(indices,y[indices],groups=groups[indices]):
        tr,va = indices[tr_local],indices[va_local]
        if set(groups[tr]) & set(groups[va]):
            raise RuntimeError("An inner validation family leaked into training")
        if set(y[tr]) != set(y[indices]):
            raise RuntimeError("An inner training fold is missing a supported class")
        for name in CANDIDATES:
            model = build(name)
            model.fit(inputs(name,texts,vectors,tr),y[tr])
            a = model.predict(inputs(name,texts,vectors,va))
            b = model.predict(inputs(name,noisy,noisy_vectors,va))
            labels = sorted(set(y[indices]))
            results[name].append(float((f1_score(y[va],a,labels=labels,average="macro",zero_division=0)+
                                        f1_score(y[va],b,labels=labels,average="macro",zero_division=0))/2))
    means = {name:float(np.mean(values)) for name,values in results.items()}
    chosen = max(CANDIDATES,key=lambda name:means[name])
    return chosen,{name:{"folds":results[name],"mean":means[name]} for name in CANDIDATES}


class RobustClassifier:
    def __init__(self,variant,model,encoder_path):
        self.variant,self.model,self.encoder_path = variant,model,str(encoder_path)
        self.classes_ = model.classes_
        self._encoder = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_encoder"] = None
        return state

    def predict_proba(self,texts):
        if self.variant.startswith("frozen_"):
            if self._encoder is None:
                self._encoder = FrozenEncoder(self.encoder_path)
            return self.model.predict_proba(self._encoder.encode(list(texts)))
        return self.model.predict_proba(texts)

    def predict(self,texts):
        return self.classes_[self.predict_proba(texts).argmax(1)]


def bootstrap_difference(y,new,reference,groups,labels,repetitions=500):
    rng = np.random.default_rng(42)
    unique,inverse = np.unique(groups,return_inverse=True)
    deltas = []
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))
        weights = counts[inverse]
        a = f1_score(y,new,labels=labels,average="macro",sample_weight=weights,zero_division=0)
        b = f1_score(y,reference,labels=labels,average="macro",sample_weight=weights,zero_division=0)
        deltas.append(a-b)
    return {"method":"paired template-family bootstrap; approximate, not independent external evidence",
            "repetitions":repetitions,"lower_95":float(np.quantile(deltas,.025)),"upper_95":float(np.quantile(deltas,.975))}


def promotion_allowed(gain,clean_lower_bound,typo_f1,reference_typo_f1):
    return bool(gain >= .02 and clean_lower_bound > 0 and typo_f1 >= reference_typo_f1-.005)


@threadpool_limits.wrap(limits=1)
def train(root=ROOT):
    start = time.perf_counter()
    df = load("call_scam",root).drop_duplicates("group").reset_index(drop=True)
    texts = df.text.tolist()
    noisy = [corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(texts)]
    y = df.label.to_numpy()
    groups = families(texts,.8)
    labels = sorted(set(y))
    path = root/".cache/pretrained/distilmbert"
    digest = hashlib.sha256((json.dumps(texts)+json.dumps(noisy)+str(path)).encode()).hexdigest()
    cache = root/".cache/experiments"/f"call_frozen_{digest}.npz"
    cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        with np.load(cache) as stored:
            vectors,noisy_vectors = stored["clean"],stored["noisy"]
    else:
        encoder = FrozenEncoder(path)
        vectors,noisy_vectors = encoder.encode(texts),encoder.encode(noisy)
        np.savez_compressed(cache,clean=vectors,noisy=noisy_vectors)
        del encoder
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    predicted = np.empty(len(y),dtype=object)
    noisy_predicted = np.empty(len(y),dtype=object)
    reference = np.empty(len(y),dtype=object)
    reference_noisy = np.empty(len(y),dtype=object)
    seen = np.zeros(len(y),dtype=int)
    folds = []
    for fold,(tr,te) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(texts,y,groups)):
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("An outer assessment family leaked into training")
        if set(y[tr]) != set(labels):
            raise RuntimeError("An outer training fold is missing a supported class")
        chosen,search = select_inner(texts,noisy,y,groups,vectors,noisy_vectors,tr,100+fold)
        model = build(chosen)
        model.fit(inputs(chosen,texts,vectors,tr),y[tr])
        predicted[te] = model.predict(inputs(chosen,texts,vectors,te))
        noisy_predicted[te] = model.predict(inputs(chosen,noisy,noisy_vectors,te))
        baseline = make_model("word").fit([texts[i] for i in tr],y[tr])
        reference[te] = baseline.predict([texts[i] for i in te])
        reference_noisy[te] = baseline.predict([noisy[i] for i in te])
        seen[te] += 1
        folds.append({"fold":fold,"selected_in_inner_cv":chosen,"inner_search":search,
                      "train_rows":len(tr),"assessment_rows":len(te),"overlapping_families":0})
        print("Outer fold",fold,"selected",chosen,"assessment rows",len(te),flush=True)
    if not np.all(seen == 1):
        raise RuntimeError("Every row must be assessed exactly once")
    clean,typos = score(y,predicted,labels),score(y,noisy_predicted,labels)
    old_clean,old_typos = score(y,reference,labels),score(y,reference_noisy,labels)
    uncertainty = bootstrap_difference(y,predicted,reference,groups,labels)
    gain = (clean["macro_f1"]+typos["macro_f1"]-old_clean["macro_f1"]-old_typos["macro_f1"])/2
    # Criteria fixed before looking at these assessment results.
    accepted = promotion_allowed(gain,uncertainty["lower_95"],typos["macro_f1"],old_typos["macro_f1"])
    final_variant,final_search = select_inner(texts,noisy,y,groups,vectors,noisy_vectors,np.arange(len(y)),2026)
    final_model = build(final_variant).fit(inputs(final_variant,texts,vectors,np.arange(len(y))),y)
    wrapper = RobustClassifier(final_variant,final_model,path)
    joblib.dump({"model":wrapper,"mode":final_variant,"task":"call_scam"},root/"models/call_scam_robust.joblib",compress=3)
    result = {"protocol":"5 outer family-grouped folds; 3 inner family-grouped folds; six predeclared candidates",
              "family_threshold":.8,"families":len(set(groups)),"rows":len(y),"candidates":list(CANDIDATES),
              "nested":clean,"nested_typos":typos,"reference":old_clean,"reference_typos":old_typos,
              "paired_uncertainty":uncertainty,"mean_clean_noisy_gain":gain,"adopt":bool(accepted),
              "adoption_rule":"Mean gain >=0.02; paired clean family-bootstrap lower bound >0; typo F1 drop <=0.005",
              "folds":folds,"deployment_variant":final_variant,"deployment_inner_search":final_search,
              "deployment_training_rows":len(y),"seconds":time.perf_counter()-start,
              "limits":"Previously explored development corpus, heuristic families, no independent new dataset. Frozen encoder saw no task training. Fold scalers and text vocabularies fit training only. Refit-on-all artifact is distinct from CV assessment models."}
    save_report(root/"reports/call_scam_robust.json",result)
    print("Nested clean",clean["macro_f1"],"typos",typos["macro_f1"],"gain",gain,"adopt",accepted,flush=True)


if __name__ == "__main__":
    from .robust import train as train_module
    train_module()
