"""A conditional financial-fraud expert, preserving the base model's parent mass."""
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from .data import ROOT, load
from .metrics import evaluate, save_report
from .preprocessing import corrupt

PREFIX = "Online Financial Fraud :: "


class FinancialExpert:
    def __init__(self, base, expert):
        self.base, self.expert = base, expert
        self.classes_ = base.classes_

    def combine(self, base_probs, conditional):
        combined = base_probs.copy()
        indices = [i for i,label in enumerate(self.classes_) if str(label).startswith(PREFIX)]
        mass = combined[:,indices].sum(1)
        order = [list(self.expert.classes_).index(self.classes_[i]) for i in indices]
        combined[:,indices] = mass[:,None]*conditional[:,order]
        return combined

    def predict_proba(self, texts):
        return self.combine(self.base.predict_proba(texts), self.expert.predict_proba(texts))

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(1)]


def train(root=ROOT):
    start = time.perf_counter()
    df = load("complaint",root)
    tr = df[df.split == "train"]
    va = df[df.split == "validation"]
    financial = tr[tr.label.str.startswith(PREFIX)]
    base = joblib.load(root/"models/complaint_baseline.joblib")["model"]
    features = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=65000,sublinear_tf=True,dtype=np.float32)
    x = features.fit_transform(financial.text)
    noisy = [corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(va.text)]
    v, vn = features.transform(va.text), features.transform(noisy)
    base_v, base_vn = base.predict_proba(va.text), base.predict_proba(noisy)
    labels = sorted(tr.label.unique())
    best, best_score, comparison = None,-1,{}
    for c, weights in ((1,"balanced"),(4,"balanced"),(4,None)):
        classifier = OneVsRestClassifier(LogisticRegression(solver="liblinear",C=c,
            class_weight=weights,max_iter=300,tol=.001,random_state=42),n_jobs=1)
        classifier.fit(x,financial.label)
        expert = Pipeline([("features",features),("classifier",classifier)])
        model = FinancialExpert(base,expert)
        clean = model.classes_[model.combine(base_v,classifier.predict_proba(v)).argmax(1)]
        noisy_pred = model.classes_[model.combine(base_vn,classifier.predict_proba(vn)).argmax(1)]
        a = f1_score(va.label,clean,labels=labels,average="macro",zero_division=0)
        b = f1_score(va.label,noisy_pred,labels=labels,average="macro",zero_division=0)
        name = f"C={c}, class_weight={weights}"
        comparison[name] = {"validation_macro_f1":a,"validation_typos_macro_f1":b}
        print(name,"clean",round(a,4),"typos",round(b,4),flush=True)
        if (a+b)/2 > best_score:
            best, best_score = model,(a+b)/2
    result = evaluate(best.predict,df,sorted(df.label.unique()))
    result.update({"financial_training_rows":len(financial),"financial_labels":int(financial.label.nunique()),
                   "candidates":comparison,"selection_score":best_score,"seconds":time.perf_counter()-start,
                   "mechanism":"Keep parent probability mass; redistribute financial subtypes with a training-only expert"})
    joblib.dump({"model":best,"mode":"hierarchical","task":"complaint"},root/"models/complaint_hierarchical.joblib",compress=3)
    save_report(root/"reports/complaint_hierarchical.json",result)
    print("Hierarchical test macro-F1:",result["test"]["macro_f1"],flush=True)


if __name__ == "__main__":
    # Serialize custom estimator classes under an importable module, not __main__.
    from .hierarchical import train as train_from_module
    train_from_module()
