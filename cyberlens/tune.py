"""Validation-only optimizer comparison. Does not read test rows into fitting."""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from .data import ROOT, load
from .metrics import save_report


def main():
    df = load("complaint")
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    features = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=65000,
                              sublinear_tf=True, dtype=np.float32)
    x = features.fit_transform(tr.text)
    v = features.transform(va.text)
    candidates = {
        "sgd_balanced_1e4": SGDClassifier(loss="log_loss",alpha=1e-4,average=True,class_weight="balanced",max_iter=60,random_state=42,n_jobs=4),
        "sgd_unweighted_1e5": SGDClassifier(loss="log_loss",alpha=1e-5,average=True,max_iter=60,random_state=42,n_jobs=4),
        "linear_svc_balanced": LinearSVC(C=1,class_weight="balanced",max_iter=2000,random_state=42),
    }
    result = {}
    for name, model in candidates.items():
        start = time.perf_counter()
        model.fit(x, tr.label)
        p = model.predict(v)
        result[name] = {"validation_macro_f1": f1_score(va.label,p,average="macro",zero_division=0),
                        "validation_accuracy": accuracy_score(va.label,p), "seconds":time.perf_counter()-start}
        print(name, result[name], flush=True)
        save_report(ROOT/"reports/optimizer_validation.json",result)


if __name__ == "__main__":
    main()
