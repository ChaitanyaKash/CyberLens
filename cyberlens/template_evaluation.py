"""Harder call evaluation: whole near-duplicate template families stay together.

The estimator is frozen to the previously selected sparse baseline. This is
grouped cross-validation, not an independent real-world benchmark.
"""
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import StratifiedGroupKFold
from .baseline import make_model
from .data import ROOT, load
from .metrics import score, save_report
from .preprocessing import corrupt


def families(texts, threshold=.8):
    # Features used ONLY for duplicate grouping; never reused by a classifier.
    x = TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=1).fit_transform(texts)
    similarities = linear_kernel(x,x)
    return connected_components(csr_matrix(similarities >= threshold),directed=False)[1]


def main():
    df = load("call_scam").drop_duplicates("group").reset_index(drop=True)
    audit = {}
    for threshold in (.75,.8,.85,.9):
        group = families(df.text,threshold)
        counts = df.assign(family=group).groupby("label").family.nunique().to_dict()
        audit[str(threshold)] = {"families":int(len(set(group))),"families_per_class":counts}
    print("Family counts:",json.dumps(audit),flush=True)
    groups = families(df.text,.8)
    if len(set(groups)) < 5:
        raise RuntimeError("Fewer than five independent template groups; a five-fold estimate is not possible")
    selected = json.loads((ROOT/"reports/call_scam_baseline.json").read_text())["selected"]
    cv = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    labels = sorted(df.label.unique())
    predictions = np.empty(len(df),dtype=object)
    noisy_predictions = np.empty(len(df),dtype=object)
    seen = np.zeros(len(df),dtype=int)
    folds = []
    for fold,(tr,te) in enumerate(cv.split(df.text,df.label,groups=groups)):
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("Template family crossed train/test")
        model = make_model(selected)
        model.fit(df.text.iloc[tr],df.label.iloc[tr])
        predictions[te] = model.predict(df.text.iloc[te])
        noisy = [corrupt(df.text.iloc[i],seed=42+int(i),rate=.25) for i in te]
        noisy_predictions[te] = model.predict(noisy)
        seen[te] += 1
        missing = sorted(set(df.label.iloc[te])-set(df.label.iloc[tr]))
        folds.append({"fold":fold,"train_rows":len(tr),"test_rows":len(te),
                      "train_families":len(set(groups[tr])),"test_families":len(set(groups[te])),
                      "test_labels_absent_from_training":missing})
        print("Fold",fold,"held-out",len(te),"unsupported labels",missing,flush=True)
    if not np.all(seen == 1):
        raise RuntimeError("Every example must be evaluated exactly once")
    result = {"protocol":"Five-fold stratified group CV, near-duplicate family connected components; frozen previously selected estimator",
              "threshold":.8,"group_features":"Character 3-5-gram TF-IDF cosine; grouping only",
              "estimator":selected,"rows":len(df),"families":len(set(groups)),"sensitivity":audit,"folds":folds,
              "oof":score(df.label,predictions,labels),"oof_typos":score(df.label,noisy_predictions,labels),
              "limitation":"Heuristic text similarity groups, not human-verified semantic families. Some classes may have too few independent families; fold coverage is disclosed. Not a real-world benchmark."}
    save_report(ROOT/"reports/call_family_evaluation.json",result)
    print("Family-held-out macro-F1",result["oof"]["macro_f1"],flush=True)


if __name__ == "__main__":
    main()
