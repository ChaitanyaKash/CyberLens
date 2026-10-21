"""A small English-classifier comparison using validation only for selection."""
import json
import time
import joblib
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from .data import ROOT, load
from .metrics import score, evaluate, save_report
from .preprocessing import corrupt


def train(root=ROOT):
    start = time.perf_counter()
    df = load("complaint",root)
    tr,va = df[df.split == "train"],df[df.split == "validation"]
    baseline = joblib.load(root/"models/complaint_baseline.joblib")["model"]
    features = baseline.named_steps["features"]
    print("Transforming the unchanged training and validation splits",flush=True)
    x = features.transform(tr.text)
    v = features.transform(va.text)
    noisy = [corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(va.text)]
    vn = features.transform(noisy)
    labels = sorted(tr.label.unique())
    previous = json.loads((root/"reports/complaint_baseline.json").read_text())
    previous = previous[previous["selected"]]
    reference_score = (previous["validation"]["macro_f1"]+previous["validation_typos"]["macro_f1"])/2
    candidates = {
        "complement_nb":ComplementNB(alpha=.5),
        "l2_balanced":OneVsRestClassifier(LogisticRegression(solver="liblinear",C=1,class_weight="balanced",max_iter=200,tol=.001,random_state=42),n_jobs=1),
        "l2_unweighted":OneVsRestClassifier(LogisticRegression(solver="liblinear",C=1,max_iter=200,tol=.001,random_state=42),n_jobs=1),
    }
    selected,best,selected_name = None,-1,None
    search = {}
    for name,classifier in candidates.items():
        t = time.perf_counter()
        print("Fitting",name,flush=True)
        classifier.fit(x,tr.label)
        clean = score(va.label,classifier.predict(v),labels)
        typos = score(va.label,classifier.predict(vn),labels)
        selection = (clean["macro_f1"]+typos["macro_f1"])/2
        search[name] = {"validation":clean,"validation_typos":typos,"selection_score":selection,"seconds":time.perf_counter()-t}
        print(name,"selection score",selection,flush=True)
        if selection > best:
            selected,best,selected_name = classifier,selection,name
    # Fixed minimum practical improvement prevents promoting a tiny validation win.
    accepted = best >= reference_score+.02
    pipeline = Pipeline([("features",features),("classifier",selected)])
    result = {"search":search,"selected":selected_name,"reference_selection_score":reference_score,
              "selection_score":best,"adopt":bool(accepted),"adoption_rule":"Clean/noisy validation mean macro-F1 gain >=0.02",
              "protocol":"Three fixed regularized heads; existing training-only TF-IDF features; no test access during selection",
              "validation":search[selected_name]["validation"],"validation_typos":search[selected_name]["validation_typos"]}
    joblib.dump({"model":pipeline,"mode":selected_name,"task":"complaint"},root/"models/complaint_regularized.joblib",compress=3)
    if accepted:
        print("Validation criterion passed; evaluating the selected candidate on the supplied test once",flush=True)
        result.update(evaluate(pipeline.predict,df,sorted(df.label.unique())))
    else:
        result["test_policy"] = "Not evaluated: no material validation improvement. Existing baseline retained."
    result["seconds"] = time.perf_counter()-start
    result["limits"] = "Small fixed search, not proof of no overfitting. Validation reused in earlier development. Official test is an existing benchmark, not a new independent collection. Labels and rare classes unchanged."
    save_report(root/"reports/complaint_regularized.json",result)
    print("Selected",selected_name,"gain",best-reference_score,"adopt",accepted,flush=True)


if __name__ == "__main__":
    train()
