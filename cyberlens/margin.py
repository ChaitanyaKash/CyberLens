"""Select a bounded margin-classifier experiment within the training partition."""
import json
import time
import joblib
import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from .baseline import make_model
from .data import ROOT, load
from .metrics import score, evaluate, save_report
from .preprocessing import corrupt


class MarginHead(ClassifierMixin, BaseEstimator):
    """Softmax is an uncalibrated ranking conversion, not probability calibration."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y):
        self.estimator.fit(x,y)
        self.classes_ = self.estimator.classes_
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def predict_proba(self, x):
        margins = self.estimator.decision_function(x)
        if margins.ndim == 1:
            margins = np.column_stack((-margins,margins))
        return softmax(margins,axis=1)


def head(name):
    if name == "linear_svc":
        estimator = LinearSVC(C=1,class_weight="balanced",dual="auto",max_iter=2000,tol=.001,random_state=42)
    else:
        estimator = SGDClassifier(loss="hinge" if name == "sgd_hinge" else "modified_huber",
                                  alpha=1e-5,class_weight="balanced",max_iter=80,tol=1e-4,
                                  n_iter_no_change=5,average=True,random_state=42,n_jobs=4)
    return MarginHead(estimator)


@threadpool_limits.wrap(limits=1)
def train(root=ROOT):
    start = time.perf_counter()
    df = load("complaint",root)
    tr,va = df[df.split == "train"],df[df.split == "validation"]
    # The source training partition is unique by normalized group. No official
    # validation/test rows enter this inner selection or its fitted vocabulary.
    if tr.group.duplicated().any():
        raise RuntimeError("Inner split requires the prepared unique training groups")
    inner_train,inner_val = train_test_split(tr,test_size=.2,stratify=tr.label,random_state=2026)
    features = make_model("hybrid").named_steps["features"]
    print("Fitting inner-training-only hybrid features",flush=True)
    x = features.fit_transform(inner_train.text)
    v = features.transform(inner_val.text)
    vn = features.transform([corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(inner_val.text)])
    labels = sorted(tr.label.unique())
    search = {}
    for name in ("sgd_hinge","modified_huber","linear_svc"):
        tick = time.perf_counter()
        print("Fitting inner candidate",name,flush=True)
        model = head(name).fit(x,inner_train.label)
        clean = score(inner_val.label,model.predict(v),labels)
        noisy = score(inner_val.label,model.predict(vn),labels)
        value = (clean["macro_f1"]+noisy["macro_f1"])/2
        search[name] = {"inner_validation":clean,"inner_validation_typos":noisy,
                        "selection_score":value,"seconds":time.perf_counter()-tick}
        print(name,"inner selection",value,flush=True)
    selected = max(search,key=lambda name:search[name]["selection_score"])
    del x,v,vn,features
    # Reuse the existing full-training-only features for the one selected refit.
    features = joblib.load(root/"models/complaint_baseline.joblib")["model"].named_steps["features"]
    print("Refitting selected candidate on full original training split:",selected,flush=True)
    model = head(selected).fit(features.transform(tr.text),tr.label)
    pipeline = Pipeline([("features",features),("classifier",model)])
    clean = score(va.label,pipeline.predict(va.text),labels)
    noisy = score(va.label,pipeline.predict([corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(va.text)]),labels)
    selection = (clean["macro_f1"]+noisy["macro_f1"])/2
    base = json.loads((root/"reports/complaint_baseline.json").read_text())
    base = base[base["selected"]]
    reference = (base["validation"]["macro_f1"]+base["validation_typos"]["macro_f1"])/2
    accepted = selection >= reference+.02
    result = {"search":search,"selected":selected,"validation":clean,"validation_typos":noisy,
              "reference_selection_score":reference,"selection_score":selection,"adopt":bool(accepted),
              "protocol":"Three fixed heads selected on an inner split of original training; selected refit assessed once on original validation",
              "adoption_rule":"Original clean/noisy validation mean macro-F1 gain >=0.02",
              "inner_train_rows":len(inner_train),"inner_validation_rows":len(inner_val),
              "inner_overlapping_groups":len(set(inner_train.group)&set(inner_val.group)),
              "limits":"Original validation was used in previous development; this is not independent confirmation. Softmax margins are uncalibrated. Labels and official test unchanged."}
    joblib.dump({"model":pipeline,"mode":selected,"task":"complaint"},root/"models/complaint_margin.joblib",compress=3)
    if accepted:
        print("Validation rule passed; evaluating selected candidate on official test",flush=True)
        result.update(evaluate(pipeline.predict,df,sorted(df.label.unique())))
    else:
        result["test_policy"] = "Not evaluated: selected candidate did not pass the validation improvement rule."
    result["seconds"] = time.perf_counter()-start
    save_report(root/"reports/complaint_margin.json",result)
    print("Selected",selected,"validation mean",selection,"gain",selection-reference,"adopt",accepted,flush=True)


if __name__ == "__main__":
    from .margin import train as train_module
    train_module()
