"""Predeclared three-part training-internal complaint improvement experiment."""
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from .baseline import make_model
from .data import ROOT, load
from .metrics import score, save_report
from .nb_features import NBScaledLinear
from .preprocessing import corrupt
from .uncertainty import paired_cluster_interval


CANDIDATES = ("reference", "sqrt_weights", "sqrt_less_regularized", "nb_ratio_svm")


def head(name,y):
    if name == "nb_ratio_svm":
        return NBScaledLinear(c=1.,workers=4)
    labels,counts = np.unique(y,return_counts=True)
    power = 1. if name == "reference" else .5
    weights = (len(y)/(len(labels)*counts))**power
    weights /= np.average(weights,weights=counts)
    return SGDClassifier(loss="log_loss",alpha=3e-6 if name == "sqrt_less_regularized" else 1e-5,
                         max_iter=80,tol=1e-4,n_iter_no_change=5,class_weight=dict(zip(labels,weights)),
                         average=True,random_state=42,n_jobs=4)


def feature_map(name):
    if name == "nb_ratio_svm":
        return CountVectorizer(binary=True,ngram_range=(1,2),min_df=2,max_features=100000,dtype=np.float64)
    return make_model("hybrid").named_steps["features"]


def noisy_text(part):
    return [corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(part.text)]


def assess(model,part,labels):
    clean = score(part.label,model.predict(part.text.tolist()),labels)
    noisy = score(part.label,model.predict(noisy_text(part)),labels)
    return {"clean":clean,"typos":noisy,"mean":(clean["macro_f1"]+noisy["macro_f1"])/2}


def assessment_from_matrices(model,x,clean,noisy,labels,y):
    model.fit(x,y)
    a = score(clean[0],model.predict(clean[1]),labels)
    b = score(noisy[0],model.predict(noisy[1]),labels)
    return {"clean":a,"typos":b,"mean":(a["macro_f1"]+b["macro_f1"])/2,
            "fit_warnings":getattr(model,"fit_warnings_",{})}


def fit_pipeline(name,part):
    print("Fitting",name,"on",len(part),"training messages",flush=True)
    features = feature_map(name)
    x = features.fit_transform(part.text)
    model = head(name,part.label).fit(x,part.label)
    return Pipeline([("features",features),("classifier",model)])


def validation_rule(candidate,reference,gain=.01):
    return bool(candidate["mean"] >= reference["mean"]+gain
                and candidate["clean"]["macro_f1"] >= reference["clean"]["macro_f1"]
                and candidate["clean"]["accuracy"] >= reference["clean"]["accuracy"]-.005
                and candidate["typos"]["macro_f1"] >= reference["typos"]["macro_f1"]-.005)


@threadpool_limits.wrap(limits=1)
def train(root=ROOT):
    start = time.perf_counter()
    df = load("complaint",root)
    tr = df[df.split == "train"].reset_index(drop=True)
    labels = sorted(tr.label.unique())
    if tr.group.duplicated().any():
        raise RuntimeError("Training-internal splits require unique prepared normalized groups")
    fit,rest = train_test_split(tr,test_size=.3,stratify=tr.label,random_state=20261021)
    selection,confirmation = train_test_split(rest,test_size=.5,stratify=rest.label,random_state=20261022)
    parts = {"fit":fit,"selection":selection,"confirmation":confirmation}
    signatures = {}
    for name,part in parts.items():
        if set(part.label) != set(labels):
            raise RuntimeError("A training-internal partition is missing a class")
        signatures[name] = {"rows":len(part),"group_digest":hashlib.sha256("".join(sorted(part.group)).encode()).hexdigest()}
    if sum(len(set(part.group)) for part in parts.values()) != len(set(tr.group)):
        raise RuntimeError("Training-internal groups overlap")
    plan = {"declared_utc":datetime.now(timezone.utc).isoformat(),"candidates":list(CANDIDATES),
            "split_seeds":[20261021,20261022],"partitions":signatures,
            "selection":"Maximum clean/noisy mean macro-F1, clean accuracy no worse than reference by 0.005; minimum mean gain 0.005",
            "confirmation":"One selected candidate refit on fit+selection; mean macro-F1 gain >=0.01, nonnegative clean F1 gain, accuracy and typo F1 drops <=0.005",
            "validation":"Selected candidate refit on full training; same fixed confirmation criteria against saved baseline on existing validation",
            "test_release":"Only after validation passes: one benchmark evaluation; positive clean F1 gain, accuracy/typo F1 drops <=0.005 and paired normalized-template bootstrap lower bound >0",
            "limits":"Repeatedly explored development corpus; split fingerprints control normalized equality, not all paraphrases; existing test is not fresh independent evidence",
            "source":"https://aclanthology.org/P12-2018/",
            "nb_variant":"Balanced one-versus-rest SVM on binary word/bigram NB ratios; C=1; no NB/SVM interpolation",
            "class_weights":"Balanced exponent 1 versus normalized square-root exponent 0.5; reference alpha 1e-5 versus fixed 3e-6 alternative"}
    save_report(root/"reports/complaint_improved_plan.json",plan)
    notes = root/"local_notes/improved_experiment.md"
    notes.parent.mkdir(parents=True,exist_ok=True)
    notes.write_text("# Complaint improvement experiment\n\nI am testing class weighting and NB feature scaling because prior comparisons changed model families without isolating those mechanisms. The candidate set and acceptance criteria are fixed before evaluation. All fitted features and count ratios stay within the relevant training partition.\n\n"+json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    result = {"plan":plan,"adopt":False,"status":"running","search":{}}

    def finish(status):
        result["status"] = status
        result["seconds"] = time.perf_counter()-start
        save_report(root/"reports/complaint_improved.json",result)
        with notes.open("a",encoding="utf-8") as stream:
            stream.write(f"\nOutcome: {status}; adopted: {result['adopt']}.\n")
        print("Outcome",status,"adopt",result["adopt"],flush=True)

    print("Fitting selection-stage hybrid features",flush=True)
    features = feature_map("reference")
    x = features.fit_transform(fit.text)
    v,vn = features.transform(selection.text),features.transform(noisy_text(selection))
    for name in CANDIDATES[:3]:
        tick = time.perf_counter()
        print("Selecting",name,flush=True)
        result["search"][name] = assessment_from_matrices(head(name,fit.label),x,(selection.label,v),(selection.label,vn),labels,fit.label)
        result["search"][name]["seconds"] = time.perf_counter()-tick
        print(name,"inner mean",result["search"][name]["mean"],"accuracy",result["search"][name]["clean"]["accuracy"],flush=True)
        save_report(root/"reports/complaint_improved_progress.json",result)
    del x,v,vn,features
    gc.collect()
    tick = time.perf_counter()
    model = fit_pipeline("nb_ratio_svm",fit)
    result["search"]["nb_ratio_svm"] = assess(model,selection,labels)
    result["search"]["nb_ratio_svm"]["fit_warnings"] = model.named_steps["classifier"].fit_warnings_
    result["search"]["nb_ratio_svm"]["seconds"] = time.perf_counter()-tick
    del model
    gc.collect()
    reference = result["search"]["reference"]
    eligible = [name for name in CANDIDATES if result["search"][name]["clean"]["accuracy"] >= reference["clean"]["accuracy"]-.005]
    selected = max(eligible,key=lambda name:result["search"][name]["mean"])
    result["selected"] = selected
    print("Selected",selected,"mean",result["search"][selected]["mean"],flush=True)
    if selected == "reference" or result["search"][selected]["mean"] < reference["mean"]+.005:
        finish("selection_gain_not_confirmed")
        return
    combined = tr.loc[tr.index.isin(set(fit.index)|set(selection.index))]
    candidate = fit_pipeline(selected,combined)
    reference_model = fit_pipeline("reference",combined)
    new,old = assess(candidate,confirmation,labels),assess(reference_model,confirmation,labels)
    result["confirmation"] = {"candidate":new,"reference":old,"passed":validation_rule(new,old)}
    print("Confirmation means",new["mean"],old["mean"],"passed",result["confirmation"]["passed"],flush=True)
    del candidate,reference_model
    gc.collect()
    if not result["confirmation"]["passed"]:
        finish("confirmation_gate_failed")
        return
    model = fit_pipeline(selected,tr)
    va = df[df.split == "validation"]
    new = assess(model,va,labels)
    baseline = json.loads((root/"reports/complaint_baseline.json").read_text())
    baseline = baseline[baseline["selected"]]
    old = {"clean":baseline["validation"],"typos":baseline["validation_typos"],
           "mean":(baseline["validation"]["macro_f1"]+baseline["validation_typos"]["macro_f1"])/2}
    result.update({"validation":new["clean"],"validation_typos":new["typos"],"selection_score":new["mean"],
                   "reference_selection_score":old["mean"],"validation_passed":validation_rule(new,old),
                   "adoption_rule":plan["test_release"]})
    joblib.dump({"model":model,"mode":selected,"task":"complaint"},root/"models/complaint_improved.joblib",compress=3)
    print("Validation means",new["mean"],old["mean"],"passed",result["validation_passed"],flush=True)
    if not result["validation_passed"]:
        result["test_policy"] = "Not evaluated: training-internal candidate did not pass existing-validation criteria."
        finish("validation_gate_failed")
        return
    te = df[df.split == "test"]
    test_labels = sorted(set(labels)|set(te.label))
    predicted = model.predict(te.text.tolist())
    result["test"] = score(te.label,predicted,test_labels)
    result["test_typos"] = score(te.label,model.predict(noisy_text(te)),test_labels)
    unique = te.drop_duplicates(["group","label"])
    result["test_unique_examples"] = score(unique.label,model.predict(unique.text.tolist()),test_labels)
    reference_model = joblib.load(root/"models/complaint_baseline.joblib")["model"]
    reference_predicted = reference_model.predict(te.text.tolist())
    interval = paired_cluster_interval(te.label,predicted,reference_predicted,te.group,test_labels)
    result["test_uncertainty"] = interval
    result["adopt"] = bool(result["test"]["macro_f1"] > baseline["test"]["macro_f1"]
                           and result["test"]["accuracy"] >= baseline["test"]["accuracy"]-.005
                           and result["test_typos"]["macro_f1"] >= baseline["test_typos"]["macro_f1"]-.005
                           and interval["lower_95"] > 0)
    finish("test_confirmed" if result["adopt"] else "test_confirmation_failed")


if __name__ == "__main__":
    train()
