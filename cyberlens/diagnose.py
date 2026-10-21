"""Read saved confusion matrices to inspect class failures without exposing source text."""
import json
import numpy as np
from .data import ROOT, load
from .metrics import save_report


def parent_metrics(report):
    labels = report["confusion_labels"]
    source = np.array(report["confusion_matrix"])
    parents = [x.split(" :: ")[0] for x in labels]
    names = sorted(set(parents))
    matrix = np.zeros((len(names),len(names)))
    index = {name:i for i,name in enumerate(names)}
    for i, a in enumerate(parents):
        for j,b in enumerate(parents):
            matrix[index[a],index[b]] += source[i,j]
    tp = matrix.diagonal()
    f1 = np.divide(2*tp,matrix.sum(0)+matrix.sum(1),out=np.zeros_like(tp),where=(matrix.sum(0)+matrix.sum(1))>0)
    return {"accuracy":float(tp.sum()/matrix.sum()),"macro_f1":float(f1.mean()),
            "classes":{name:{"f1":float(f1[i]),"support":int(matrix[i].sum())} for i,name in enumerate(names)}}


def main():
    base = json.loads((ROOT/"reports/complaint_baseline.json").read_text())
    models = {"baseline":base[base["selected"]]}
    for name in ("word2vec","subword","transformer","hierarchical"):
        file = ROOT/"reports"/f"complaint_{name}.json"
        if file.exists():
            models[name] = json.loads(file.read_text())
    diagnostic = {}
    for name,report in models.items():
        diagnostic[name] = {split:parent_metrics(report[split]) for split in ("validation","test")}
        print(name, "validation fine macro-F1", round(report["validation"]["macro_f1"],4),
              "test fine macro-F1",round(report["test"]["macro_f1"],4),
              "test accuracy",round(report["test"]["accuracy"],4),flush=True)
        print("Parent metrics:",diagnostic[name]["test"],flush=True)
    print("Baseline per-class validation F1:")
    for name,r in models["baseline"]["validation"]["per_class"].items():
        if " :: " in name:
            print(name, round(r["f1-score"],3),"support",r["support"])
    save_report(ROOT/"reports/parent_diagnostic.json",diagnostic)
    df = load("complaint")
    tr,te = df[df.split == "train"],df[df.split == "test"]
    counts = tr.label.value_counts()
    report = models["baseline"]["test"]
    classes = []
    for label in report["confusion_labels"]:
        row = report["per_class"][label]
        a,b = int((tr.label==label).sum()),int((te.label==label).sum())
        classes.append({"label":label,"train_rows":a,"test_rows":b,
                        "train_fraction":a/len(tr),"test_fraction":b/len(te),"test_f1":row["f1-score"]})
    matrix = np.array(report["confusion_matrix"])
    labels = report["confusion_labels"]
    errors = [{"actual":labels[i],"predicted":labels[j],"count":int(matrix[i,j])}
              for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i,j]]
    save_report(ROOT/"reports/complaint_error_audit.json",{
        "training_support_min":int(counts.min()),"training_support_max":int(counts.max()),
        "training_support_ratio":float(counts.max()/counts.min()),
        "unsupported_test_labels":sorted(set(te.label)-set(tr.label)),
        "test_conflicting_normalized_groups":int((te.groupby("group").label.nunique()>1).sum()),
        "baseline_test_metrics":{key:report[key] for key in ("macro_f1","weighted_f1","accuracy")},
        "classes":classes,"largest_confusions":sorted(errors,key=lambda row:row["count"],reverse=True)[:20],
        "limits":"Diagnostic of an already-examined benchmark; these aggregate observations do not establish causality or justify tuning on test."})


if __name__ == "__main__":
    main()
