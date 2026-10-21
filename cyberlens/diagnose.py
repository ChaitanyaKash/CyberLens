"""Read saved confusion matrices to inspect class failures without exposing source text."""
import json
import numpy as np
from .data import ROOT
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
    for name in ("word2vec","subword","transformer"):
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


if __name__ == "__main__":
    main()
