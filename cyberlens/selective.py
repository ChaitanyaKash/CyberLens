"""Selective complaint prediction: answer only when the model is confident, route the rest to review.

The confidence threshold is chosen on validation (training-only fit) as the widest coverage whose
mean clean/typo answered macro-F1 reaches TARGET, then applied once to the official test.
Answered scores describe the answered share only; the full-taxonomy score is reported beside them.
"""
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from .adjusted import fit
from .data import ROOT, load
from .metrics import save_report
from .preprocessing import corrupt

TARGET = 0.50
COVERAGES = tuple(round(c, 2) for c in np.arange(.05, 1.0001, .05))


def answered(y, predicted, confidence, threshold):
    keep = confidence >= threshold
    if not keep.any():
        return {"coverage": 0.0, "n": 0, "macro_f1": 0.0, "accuracy": 0.0, "weighted_f1": 0.0}
    y, p = np.asarray(y)[keep], np.asarray(predicted)[keep]
    labels = sorted(set(y) | set(p))
    return {"coverage": float(keep.mean()), "n": int(keep.sum()),
            "macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, p)),
            "weighted_f1": float(f1_score(y, p, labels=labels, average="weighted", zero_division=0))}


def outputs(model, texts):
    probs = model.predict_proba(list(texts))
    return model.classes_[probs.argmax(1)], probs.max(1)


def choose(curve):
    passing = [row for row in curve if row["selection"] >= TARGET]
    if passing:
        return max(passing, key=lambda r: r["target_coverage"]), True
    return max(curve, key=lambda r: r["selection"]*r["clean"]["coverage"]), False


def train(root=ROOT, model_path="models/complaint_adjusted.joblib"):
    df = load("complaint", root, protocol="official")
    tr, va, te = (df[df.split == s] for s in ("train", "validation", "test"))
    deployed = joblib.load(root/model_path)["model"]
    # Validation needs a model that has not seen validation rows: same configuration, training split only.
    probe = fit(tr.text, tr.label, deployed.pipeline.named_steps["classifier"].alpha, deployed.tau)
    clean = outputs(probe, va.text)
    typo = outputs(probe, [corrupt(t, seed=42+i, rate=.25) for i, t in enumerate(va.text)])
    curve = []
    for coverage in COVERAGES:
        threshold = float(np.quantile(clean[1], 1-coverage)) if coverage < 1 else 0.0
        c, t = answered(va.label, *clean, threshold), answered(va.label, *typo, threshold)
        curve.append({"target_coverage": coverage, "threshold": threshold, "clean": c, "typo": t,
                      "selection": (c["macro_f1"]+t["macro_f1"])/2})
        print(coverage, round(threshold, 4), round(c["macro_f1"], 4), round(c["accuracy"], 4), flush=True)
    chosen, reached = choose(curve)
    threshold = chosen["threshold"]
    report = {"target_macro_f1": TARGET, "validation_curve": curve, "selected": chosen, "target_reached_on_validation": reached,
              "rule": "Largest validation coverage with mean clean/typo answered macro-F1 >= target; else max F1 x coverage",
              "macro_definition": "Macro-F1 over labels present among answered rows' true or predicted labels"}
    novel = te[~te.seen_template.astype(bool)]
    for name, part, noisy in (("test", te, False), ("test_typos", te, True), ("test_novel_templates", novel, False)):
        texts = [corrupt(t, seed=42+i, rate=.25) for i, t in enumerate(part.text)] if noisy else part.text
        predicted, confidence = outputs(deployed, texts)
        report[name] = answered(part.label, predicted, confidence, threshold)
        keep = confidence >= threshold
        report[name]["full_taxonomy"] = answered(part.label, predicted, confidence, -1.0)
        if name == "test":
            labels = sorted(set(part.label))
            report["test_per_class_answered_share"] = {
                label: float(keep[part.label.to_numpy() == label].mean()) for label in labels}
        print(name, {k: report[name][k] for k in ("coverage", "macro_f1", "accuracy")}, flush=True)
    save_report(root/"reports/complaint_selective.json", report)
    return report


if __name__ == "__main__":
    train()
