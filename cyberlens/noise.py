"""Confident-learning filter for noisy complaint labels (training rows only).

Out-of-fold probabilities flag training rows whose given label disagrees with per-class
confidence thresholds (Northcutt et al., 2021). Flagged rows are dropped or down-weighted,
validation selects the treatment, and validation/test labels are never modified.
"""
import time
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .adjusted import PriorAdjustedClassifier, accuracy_interval, log_prior, test_metrics
from .baseline import make_model
from .data import ROOT, load
from .metrics import save_report, score
from .preprocessing import corrupt
from .uncertainty import paired_cluster_interval
from .adjusted import classifier as sgd

TREATMENTS = {"reference": None, "drop": 0.0, "half_weight": 0.5}
MIN_GAIN = 0.005


def fit_weighted(texts, labels, weights, alpha, tau):
    pipeline = make_model("hybrid")
    pipeline.set_params(classifier=sgd(alpha))
    keep = weights > 0
    pipeline.fit(texts[keep], labels[keep], classifier__sample_weight=weights[keep])
    return PriorAdjustedClassifier(pipeline, log_prior(labels[keep], pipeline.classes_), tau)


def flag_issues(texts, labels, alpha, folds=5, seed=42):
    """Return a boolean mask of rows whose label disagrees with out-of-fold confident predictions."""
    classes = np.array(sorted(set(labels)))
    probs = np.zeros((len(texts), len(classes)))
    for train_index, held_index in StratifiedKFold(folds, shuffle=True, random_state=seed).split(texts, labels):
        pipeline = make_model("hybrid")
        pipeline.set_params(classifier=sgd(alpha))
        pipeline.fit(texts.iloc[train_index], labels.iloc[train_index])
        columns = np.searchsorted(classes, pipeline.classes_)
        probs[np.ix_(held_index, columns)] = pipeline.predict_proba(texts.iloc[held_index])
    given = np.searchsorted(classes, labels.to_numpy())
    thresholds = np.array([probs[given == k, k].mean() for k in range(len(classes))])
    confident = np.where(probs >= thresholds, probs, -1.0)
    has_confident = (confident > -1).any(axis=1)
    flagged = has_confident & (confident.argmax(axis=1) != given)
    return flagged, thresholds


def train(root=ROOT):
    start = time.perf_counter()
    df = load("complaint", root, protocol="official")
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    labels = sorted(set(tr.label))
    deployed = joblib.load(root/"models/complaint_adjusted.joblib")["model"]
    alpha, tau = deployed.pipeline.named_steps["classifier"].alpha, deployed.tau
    flagged, _ = flag_issues(tr.text, tr.label, alpha)
    print("flagged", int(flagged.sum()), "of", len(tr), round(time.perf_counter()-start), "s", flush=True)
    va_typos = [corrupt(t, seed=42+i, rate=.25) for i, t in enumerate(va.text)]
    search = {}
    for name, weight in TREATMENTS.items():
        weights = np.ones(len(tr)) if weight is None else np.where(flagged, weight, 1.0)
        model = fit_weighted(tr.text.reset_index(drop=True), tr.label.reset_index(drop=True), weights, alpha, tau)
        clean = score(va.label, model.predict(va.text.tolist()), labels)
        typo = score(va.label, model.predict(va_typos), labels)
        search[name] = {"clean_macro_f1": clean["macro_f1"], "typo_macro_f1": typo["macro_f1"],
                        "clean_accuracy": clean["accuracy"], "selection": (clean["macro_f1"]+typo["macro_f1"])/2}
        print(name, {k: round(v, 4) for k, v in search[name].items()}, flush=True)
    best = max(search, key=lambda k: search[k]["selection"])
    gain = search[best]["selection"]-search["reference"]["selection"]
    report = {"treatments": TREATMENTS, "search": search, "selected": best, "validation_gain": gain,
              "flagged_training_rows": int(flagged.sum()), "training_rows": len(tr),
              "flagged_by_label": tr.label[flagged].value_counts().to_dict(),
              "flagged_share_by_label": (tr.label[flagged].value_counts()/tr.label.value_counts()).fillna(0).to_dict(),
              "rule": f"Selection by mean clean/typo validation macro-F1; adoption requires gain >= {MIN_GAIN}",
              "limits": "Flags are model-based suspicions, not verified label errors; validation labels are themselves noisy."}
    if best != "reference" and gain >= MIN_GAIN:
        development = df[df.split != "test"].reset_index(drop=True)
        dev_flags, _ = flag_issues(development.text, development.label, alpha)
        weights = np.where(dev_flags, TREATMENTS[best], 1.0)
        model = fit_weighted(development.text, development.label, weights, alpha, tau)
        results = test_metrics(model.predict, df, labels)
        reference = test_metrics(deployed.predict, df, labels)
        test = df[df.split == "test"].reset_index(drop=True)
        new, old = model.predict(test.text.tolist()), deployed.predict(test.text.tolist())
        split_labels = sorted(set(labels) | set(test.label))
        intervals = {}
        for key, mask in (("test", np.ones(len(test), bool)), ("test_novel_templates", ~test.seen_template.astype(bool).to_numpy())):
            y, groups = test.label.to_numpy()[mask], test.group.to_numpy()[mask]
            intervals[key] = paired_cluster_interval(y, new[mask], old[mask], groups, split_labels)
            intervals[key].update(accuracy_interval(y, new[mask], old[mask], groups))
        report.update({"development_flagged_rows": int(dev_flags.sum()), "test_results": results,
                       "adjusted_results": reference, "uncertainty_vs_adjusted": intervals})
        report["adopt"] = bool(results["test"]["macro_f1"] >= reference["test"]["macro_f1"]
                               and results["test"]["accuracy"] >= reference["test"]["accuracy"]
                               and results["test_novel_templates"]["macro_f1"] >= reference["test_novel_templates"]["macro_f1"]-.005)
        for key in ("test", "test_typos", "test_novel_templates"):
            print(key, "cleaned", round(results[key]["macro_f1"], 4), round(results[key]["accuracy"], 4),
                  "adjusted", round(reference[key]["macro_f1"], 4), round(reference[key]["accuracy"], 4), flush=True)
        if report["adopt"]:
            joblib.dump({"model": model, "task": "complaint", "mode": f"adjusted-hybrid-cl-{best}", "labels": labels,
                         "selection_score": search[best]["selection"]}, root/"models/complaint_cleaned.joblib", compress=3)
    else:
        report["adopt"] = False
        report["test_policy"] = "Not evaluated: validation gain below the predeclared minimum."
    print("adopt", report["adopt"], flush=True)
    report["seconds"] = time.perf_counter()-start
    save_report(root/"reports/complaint_cleaned.json", report)
    return report


if __name__ == "__main__":
    from cyberlens.noise import train as canonical_train
    canonical_train()
