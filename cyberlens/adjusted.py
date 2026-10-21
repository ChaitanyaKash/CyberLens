"""Complaint backend trained under the official-train protocol with a post-hoc class-prior correction.

Balanced class weights make rare subtypes cheap to predict, so their precision collapses.
The correction adds tau * log(training prior) to log-probabilities; tau and the SGD alpha are
chosen on validation only from a fixed grid, then the chosen setting is refitted on
training + validation. See local protocol notes for the predeclared plan.
"""
import time
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from .baseline import make_model
from .data import ROOT, load
from .metrics import save_report, score
from .preprocessing import corrupt
from .uncertainty import paired_cluster_interval

ALPHAS = (2e-6, 5e-6, 1e-5, 2e-5)
TAUS = (0.0, 0.25, 0.5, 0.75, 1.0)
REFERENCE = (1e-5, 0.0)
# Predeclared: full-test F1 and accuracy may not fall below the current default, and F1 on test
# templates absent from train.csv may not fall by more than this (guards against memorisation-only gains).
NOVEL_TOLERANCE = 0.005


class PriorAdjustedClassifier:
    """Pipeline wrapper: softmax(log p(y|x) + tau * log prior(y))."""

    def __init__(self, pipeline, log_prior, tau):
        self.pipeline, self.log_prior, self.tau = pipeline, np.asarray(log_prior), float(tau)
        self.classes_ = pipeline.classes_

    @property
    def named_steps(self):
        return self.pipeline.named_steps

    def adjust(self, probs):
        scores = np.log(np.clip(probs, 1e-12, None)) + self.tau*self.log_prior
        scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return scores / scores.sum(axis=1, keepdims=True)

    def predict_proba_features(self, features):
        return self.adjust(self.pipeline.named_steps["classifier"].predict_proba(features))

    def predict_proba(self, texts):
        return self.adjust(self.pipeline.predict_proba(texts))

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def classifier(alpha):
    return SGDClassifier(loss="log_loss", alpha=alpha, max_iter=80, tol=1e-4, n_iter_no_change=5,
                         class_weight="balanced", average=True, random_state=42, n_jobs=4)


def log_prior(labels, classes):
    counts = labels.value_counts().reindex(classes).to_numpy(float)
    return np.log(counts / counts.sum())


def fit(texts, labels, alpha, tau):
    pipeline = make_model("hybrid")
    pipeline.set_params(classifier=classifier(alpha))
    pipeline.fit(texts, labels)
    return PriorAdjustedClassifier(pipeline, log_prior(labels, pipeline.classes_), tau)


def test_metrics(predict, df, labels):
    """Official test (all rows), synthetic typos, unique templates and the novel-template slice."""
    test = df[df.split == "test"]
    split_labels = sorted(set(labels) | set(test.label))
    unique = test.drop_duplicates(["group", "label"])
    novel = test[~test.seen_template.astype(bool)]
    noisy = [corrupt(t, seed=42+i, rate=.25) for i, t in enumerate(test.text)]
    return {"test": score(test.label, predict(test.text.tolist()), split_labels),
            "test_typos": score(test.label, predict(noisy), split_labels),
            "test_unique_examples": score(unique.label, predict(unique.text.tolist()), split_labels),
            "test_novel_templates": score(novel.label, predict(novel.text.tolist()), split_labels)}


def accuracy_interval(y, new, reference, groups, repetitions=500, seed=20261021):
    _, inverse = np.unique(groups, return_inverse=True)
    n = inverse.max()+1
    gain = np.bincount(inverse, weights=(new == y).astype(float)-(reference == y), minlength=n)
    size = np.bincount(inverse, minlength=n)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        weights = np.bincount(rng.integers(0, n, n), minlength=n)
        draws.append((weights @ gain)/(weights @ size))
    return {"accuracy_lower_95": float(np.quantile(draws, .025)), "accuracy_upper_95": float(np.quantile(draws, .975))}


def compare_with_default(model, df, labels, root=ROOT):
    """Score the saved default on the same test slices, apply the adoption rule, add paired intervals."""
    default = joblib.load(root/"models/complaint_baseline.joblib")["model"]
    slices = test_metrics(default.predict, df, labels)
    save_report(root/"reports/complaint_baseline_slices.json", slices)
    test = df[df.split == "test"].reset_index(drop=True)
    new, old = model.predict(test.text.tolist()), default.predict(test.text.tolist())
    split_labels = sorted(set(labels) | set(test.label))
    intervals = {}
    for name, mask in (("test", np.ones(len(test), bool)), ("test_novel_templates", ~test.seen_template.astype(bool).to_numpy())):
        y, groups = test.label.to_numpy()[mask], test.group.to_numpy()[mask]
        intervals[name] = paired_cluster_interval(y, new[mask], old[mask], groups, split_labels)
        intervals[name].update(accuracy_interval(y, new[mask], old[mask], groups))
    return {"default_slices": {k: {m: slices[k][m] for m in ("n", "accuracy", "macro_f1", "weighted_f1")} for k in slices},
            "default_slices_per_class": {k: v for k, v in slices["test"]["per_class"].items() if k in split_labels},
            "uncertainty_vs_default": intervals, "adopt": None, "adoption_rule":
            f"Full-test macro-F1 and accuracy >= current default; novel-template macro-F1 drop <= {NOVEL_TOLERANCE}"}


def decide(report):
    new, old = report["selected_refit"], report["default_slices"]
    return bool(new["test"]["macro_f1"] >= old["test"]["macro_f1"] and new["test"]["accuracy"] >= old["test"]["accuracy"]
                and new["test_novel_templates"]["macro_f1"] >= old["test_novel_templates"]["macro_f1"]-NOVEL_TOLERANCE)


def train(root=ROOT):
    start = time.perf_counter()
    df = load("complaint", root, protocol="official")
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    labels = sorted(set(tr.label))
    features = make_model("hybrid").named_steps["features"]
    x_train = features.fit_transform(tr.text)
    x_val = features.transform(va.text)
    x_val_typo = features.transform([corrupt(t, seed=42+i, rate=.25) for i, t in enumerate(va.text)])
    print("features", x_train.shape, round(time.perf_counter()-start), "s", flush=True)
    grid = []
    for alpha in ALPHAS:
        clf = classifier(alpha).fit(x_train, tr.label)
        probs = clf.predict_proba(x_val), clf.predict_proba(x_val_typo)
        prior = log_prior(tr.label, clf.classes_)
        for tau in TAUS:
            wrapper = PriorAdjustedClassifier(Pipeline([("classifier", clf)]), prior, tau)
            clean, typo = (score(va.label, clf.classes_[wrapper.adjust(p).argmax(1)], labels) for p in probs)
            row = {"alpha": alpha, "tau": tau, "clean_macro_f1": clean["macro_f1"], "typo_macro_f1": typo["macro_f1"],
                   "clean_accuracy": clean["accuracy"], "typo_accuracy": typo["accuracy"],
                   "selection": (clean["macro_f1"]+typo["macro_f1"])/2}
            grid.append(row)
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    best = max(grid, key=lambda r: r["selection"])
    reference = next(r for r in grid if (r["alpha"], r["tau"]) == REFERENCE)
    print("selected", best, flush=True)
    # Per-class validation evidence for inference review flags, from the training-only fit.
    model = fit(tr.text, tr.label, best["alpha"], best["tau"])
    validation = score(va.label, model.predict(va.text.tolist()), labels)
    development = df[df.split != "test"]
    report = {"protocol": "official-train: test.csv not consulted for training; majority labels for "
                          "conflicting templates with >=60% agreement",
              "grid": grid, "selected": {"alpha": best["alpha"], "tau": best["tau"]},
              "selection_rule": "Max mean clean + synthetic-typo validation macro-F1 over the fixed grid",
              "reference_validation": reference, "selected_validation": best, "validation": validation}
    for name, (alpha, tau) in (("reference", REFERENCE), ("selected", (best["alpha"], best["tau"]))):
        t = time.perf_counter()
        # Refit on train+validation: validation numbers above come from the training-only fit.
        model = fit(development.text, development.label, alpha, tau)
        results = test_metrics(model.predict, df, labels)
        results["seconds"] = time.perf_counter()-t
        results["config"] = {"alpha": alpha, "tau": tau}
        report[f"{name}_refit"] = results
        print(name, {k: round(results[k]["macro_f1"], 4) for k in ("test", "test_typos", "test_novel_templates")},
              "acc", round(results["test"]["accuracy"], 4), flush=True)
        if name == "selected":
            joblib.dump({"model": model, "task": "complaint", "mode": "adjusted-hybrid", "labels": labels,
                         "selection_score": best["selection"]}, root/"models/complaint_adjusted.joblib", compress=3)
    report.update(compare_with_default(model, df, labels, root))
    report["adopt"] = decide(report)
    report["seconds"] = time.perf_counter()-start
    save_report(root/"reports/complaint_adjusted.json", report)
    return report


if __name__ == "__main__":
    from cyberlens.adjusted import train as canonical_train
    canonical_train()
