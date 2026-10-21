"""CPU baselines: sparse word/character n-grams, no spelling dictionary."""
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier
from .data import ROOT, TASKS, load
from .metrics import evaluate, save_report


def make_model(mode):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=65000, sublinear_tf=True, dtype=np.float32)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          max_features=100000, sublinear_tf=True, dtype=np.float32)
    features = {"word": word, "char": char, "hybrid": FeatureUnion([("word", word), ("char", char)])}[mode]
    return Pipeline([("features", features), ("classifier", SGDClassifier(
        loss="log_loss", alpha=1e-5, max_iter=80, tol=1e-4, n_iter_no_change=5,
        class_weight="balanced", average=True, random_state=42, n_jobs=4))])


def train(tasks=TASKS, root=ROOT):
    for task in tasks:
        df = load(task, root)
        tr = df[df.split == "train"]
        labels = sorted(df.label.unique())
        all_results = {}
        best, best_score, best_name = None, -1, None
        for mode in ("word", "char", "hybrid"):
            start = time.perf_counter()
            model = make_model(mode)
            print(f"Training {task}/{mode}: {len(tr)} examples", flush=True)
            model.fit(tr.text, tr.label)
            results = evaluate(model.predict, df, labels)
            results["seconds"] = time.perf_counter()-start
            all_results[mode] = results
            # ONLY validation is used to select the deployed model.
            selection = (results["validation"]["macro_f1"] + results["validation_typos"]["macro_f1"]) / 2
            print(mode, "validation", round(results["validation"]["macro_f1"],4),
                  "test", round(results["test"]["macro_f1"],4), flush=True)
            if selection > best_score:
                best, best_score, best_name = model, selection, mode
        model_path = root / "models" / f"{task}_baseline.joblib"
        joblib.dump({"model": best, "task": task, "mode": best_name,
                     "labels": labels, "selection_score": best_score}, model_path, compress=3)
        all_results["selected"] = best_name
        all_results["selection_rule"] = "Mean clean + synthetic-typo validation macro-F1; test never used for selection"
        save_report(root / "reports" / f"{task}_baseline.json", all_results)
        print("Saved", model_path, flush=True)
