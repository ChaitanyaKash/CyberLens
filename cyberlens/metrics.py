import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .preprocessing import corrupt


def score(y, predicted, labels):
    return {"n": len(y), "accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, predicted, labels=labels, average="weighted", zero_division=0)),
            "per_class": classification_report(y, predicted, labels=labels, output_dict=True, zero_division=0),
            "confusion_labels": list(labels), "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist()}


def evaluate(predict, df, labels):
    results = {}
    for split in ("validation", "test"):
        part = df.loc[df.split == split]
        split_labels = sorted(set(df.loc[df.split == "train", "label"]) | set(part.label))
        results[split] = score(part.label, predict(part.text.tolist()), split_labels)
        noisy = [corrupt(t, seed=42+i, rate=.25) for i, t in enumerate(part.text)]
        results[f"{split}_typos"] = score(part.label, predict(noisy), split_labels)
        if split == "test" and "group" in part and part.group.duplicated().any():
            unique = part.drop_duplicates(["group", "label"])
            results["test_unique_examples"] = score(unique.label, predict(unique.text.tolist()), split_labels)
    return results


def save_report(path, data):
    content = json.dumps(data, indent=2, ensure_ascii=False)
    # Keep confusion-matrix rows readable without thousands of one-number lines.
    pattern = r"\[\n\s+(-?\d+(?:\.\d+)?(?:,\n\s+-?\d+(?:\.\d+)?)+)\n\s+\]"
    content = re.sub(pattern, lambda m:"["+re.sub(r",\n\s+", ", ", m[1])+"]", content)
    target = Path(path)
    temporary = target.with_suffix(target.suffix+".tmp")
    temporary.write_text(content+"\n", encoding="utf-8")
    temporary.replace(target)
