"""Recompute metrics from saved confusion counts; no fitting or test-based selection.

Initial validation reports included zero-support test-only labels in their macro
denominator. Predictions are unchanged; validation should use the training label
vocabulary. Test reports retain unsupported labels and their zero recall.
"""
import json
import numpy as np
from .data import ROOT, TASKS, load
from .metrics import score, save_report


def correct(result, supported):
    for split in ("validation", "validation_typos"):
        original = result[split]
        labels = np.array(original["confusion_labels"])
        matrix = np.array(original["confusion_matrix"])
        rows, cols = np.nonzero(matrix)
        counts = matrix[rows,cols]
        y = np.repeat(labels[rows],counts)
        p = np.repeat(labels[cols],counts)
        if not set(p).issubset(supported) or not set(y).issubset(supported):
            raise RuntimeError("Validation predictions/targets contain unsupported training labels")
        result[split] = score(y,p,sorted(supported))


def main():
    for task in TASKS:
        df = load(task)
        supported = set(df.loc[df.split == "train","label"])
        for mode in ("baseline","word2vec","subword","transformer"):
            path = ROOT/"reports"/f"{task}_{mode}.json"
            if not path.exists():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            if mode == "baseline":
                for variant in ("word","char","hybrid"):
                    correct(report[variant],supported)
            else:
                correct(report,supported)
            report["validation_label_policy"] = "Training label vocabulary only; unsupported official-test labels remain in test metrics"
            save_report(path,report)
        print("Corrected validation metric vocabulary:",task,len(supported),flush=True)


if __name__ == "__main__":
    main()
