"""CPU baseline for scam vs not-scam SMS in English, Hinglish and Hindi.

Train on india14 (English/Hinglish/Hindi rows) plus bolewara English. Every split is grouped by
normalised text, so near-twins never land on both sides. Only validation picks C and the digit
handling; the four tests are scored once at the end.
Usage: python -m cyberlens.scam_binary
"""
import json
import re
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from .data import ROOT
from .metrics import save_report

RAW = ROOT / "data" / "raw" / "scam_binary"
LANGUAGES = ("English", "Hinglish", "Hindi")
C_GRID = (0.3, 1.0, 3.0, 10.0)
SEED = 42
# Common romanised-Hindi words; two or more in a Latin-script text marks it as Hinglish.
HINDI_WORDS = set("hai hain kya aap aapka aapki aapke apna apne karo kare karein karen nahi nahin nhi mein hoga "
                  "hogi gaya gayi raha rahe rahi jaldi turant abhi yahan wala wali paisa paise kijiye kripya "
                  "tumhara mera meri mujhe hamara bhej bhejo bhejein milega jayega jayegi liye diya karna hoon "
                  "kal aaj".split())


def normalise(text):
    return re.sub(r"\s+", " ", re.sub(r"\d", "", str(text).lower())).strip()


def keep_digits(text):
    return str(text).lower()


def mask_digits(text):
    # Phone numbers, amounts and OTPs all become the same shape, so the model can't memorise them.
    return re.sub(r"\d", "0", str(text).lower())


def is_hinglish(text):
    if re.search(r"[ऀ-ॿ]", text):
        return False
    return sum(word in HINDI_WORDS for word in re.findall(r"[a-z]+", text.lower())) >= 2


def load_india14():
    df = pd.read_csv(RAW / "india14" / "scam_ham_india_14_languages.csv")
    df = df[df.language.isin(LANGUAGES)].copy()
    df["label"] = (df.label == "scam").astype(int)
    df["source"] = "india14"
    return df[["text", "label", "language", "source"]]


def load_india14_suite():
    df = pd.read_csv(RAW / "india14" / "indic_test_suite_clean.csv")
    df = df[df.language.isin(LANGUAGES)].copy()
    df["label"] = (df.label == "scam").astype(int)
    df["source"] = "india14_suite"
    return df[["text", "label", "language", "source"]]


def load_bolewara():
    rows = json.loads((RAW / "bolewara" / "data.json").read_text(encoding="utf-8"))["rows"]
    df = pd.DataFrame(rows)
    df["text"] = df.text.astype(str)
    df["language"] = np.where(df.text.map(is_hinglish), "Hinglish", "English")
    df["source"] = "bolewara"
    return df[["text", "label", "language", "source"]]


def load_shade63():
    df = pd.read_csv(RAW / "shade63" / "sentinel_dataset_multiclass.csv")
    df["language"], df["source"] = "English", "shade63"
    legit = df[df.label == 0]
    scams = df[df.label == 1].sample(1000, random_state=SEED)
    return legit[["text", "label", "language", "source"]], scams[["text", "label", "language", "source"]]


def grouped_slice(df, n_splits):
    """One fold of a stratified split grouped by normalised text (about 1/n_splits of the rows)."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rest, held = next(splitter.split(df, df.label, df.text.map(normalise)))
    return df.iloc[rest], df.iloc[held]


def build_splits():
    india, bolewara = load_india14(), load_bolewara()
    test_a = load_india14_suite()
    test_c = bolewara[bolewara.language == "Hinglish"]
    english = bolewara[bolewara.language == "English"]
    english_rest, test_b = grouped_slice(english, 5)
    pool = pd.concat([india, english_rest], ignore_index=True)
    # No training row may be a near-twin of any test row.
    test_keys = set(pd.concat([test_a, test_b, test_c]).text.map(normalise))
    keys = pool.text.map(normalise)
    dropped = int(keys.isin(test_keys).sum())
    pool = pool[~keys.isin(test_keys)].reset_index(drop=True)
    train, validation = grouped_slice(pool, 7)
    shade_legit, shade_scams = load_shade63()
    splits = {"train": train, "validation": validation, "test_a_india14_suite": test_a,
              "test_b_bolewara_english": test_b, "test_c_bolewara_hinglish": test_c,
              "stress_d_shade63_legit": shade_legit, "stress_d_shade63_scams": shade_scams}
    return splits, dropped


def make_model(preprocessor, c):
    word = TfidfVectorizer(preprocessor=preprocessor, ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                           dtype=np.float32)
    char = TfidfVectorizer(preprocessor=preprocessor, analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           sublinear_tf=True, dtype=np.float32)
    return Pipeline([("features", FeatureUnion([("word", word), ("char", char)])),
                     ("classifier", LogisticRegression(C=c, class_weight="balanced", solver="liblinear",
                                                       max_iter=2000, random_state=SEED))])


def score(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp = matrix[0]
    result = {"n": int(len(y)), "n_scam": int(y.sum()), "n_ham": int((y == 0).sum()),
              "confusion_labels": ["ham", "scam"], "confusion_matrix": matrix.tolist()}
    if len(set(y)) == 2:
        result["macro_f1"] = float(f1_score(y, predicted, average="macro"))
    if y.sum():
        result["scam_precision"] = float(precision_score(y, predicted, zero_division=0))
        result["scam_recall"] = float(recall_score(y, predicted, zero_division=0))
    if (y == 0).sum():
        result["ham_false_positive_rate"] = float(fp / (tn + fp))
    return result


def examples(model, splits, n=10):
    rows = []
    for name, part in splits.items():
        if name in ("train", "validation"):
            continue
        predicted = model.predict(part.text.tolist())
        rows.append(part.assign(test=name, predicted=predicted))
    rows = pd.concat(rows)
    pick = lambda df: df.sample(min(n, len(df)), random_state=SEED)[["test", "language", "text"]].to_dict("records")
    result = {}
    # The shade63 stress rows would swamp the sample, so tests A-C and stress D are sampled separately.
    for group, part in (("tests_abc", rows[rows.test.str.startswith("test_")]),
                        ("stress_d", rows[rows.test.str.startswith("stress_")])):
        fp = part[(part.label == 0) & (part.predicted == 1)]
        fn = part[(part.label == 1) & (part.predicted == 0)]
        result[group] = {"false_positives": pick(fp), "false_negatives": pick(fn),
                         "false_positive_total": int(len(fp)), "false_negative_total": int(len(fn))}
    return result


def train(root=ROOT):
    splits, dropped = build_splits()
    tr, va = splits["train"], splits["validation"]
    print(f"train {len(tr)} validation {va.shape[0]} (dropped {dropped} near-twins of test rows)", flush=True)
    grid, best, best_score = [], None, -1
    for digits, preprocessor in (("keep", keep_digits), ("mask", mask_digits)):
        for c in C_GRID:
            start = time.perf_counter()
            model = make_model(preprocessor, c).fit(tr.text, tr.label)
            result = score(va.label, model.predict(va.text.tolist()))
            grid.append({"digits": digits, "C": c, "validation_macro_f1": result["macro_f1"],
                         "seconds": time.perf_counter()-start})
            print(digits, c, "validation", round(result["macro_f1"], 4), flush=True)
            # ONLY validation picks the configuration; ties go to the earlier (simpler) setting.
            if result["macro_f1"] > best_score:
                best, best_score = (digits, preprocessor, c), result["macro_f1"]
    digits, preprocessor, c = best
    model = make_model(preprocessor, c).fit(tr.text, tr.label)
    results = {"validation": score(va.label, model.predict(va.text.tolist()))}
    for name, part in splits.items():
        if name in ("train", "validation"):
            continue
        predicted = model.predict(part.text.tolist())
        results[name] = score(part.label, predicted)
        if name == "test_a_india14_suite":
            results[name]["per_language"] = {language: score(part.label[part.language == language],
                                                             predicted[(part.language == language).values])
                                             for language in LANGUAGES}
    report = {"selected": {"digits": digits, "C": c, "validation_macro_f1": best_score},
              "selection_rule": "Highest validation macro-F1 over digits x C; tests never used for selection",
              "grid": grid, "results": results, "examples": examples(model, splits),
              "data": {name: {"rows": int(len(part)), "scam": int(part.label.sum()),
                              "by_language": {k: int(v) for k, v in part.language.value_counts().items()},
                              "by_source": {k: int(v) for k, v in part.source.value_counts().items()}}
                       for name, part in splits.items()},
              "dropped_train_rows_matching_tests": dropped,
              "fitted_on": "train split only (validation kept out so its score stays honest)"}
    model_path = root / "models" / "scam_binary_baseline.joblib"
    joblib.dump({"model": model, "labels": ["ham", "scam"], "digits": digits, "C": c,
                 "selection_score": best_score}, model_path, compress=3)
    save_report(root / "reports" / "scam_binary_baseline.json", report)
    (root / "reports" / "scam_binary_baseline.md").write_text(markdown(report), encoding="utf-8")
    print("Saved", model_path, flush=True)
    return report


def pct(value):
    return "n/a" if value is None else f"{value:.3f}"


def markdown(report):
    r, s = report["results"], report["selected"]
    lines = ["# Scam vs not-scam baseline (TF-IDF + logistic regression)", "",
             "I trained a CPU-only baseline that says whether a message is a scam. The training data is the "
             "English, Hinglish and Hindi part of india14 plus the English part of bolewara. Every split is "
             "grouped by normalised text (lowercase, digits removed, spaces collapsed), so two messages that "
             "only differ in a phone number or amount never end up on both sides.", "",
             "## Data", "", "| split | rows | scam | languages | sources |", "|---|---|---|---|---|"]
    for name, d in report["data"].items():
        lines.append(f"| {name} | {d['rows']:,} | {d['scam']:,} | "
                     + ", ".join(f"{k} {v:,}" for k, v in d["by_language"].items()) + " | "
                     + ", ".join(f"{k} {v:,}" for k, v in d["by_source"].items()) + " |")
    lines += ["", f"I dropped {report['dropped_train_rows_matching_tests']:,} training rows whose normalised "
              "text matched a test row.", "",
              "## Model selection (validation only)", "",
              "Features are word 1-2 grams plus char_wb 2-5 grams, both TF-IDF, into a logistic regression with "
              "balanced class weights. I tried keeping digits and masking every digit as 0, with C in "
              f"{', '.join(str(c) for c in C_GRID)}.", "", "| digits | C | validation macro-F1 |", "|---|---|---|"]
    lines += [f"| {g['digits']} | {g['C']} | {g['validation_macro_f1']:.4f} |" for g in report["grid"]]
    lines += ["", f"Picked: digits={s['digits']}, C={s['C']} (validation macro-F1 {s['validation_macro_f1']:.4f}). "
              "The saved model is fitted on the train split only.", "",
              "## Results", "",
              "| set | n | macro-F1 | scam precision | scam recall | ham FPR | confusion [[TN, FP], [FN, TP]] |",
              "|---|---|---|---|---|---|---|"]
    for name, m in r.items():
        lines.append(f"| {name} | {m['n']:,} | {pct(m.get('macro_f1'))} | {pct(m.get('scam_precision'))} | "
                     f"{pct(m.get('scam_recall'))} | {pct(m.get('ham_false_positive_rate'))} | "
                     f"{m['confusion_matrix']} |")
    lines += ["", "Test A by language:", "",
              "| language | n | macro-F1 | scam precision | scam recall | ham FPR | confusion |", "|---|---|---|---|---|---|---|"]
    for language, m in r["test_a_india14_suite"]["per_language"].items():
        lines.append(f"| {language} | {m['n']:,} | {pct(m.get('macro_f1'))} | {pct(m.get('scam_precision'))} | "
                     f"{pct(m.get('scam_recall'))} | {pct(m.get('ham_false_positive_rate'))} | {m['confusion_matrix']} |")
    for group, label in (("tests_abc", "tests A-C"), ("stress_d", "stress D (shade63)")):
        ex = report["examples"][group]
        for kind, title in (("positive", "False positives (ham called scam)"),
                            ("negative", "False negatives (scam called ham)")):
            shown, total = ex[f"false_{kind}s"], ex[f"false_{kind}_total"]
            lines += ["", f"## {title}, {label}: {len(shown)} random of {total:,}", ""]
            lines += [f"- [{e['test']}, {e['language']}] {e['text']}".replace("\n", " ") for e in shown]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Import through the package so the pickled preprocessors point at cyberlens.scam_binary, not __main__.
    from cyberlens.scam_binary import train as package_train
    package_train()
