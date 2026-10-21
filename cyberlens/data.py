"""Two label vocabularies; official complaint test remains held out."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocessing import normalize, fingerprint

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("call_scam", "complaint")
AMBIGUOUS_CALL_LABELS = {"aadhaar", "amazon", "bank_kyc", "lottery", "relative"}


def frame(raw, task):
    text_col = "text" if task == "call_scam" else "crimeaditionalinfo"
    labels = raw.scam_category.astype(str).str.strip() if task == "call_scam" else (
        raw.category.str.strip() + " :: " + raw.sub_category.fillna("Unspecified").str.strip())
    df = pd.DataFrame({"text": raw[text_col], "label": labels})
    df = df.loc[df.text.notna()].copy()
    unique = {t: normalize(t) for t in df.text.unique()}
    df["text"] = df.text.map(unique)
    df = df.loc[df.text.str.len() >= 5].copy()
    groups = {t: fingerprint(t, already_normalized=True) for t in df.text.unique()}
    df["group"] = df.text.map(groups)
    return df


def prepare(root=ROOT):
    root = Path(root)
    for directory in ("models", "reports", "data/processed"):
        (root/directory).mkdir(parents=True, exist_ok=True)
    out = root / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    sources = {"call_scam": "India_Cyber_Scam_Hinglish_Dataset.csv", "complaint": "train.csv"}
    for task, filename in sources.items():
        path = root / "data/raw" / filename
        raw = pd.read_csv(path)
        df = frame(raw, task)
        info = {"file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "raw_rows": len(raw), "empty_or_short_rows_excluded": len(raw)-len(df)}
        if task == "call_scam":
            original_conflicts = df.groupby("group").label.nunique()
            conflict_ids = set(original_conflicts[original_conflicts > 1].index)
            evidence = df[df.group.isin(conflict_ids)].groupby("group").agg(
                rows=("label", "size"), labels=("label", lambda s: ", ".join(sorted(set(s)))))
            evidence.to_csv(root / "reports/call_label_conflicts.csv")
            info.update({"original_conflicting_groups": len(conflict_ids),
                         "label_mapping": {x: "other_scam" for x in sorted(AMBIGUOUS_CALL_LABELS)},
                         "original_unique_texts": int(raw.text.nunique())})
            df["label"] = df.label.where(~df.label.isin(AMBIGUOUS_CALL_LABELS), "other_scam")
            official_test = None
        else:
            test_path = root / "data/raw/test.csv"
            raw_test = pd.read_csv(test_path)
            official_test = frame(raw_test, task)
            official_test["split"] = "test"
            overlap = df.group.isin(set(official_test.group))
            info.update({"official_test_file": "test.csv", "official_test_raw_rows": len(raw_test),
                         "official_test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
                         "official_test_valid_rows": len(official_test),
                         "train_rows_removed_due_to_test_overlap": int(overlap.sum()),
                         "overlapping_templates": int(df.loc[overlap, "group"].nunique()),
                         "test_repeated_templates": int(official_test.group.duplicated().sum()),
                         "test_conflicting_templates": int((official_test.groupby("group").label.nunique()>1).sum())})
            df = df.loc[~overlap].copy()
        contradictory = df.groupby("group").label.nunique()
        bad_groups = set(contradictory[contradictory > 1].index)
        info["conflicting_rows_excluded"] = int(df.group.isin(bad_groups).sum())
        info["conflicting_groups"] = len(bad_groups)
        df = df.loc[~df.group.isin(bad_groups)].copy()
        before = len(df)
        df = df.drop_duplicates("group").reset_index(drop=True)
        info["duplicate_rows_removed"] = before-len(df)
        counts = df.label.value_counts()
        rare = counts[counts < 10].to_dict()
        df.loc[df.label.isin(rare)].to_csv(out / f"{task}_rare_excluded.csv", index=False)
        df = df.loc[~df.label.isin(rare)].reset_index(drop=True)
        if official_test is None:
            tr, rest = train_test_split(df, test_size=.4, stratify=df.label, random_state=42)
            va, te = train_test_split(rest, test_size=.5, stratify=rest.label, random_state=42)
            tr, va, te = tr.copy(), va.copy(), te.copy()
            tr["split"], va["split"], te["split"] = "train", "validation", "test"
        else:
            tr, va = train_test_split(df, test_size=.2, stratify=df.label, random_state=42)
            tr, va = tr.copy(), va.copy()
            tr["split"], va["split"] = "train", "validation"
            te = official_test
        combined = pd.concat([tr, va, te], ignore_index=True)
        if combined.groupby("group").split.nunique().max() > 1:
            raise RuntimeError("A template leaked between splits")
        if set(tr.label) != set(va.label):
            raise RuntimeError("Validation label coverage differs from training")
        combined.to_csv(out / f"{task}.csv", index=False)
        info.update({"rare_labels_excluded": rare, "rows": len(combined),
                     "labels": int(tr.label.nunique()), "splits": combined.split.value_counts().to_dict(),
                     "training_label_counts": tr.label.value_counts().to_dict(),
                     "unsupported_test_labels": sorted(set(te.label)-set(tr.label)),
                     "near_duplicate_limit": "Normalized template equality only; paraphrases may cross splits.",
                     "test_policy": "Official test unchanged except missing/short texts; duplicates and label conflicts retained"
                     if task == "complaint" else "60/20/20 deduplicated stratified internal split"})
        summary[task] = info
        print(task, info["splits"], info["labels"], "trainable labels", flush=True)
    (root / "reports/data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load(task, root=ROOT):
    if task not in TASKS:
        raise ValueError(task)
    return pd.read_csv(Path(root) / "data/processed" / f"{task}.csv")
