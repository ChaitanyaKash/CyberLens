"""Scam vs not-scam v2: v1 data plus shade63, split by template so near-identical alerts never straddle splits.

Stress D2 is ~1/3 of shade63's template groups (stratified by category) and is never trained on.
Tests A, B and C are the same rows as v1. Only validation picks settings or epochs.
Usage: python -m cyberlens.scam_binary_v2 tfidf | encoder [--model xlm-roberta-base] | report
"""
import argparse
import json
import math
import random
import re
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .data import ROOT
from .metrics import save_report
from . import scam_binary as v1

ENCODERS = {"xlm-roberta-base": "xlmr", "l3cube-pune/hing-roberta": "hingroberta"}
CATEGORIES = ("legit", "government_impersonation", "financial_fraud", "bank_phishing", "romance_parcel_sextortion")
REPORTS = ROOT / "reports"


def template_key(text):
    # Mask amounts, anything with a digit (ids, phones, dates) and capitalised words (names, banks, brands, places).
    text = re.sub(r"(rs\.?|₹|inr)\s*[\d,.]+(\s*(lakh|lakhs|crore|crores|k))?", " AMT ", text, flags=re.I)
    text = re.sub(r"\b[\w.-]*\d[\w.-]*\b", " NUM ", text)
    text = re.sub(r"\b[A-Z][a-zA-Z&]*\b", " CAP ", text)
    text = re.sub(r"(\s*CAP\s*)+", " CAP ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def template_groups(texts):
    """Link rows that share the masked text OR its first 8 masked words (union-find)."""
    full = texts.map(template_key)
    first8 = full.map(lambda s: " ".join(s.split()[:8]))
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip("F" + full, "E" + first8):
        parent[find(a)] = find(b)
    return ("shade63:" + ("F" + full).map(find)).values


def load_shade63():
    df = pd.read_csv(v1.RAW / "shade63" / "sentinel_dataset_multiclass.csv")
    df["language"], df["source"], df["category"] = "English", "shade63", df.category_name
    df["group"] = template_groups(df.text)
    return df[["text", "label", "language", "source", "category", "group"]]


def build_splits():
    india, bolewara = v1.load_india14(), v1.load_bolewara()
    test_a = v1.load_india14_suite()
    test_c = bolewara[bolewara.language == "Hinglish"]
    english_rest, test_b = v1.grouped_slice(bolewara[bolewara.language == "English"], 5)
    shade = load_shade63()
    # Stratify the template hold-out by each group's majority category.
    majority = shade.groupby("group").category.agg(lambda s: s.value_counts().index[0])
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=v1.SEED)
    rest, held = next(splitter.split(shade, shade.group.map(majority), shade.group))
    shade_rest, stress_d2 = shade.iloc[rest], shade.iloc[held]
    others = pd.concat([india, english_rest], ignore_index=True)
    others["category"], others["group"] = None, "text:" + others.text.map(v1.normalise)
    pool = pd.concat([others, shade_rest], ignore_index=True)
    test_keys = set(pd.concat([test_a, test_b, test_c]).text.map(v1.normalise))
    keys = pool.text.map(v1.normalise)
    dropped = int(keys.isin(test_keys).sum())
    pool = pool[~keys.isin(test_keys)].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=v1.SEED)
    fit, held = next(splitter.split(pool, pool.label, pool.group))
    splits = {"train": pool.iloc[fit], "validation": pool.iloc[held], "test_a_india14_suite": test_a,
              "test_b_bolewara_english": test_b, "test_c_bolewara_hinglish": test_c, "stress_d2_shade63_templates": stress_d2}
    info = {"dropped_train_rows_matching_tests": dropped, "shade63_template_groups": int(shade.group.nunique()),
            "d2_template_groups": int(stress_d2.group.nunique()),
            "groups_with_both_labels": int((shade.groupby("group").label.nunique() > 1).sum()),
            "train_validation_shared_groups": int(len(set(pool.iloc[fit].group) & set(pool.iloc[held].group)))}
    return splits, info


def evaluate(predict, splits):
    results = {}
    for name, part in splits.items():
        if name == "train":
            continue
        predicted = np.asarray(predict(part.text.tolist()))
        results[name] = v1.score(part.label, predicted)
        if name == "test_a_india14_suite":
            results[name]["per_language"] = {language: v1.score(part.label[part.language == language],
                                                                predicted[(part.language == language).values])
                                             for language in v1.LANGUAGES}
        if name.startswith("stress_d2"):
            results[name]["per_category"] = {c: v1.score(part.label[part.category == c],
                                                         predicted[(part.category == c).values]) for c in CATEGORIES}
    return results


def errors(predict, splits, n=8):
    rows = []
    for name, part in splits.items():
        if name not in ("train", "validation"):
            rows.append(part.assign(test=name, predicted=np.asarray(predict(part.text.tolist()))))
    rows = pd.concat(rows)
    fp = rows[(rows.label == 0) & (rows.predicted == 1)]
    fn = rows[(rows.label == 1) & (rows.predicted == 0)]
    pick = lambda df: df.sample(min(n, len(df)), random_state=v1.SEED)[["test", "language", "category", "text"]] \
        .astype(object).where(lambda d: d.notna(), None).to_dict("records")
    return {"false_positives": pick(fp), "false_negatives": pick(fn),
            "false_positive_total": int(len(fp)), "false_negative_total": int(len(fn))}


def data_summary(splits):
    return {name: {"rows": int(len(p)), "scam": int(p.label.sum()),
                   "by_language": {k: int(v) for k, v in p.language.value_counts().items()},
                   "by_source": {k: int(v) for k, v in p.source.value_counts().items()}} for name, p in splits.items()}


def train_tfidf(root=ROOT):
    splits, info = build_splits()
    tr, va = splits["train"], splits["validation"]
    print(f"train {len(tr)} validation {len(va)}", info, flush=True)
    grid, best, best_score = [], None, -1
    for digits, preprocessor in (("keep", v1.keep_digits), ("mask", v1.mask_digits)):
        for c in v1.C_GRID:
            model = v1.make_model(preprocessor, c).fit(tr.text, tr.label)
            f1 = v1.score(va.label, model.predict(va.text.tolist()))["macro_f1"]
            grid.append({"digits": digits, "C": c, "validation_macro_f1": f1})
            print(digits, c, "validation", round(f1, 4), flush=True)
            # ONLY validation picks the configuration; ties go to the earlier setting.
            if f1 > best_score:
                best, best_score = (digits, preprocessor, c), f1
    digits, preprocessor, c = best
    model = v1.make_model(preprocessor, c).fit(tr.text, tr.label)
    joblib.dump({"model": model, "labels": ["ham", "scam"], "digits": digits, "C": c, "selection_score": best_score},
                root / "models" / "scam_binary_v2_tfidf.joblib", compress=3)
    v1_model = joblib.load(root / "models" / "scam_binary_baseline.joblib")["model"]
    report = {"splits": info, "data": data_summary(splits),
              "tfidf_v2": {"selected": {"digits": digits, "C": c, "validation_macro_f1": best_score}, "grid": grid,
                           "results": evaluate(model.predict, splits), "errors": errors(model.predict, splits)},
              "tfidf_v1": {"note": "v1 model as saved, trained without shade63", "results": evaluate(v1_model.predict, splits),
                           "errors": errors(v1_model.predict, splits)}}
    save_report(REPORTS / "scam_binary_v2_tfidf.json", report)
    print("Saved TF-IDF v2", flush=True)


def train_encoder(model_id="xlm-roberta-base", root=ROOT, epochs=4, batch_size=16, max_length=128, lr=2e-5):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from .transformer import Encoded
    torch.manual_seed(v1.SEED)
    np.random.seed(v1.SEED)
    random.seed(v1.SEED)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    splits, info = build_splits()
    tr, va = splits["train"], splits["validation"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True, num_labels=2,
        id2label={0: "ham", 1: "scam"}, label2id={"ham": 0, "scam": 1}).to(device)
    loader = DataLoader(Encoded(tokenizer(tr.text.tolist(), truncation=True, max_length=max_length), tr.label.tolist()),
                        batch_size=batch_size, shuffle=True, num_workers=0,
                        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
    counts = tr.label.value_counts()
    weights = torch.tensor([len(tr) / (2 * counts[k]) for k in (0, 1)], device=device, dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    updates = epochs * len(loader)
    warmup = max(1, int(.06 * updates))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        step / warmup if step < warmup else max(0., (updates - step) / max(1, updates - warmup)))
    scaler = torch.amp.GradScaler("cuda")
    output = root / "models" / "scam_binary_encoder" / ENCODERS[model_id]
    output.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def predict(texts):
        model.eval()
        out = []
        for start in range(0, len(texts), 256):
            inputs = tokenizer(texts[start:start + 256], truncation=True, max_length=max_length, padding=True,
                               return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out.append(model(**inputs).logits.argmax(-1).cpu().numpy())
        return np.concatenate(out)

    history, best_f1, start = [], -1, time.perf_counter()
    print(f"Fine-tuning {model_id}: {len(tr)} rows, {len(loader)} batches/epoch", flush=True)
    for epoch in range(epochs):
        model.train()
        total = 0.
        for step, batch in enumerate(loader):
            y = batch.pop("labels").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_fn(model(**{k: v.to(device) for k, v in batch.items()}).logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite encoder loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()
            if (step + 1) % 200 == 0:
                print(f"epoch {epoch+1} batch {step+1}/{len(loader)} loss {total/(step+1):.4f}", flush=True)
        f1 = v1.score(va.label, predict(va.text.tolist()))["macro_f1"]
        history.append({"epoch": epoch + 1, "train_loss": total / len(loader), "validation_macro_f1": f1})
        print(history[-1], flush=True)
        # ONLY validation picks the epoch; the best checkpoint is kept on disk.
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(output, safe_serialization=True)
            tokenizer.save_pretrained(output)
    model = AutoModelForSequenceClassification.from_pretrained(output, local_files_only=True).to(device)
    best_epoch = max(history, key=lambda h: h["validation_macro_f1"])["epoch"]
    config = {"model_id": model_id, "epochs_max": epochs, "best_epoch": best_epoch, "batch_size": batch_size,
              "max_length": max_length, "lr": lr, "precision": "float16 autocast + GradScaler",
              "class_weights": "balanced", "seed": v1.SEED, "training_rows": len(tr)}
    (output / "training_metadata.json").write_text(json.dumps({**config, "history": history}, indent=2), encoding="utf-8")
    report = {"config": config, "history": history, "seconds": time.perf_counter() - start,
              "cuda_peak_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
              "results": evaluate(predict, splits), "errors": errors(predict, splits)}
    save_report(REPORTS / f"scam_binary_v2_{ENCODERS[model_id]}.json", report)
    print("Saved encoder", output, flush=True)


def evaluate_encoder(model_id, history, seconds, epochs_max=4, root=ROOT, max_length=128):
    """Score the best saved checkpoint after training was stopped early (epoch already chosen on validation)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda")
    splits, _ = build_splits()
    output = root / "models" / "scam_binary_encoder" / ENCODERS[model_id]
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(output, local_files_only=True).to(device).eval()

    @torch.inference_mode()
    def predict(texts):
        out = []
        for start in range(0, len(texts), 256):
            inputs = tokenizer(texts[start:start + 256], truncation=True, max_length=max_length, padding=True,
                               return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out.append(model(**inputs).logits.argmax(-1).cpu().numpy())
        return np.concatenate(out)
    best_epoch = max(history, key=lambda h: h["validation_macro_f1"])["epoch"]
    config = {"model_id": model_id, "epochs_max": epochs_max, "epochs_run": len(history), "best_epoch": best_epoch,
              "batch_size": 16, "max_length": max_length, "lr": 2e-5, "precision": "float16 autocast + GradScaler",
              "class_weights": "balanced", "seed": v1.SEED, "training_rows": len(splits["train"]),
              "stopped_early": "validation macro-F1 stopped improving; best checkpoint by validation"}
    (output / "training_metadata.json").write_text(json.dumps({**config, "history": history}, indent=2), encoding="utf-8")
    report = {"config": config, "history": history, "seconds": seconds,
              "cuda_peak_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
              "results": evaluate(predict, splits), "errors": errors(predict, splits)}
    save_report(REPORTS / f"scam_binary_v2_{ENCODERS[model_id]}.json", report)
    print("Scored saved encoder", output, flush=True)


def pct(value):
    return "n/a" if value is None else f"{value:.3f}"


def write_report(peaks=None):
    tfidf = json.loads((REPORTS / "scam_binary_v2_tfidf.json").read_text(encoding="utf-8"))
    models = {"TF-IDF v1": tfidf["tfidf_v1"], "TF-IDF v2": tfidf["tfidf_v2"]}
    for model_id, short in ENCODERS.items():
        path = REPORTS / f"scam_binary_v2_{short}.json"
        if path.exists():
            models[f"{model_id} (v2 data)"] = json.loads(path.read_text(encoding="utf-8"))
    tests = ("test_a_india14_suite", "test_b_bolewara_english", "test_c_bolewara_hinglish", "stress_d2_shade63_templates")
    # Best model = highest validation macro-F1 among v2-trained models (v1 never saw this validation set's shade63 rows).
    candidates = {k: m for k, m in models.items() if k != "TF-IDF v1"}
    best = max(candidates, key=lambda k: candidates[k]["results"]["validation"]["macro_f1"])
    s = tfidf["splits"]
    lines = ["# Scam vs not-scam v2: shade63 added, TF-IDF and encoders", "",
             "In v1 the model flagged half of shade63's real bank and OTP alerts, because india14 only has bank words "
             "on the scam side. For v2 I added shade63 to training. shade63 is almost all templates, so I split it by "
             "template: I masked amounts, anything with a digit and capitalised words (names, banks, brands, places), "
             "then linked rows that share the masked text or its first 8 masked words. That gives "
             f"{s['shade63_template_groups']} template groups. I held out {s['d2_template_groups']} of them "
             "(about a third, stratified by category) as stress set D2 and never trained on them. Tests A, B and C "
             "are the same rows as in v1.", "",
             f"Template groups with both labels: {s['groups_with_both_labels']}. Groups shared between train and "
             f"validation: {s['train_validation_shared_groups']}. Training rows dropped as near-twins of a test row: "
             f"{s['dropped_train_rows_matching_tests']}.", "",
             "## Data", "", "| split | rows | scam | languages | sources |", "|---|---|---|---|---|"]
    for name, d in tfidf["data"].items():
        lines.append(f"| {name} | {d['rows']:,} | {d['scam']:,} | " + ", ".join(f"{k} {v:,}" for k, v in d["by_language"].items())
                     + " | " + ", ".join(f"{k} {v:,}" for k, v in d["by_source"].items()) + " |")
    sel = tfidf["tfidf_v2"]["selected"]
    lines += ["", "## How each model was chosen", "",
              f"- TF-IDF v1: the saved v1 model, unchanged (trained without shade63).",
              f"- TF-IDF v2: same features and grid as v1; picked digits={sel['digits']}, C={sel['C']} "
              f"(validation macro-F1 {sel['validation_macro_f1']:.4f})."]
    for key, m in models.items():
        if "config" in m:
            c = m["config"]
            lines.append(f"- {key}: max_len {c['max_length']}, fp16, batch {c['batch_size']}, lr {c['lr']}, up to "
                         f"{c['epochs_max']} epochs; picked epoch {c['best_epoch']} on validation ("
                         + ", ".join(f"e{h['epoch']} {h['validation_macro_f1']:.4f}" for h in m["history"])
                         + f"). Training took {m['seconds']/60:.1f} min, GPU memory peak {m['cuda_peak_mb']:.0f} MB.")
    lines += ["", "## Comparison", "",
              "| model | validation | Test A | Test B | Test C | D2 macro-F1 | D2 legit FPR | D2 scam recall |",
              "|---|---|---|---|---|---|---|---|"]
    for key, m in models.items():
        r = m["results"]
        d2 = r["stress_d2_shade63_templates"]
        lines.append(f"| {key} | {pct(r['validation'].get('macro_f1'))} | " + " | ".join(pct(r[t].get("macro_f1")) for t in tests[:3])
                     + f" | {pct(d2.get('macro_f1'))} | {pct(d2['per_category']['legit'].get('ham_false_positive_rate'))} | "
                     f"{pct(d2.get('scam_recall'))} |")
    lines += ["", "The validation column for TF-IDF v1 is on the v2 validation set, which includes shade63 templates v1 never saw.", ""]
    for key, m in models.items():
        r = m["results"]
        lines += [f"## {key}", "",
                  "| set | n | macro-F1 | scam precision | scam recall | ham FPR | confusion [[TN, FP], [FN, TP]] |",
                  "|---|---|---|---|---|---|---|"]
        for name in ("validation", *tests):
            x = r[name]
            lines.append(f"| {name} | {x['n']:,} | {pct(x.get('macro_f1'))} | {pct(x.get('scam_precision'))} | "
                         f"{pct(x.get('scam_recall'))} | {pct(x.get('ham_false_positive_rate'))} | {x['confusion_matrix']} |")
        lines += ["", "| Test A language / D2 category | n | macro-F1 | scam precision | scam recall | ham FPR |", "|---|---|---|---|---|---|"]
        for group, values in (("A", r[tests[0]]["per_language"]), ("D2", r[tests[3]]["per_category"])):
            for k, x in values.items():
                lines.append(f"| {group}: {k} | {x['n']:,} | {pct(x.get('macro_f1'))} | {pct(x.get('scam_precision'))} | "
                             f"{pct(x.get('scam_recall'))} | {pct(x.get('ham_false_positive_rate'))} |")
        lines.append("")
    e = models[best]["errors"]
    lines += [f"## Errors from the best model by validation ({best})", ""]
    for kind, title in (("positive", "False positives (ham called scam)"), ("negative", "False negatives (scam called ham)")):
        shown = e[f"false_{kind}s"]
        lines += [f"{title}: {len(shown)} random of {e[f'false_{kind}_total']:,} across tests A-C and D2", ""]
        lines += [f"- [{x['test']}, {x['category'] or x['language']}] {x['text']}".replace("\n", " ") for x in shown]
        lines.append("")
    if peaks:
        lines += ["## Heat", "", peaks, ""]
    combined = {"best_by_validation": best, "models": models, "splits": s, "data": tfidf["data"], "peaks": peaks}
    save_report(REPORTS / "scam_binary_v2.json", combined)
    (REPORTS / "scam_binary_v2.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports/scam_binary_v2.md; best", best, flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Scam vs not-scam v2")
    parser.add_argument("step", choices=["tfidf", "encoder", "encoder-eval", "report"])
    parser.add_argument("--model", choices=list(ENCODERS), default="xlm-roberta-base")
    parser.add_argument("--peaks")
    parser.add_argument("--history", help="encoder-eval: JSON list of per-epoch validation results from the training log")
    parser.add_argument("--seconds", type=float, default=0.)
    args = parser.parse_args()
    # Import through the package so pickled preprocessors point at cyberlens.scam_binary, not __main__.
    from cyberlens import scam_binary_v2 as package
    if args.step == "tfidf":
        package.train_tfidf()
    elif args.step == "encoder":
        package.train_encoder(args.model)
    elif args.step == "encoder-eval":
        package.evaluate_encoder(args.model, json.loads(open(args.history, encoding="utf-8").read()), args.seconds)
    else:
        package.write_report(args.peaks)


if __name__ == "__main__":
    main()
