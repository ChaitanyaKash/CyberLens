"""Generate a human-readable experiment record from measured artifacts."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from .data import ROOT, TASKS


def main():
    reports = ROOT / "reports"
    audit = json.loads((reports / "data_audit.json").read_text(encoding="utf-8"))
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "utc_time": datetime.now(timezone.utc).isoformat(), "packages": {}}
    for name in ("numpy", "pandas", "scikit-learn", "joblib", "torch", "transformers"):
        try:
            environment["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        import torch
        environment["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            environment["gpu"] = torch.cuda.get_device_name(0)
            environment["vram_gb"] = torch.cuda.get_device_properties(0).total_memory/1024**3
    except ImportError:
        pass
    (reports / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (ROOT / "requirements-tested.txt").write_text("# Versions already present during the tested run; no packages installed by this project.\n" +
        "\n".join(f"{k}=={v}" for k,v in environment["packages"].items()) + "\n", encoding="utf-8")
    copies = []
    for directory in (ROOT / "data/raw", ROOT / "reference"):
        for file in directory.rglob("*"):
            if not file.is_file():
                continue
            relative = file.relative_to(directory)
            original = ROOT.parent / relative
            if not original.is_file():
                continue
            a = hashlib.sha256(file.read_bytes()).hexdigest()
            b = hashlib.sha256(original.read_bytes()).hexdigest()
            copies.append({"copy": str(file.relative_to(ROOT)), "original": str(original),
                           "copy_sha256": a, "original_sha256": b, "matches": a == b})
    (reports / "source_integrity.json").write_text(json.dumps(copies, indent=2), encoding="utf-8")
    if not all(x["matches"] for x in copies):
        raise RuntimeError("A source copy differs from its original")
    lines = ["# My experiment results", "", "I kept the full numbers here so I can compare the models without relying on memory. These tables come from the saved JSON reports; the source CSVs are unchanged.", "",
             "## Dataset audit", "", "| Task | Raw training rows | Training | Validation | Test | Trainable labels |",
             "|---|---:|---:|---:|---:|---:|"]
    for task, info in audit.items():
        s = info["splits"]
        lines.append(f"| {task} | {info['raw_rows']} | {s['train']} | {s['validation']} | {s['test']} | {info['labels']} |")
    lines += ["", "Complaint test uses the supplied `test.csv`. Overlapping normalized templates are removed from training, not from test.",
              "", f"Complaint training rows removed for official-test overlap: **{audit['complaint']['train_rows_removed_due_to_test_overlap']}**.",
              f"Complaint test repeated templates: **{audit['complaint']['test_repeated_templates']}**; conflicting templates: **{audit['complaint']['test_conflicting_templates']}**.",
              "", "Five contradictory Hinglish subtypes are merged into `other_scam`; see `call_label_conflicts.csv`.",
              "Normalized duplicates are controlled, but semantic paraphrases may still cross splits.",
              "", "## Held-out metrics", "", "All scores below are fractions, not percentages. Typo tests use synthetic internal-letter swaps in 25% of eligible words.",
              "", "| Task | Model | Validation macro-F1 | Test macro-F1 | Typo test macro-F1 | Test accuracy | Seconds |",
              "|---|---|---:|---:|---:|---:|---:|"]
    measured = {}
    family_path = reports/"call_family_evaluation.json"
    family = json.loads(family_path.read_text()) if family_path.exists() else None
    for task in TASKS:
        baseline = json.loads((reports / f"{task}_baseline.json").read_text())
        for mode in ("word", "char", "hybrid"):
            r = baseline[mode]
            suffix = " (selected baseline)" if mode == baseline["selected"] else ""
            lines.append(f"| {task} | TF-IDF {mode}{suffix} | {r['validation']['macro_f1']:.4f} | {r['test']['macro_f1']:.4f} | {r['test_typos']['macro_f1']:.4f} | {r['test']['accuracy']:.4f} | {r['seconds']:.1f} |")
        measured[task] = {"baseline": baseline[baseline["selected"]]}
        for mode in ("word2vec", "subword"):
            r = json.loads((reports / f"{task}_{mode}.json").read_text())
            measured[task][mode] = r
            lines.append(f"| {task} | {mode} | {r['validation']['macro_f1']:.4f} | {r['test']['macro_f1']:.4f} | {r['test_typos']['macro_f1']:.4f} | {r['test']['accuracy']:.4f} | {r['seconds']:.1f} |")
        for mode,title in (("transformer","multilingual DistilBERT"),("hierarchical","financial expert + baseline")):
            extra_path = reports / f"{task}_{mode}.json"
            if extra_path.exists():
                r = json.loads(extra_path.read_text())
                measured[task][mode] = r
                lines.append(f"| {task} | {title} | {r['validation']['macro_f1']:.4f} | {r['test']['macro_f1']:.4f} | {r['test_typos']['macro_f1']:.4f} | {r['test']['accuracy']:.4f} | {r['seconds']:.1f} |")
    champions = {}
    for task, candidates in measured.items():
        best = max(candidates, key=lambda k:(candidates[k]["validation"]["macro_f1"] + candidates[k]["validation_typos"]["macro_f1"])/2)
        champions[task] = {"backend":best,"selection":"mean clean + synthetic-typo validation macro-F1",
                           "score":(candidates[best]["validation"]["macro_f1"]+candidates[best]["validation_typos"]["macro_f1"])/2}
    (ROOT/"models/champions.json").write_text(json.dumps(champions,indent=2),encoding="utf-8")
    if family:
        lines += ["", "## Harder call evaluation: template families", "",
                  f"I grouped the 743 messages into **{family['families']}** near-duplicate families before evaluating the frozen baseline with five-fold group cross-validation.",
                  f"Family-held-out macro-F1: **{family['oof']['macro_f1']:.4f}**; typo macro-F1: **{family['oof_typos']['macro_f1']:.4f}**; accuracy: **{family['oof']['accuracy']:.4f}**.",
                  "", "Each message is evaluated exactly once; no family crosses a fold's training and test sets. All four classes are represented in training for each recorded fold.",
                  "The 1.0000 call scores above came from the earlier exact-deduplicated split. They were too optimistic because related templates remained on both sides. I keep them for comparison, but use the family-held-out result in the README headline.",
                  "Family grouping uses a 0.80 character-TF-IDF cosine threshold and connected components. This is a heuristic stress test, not a human-verified semantic split or an independent real-world benchmark."]
    lines += ["", "Default inference uses validation-selected backends:", ""]
    for task, info in champions.items():
        lines.append(f"- {task}: **{info['backend']}** (selection score {info['score']:.4f}).")
    optimization = reports/"optimizer_validation.json"
    if optimization.exists():
        lines += ["", "Additional word-feature optimizer checks (validation only):", ""]
        for name,r in json.loads(optimization.read_text()).items():
            lines.append(f"- {name}: macro-F1 {r['validation_macro_f1']:.4f}, accuracy {r['validation_accuracy']:.4f}.")
    lines += ["", "Baseline selection uses the mean of clean and noisy validation macro-F1, never test scores.",
              "Word2Vec/subword use fixed learning-experiment settings, not an exhaustive hyperparameter search.",
              "A perfect result on the small templated call dataset does not establish real-world or cross-language reliability.",
              "", "## Selected baseline: weakest complaint classes", "", "| Class | Precision | Recall | F1 | Test support |",
              "|---|---:|---:|---:|---:|"]
    classes = measured["complaint"]["baseline"]["test"]["per_class"]
    for label, row in sorted(((k,v) for k,v in classes.items() if isinstance(v,dict) and "f1-score" in v
                              and k not in ("macro avg", "weighted avg", "micro avg")), key=lambda p:p[1]["f1-score"])[:10]:
        lines.append(f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1-score']:.3f} | {row['support']:.0f} |")
    lines += ["", "## Duplicate-aware official-test sensitivity", ""]
    for mode, r in measured["complaint"].items():
        if "test_unique_examples" in r:
            unique = r["test_unique_examples"]
            lines.append(f"- {mode}: deduplicated (template, label) test macro-F1 **{unique['macro_f1']:.4f}** on {unique['n']} examples. Contradictory labels are retained.")
    challenge_path = reports / "manual_challenge.json"
    if challenge_path.exists():
        challenge = json.loads(challenge_path.read_text())
        lines += ["", "## Small manual diagnostic", "", "Hand-authored mixed-language development examples, not a representative benchmark and never used for fitting or model selection. Expected labels were clarified after inspecting source semantics; collect a fresh independent gold set for a final claim.", ""]
        for backend, r in challenge.items():
            lines.append(f"- {backend}: {r['correct']}/{r['total']} correct. Inspect every example in `manual_challenge.json`, including warning/negation failures.")
            for item in [x for x in r["records"] if x["expected"] != x["predicted"]][:6]:
                lines.append(f"  - Expected `{item['expected']}`, predicted `{item['predicted']}` (score {item['score']:.3f}): {item['text']}")
    lines += ["", "## Integrity and environment", "", f"Verified **{len(copies)}** copied source files against their originals using SHA-256.",
              "See `source_integrity.json` and `environment.json` for exact values.",
              "", "## Limits", "", "- English and Hinglish label vocabularies differ; cross-language transfer is not independently validated.",
              "- Complaint classification has no benign class; call `none` does not prove safety.",
              "- Rule-based PII masking is incomplete; raw copies/reference notebooks remain sensitive.",
              "- Confidence scores and review threshold are uncalibrated; unknown scams can receive high scores.",
              "- Final reports supersede internal-test exploratory artifacts under `superseded/`.",
              "- No RAG layer, live-call integration, or automatic complaint submission is included."]
    (reports / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    readme = ROOT/"README.md"
    readme_text = readme.read_text(encoding="utf-8")
    start_marker, end_marker = "<!-- RESULTS SUMMARY START -->", "<!-- RESULTS SUMMARY END -->"
    if start_marker in readme_text and end_marker in readme_text:
        summary_lines = ["", "The protocols differ, so I list them explicitly. The call result uses the harder family split; the complaint result uses the supplied test file.", "",
                         "| Task | Backend | Evaluation protocol | Macro-F1 | Typo macro-F1 | Accuracy |",
                         "|---|---|---|---:|---:|---:|"]
        for task,info in champions.items():
            r = measured[task][info["backend"]]
            clean, noisy, protocol = r["test"],r["test_typos"],"Supplied test.csv"
            if task == "call_scam" and family:
                clean,noisy,protocol = family["oof"],family["oof_typos"],"5-fold template-family CV (frozen baseline)"
            summary_lines.append(f"| {task} | {info['backend']} | {protocol} | {clean['macro_f1']:.4f} | {noisy['macro_f1']:.4f} | {clean['accuracy']:.4f} |")
        summary_lines += ["", "[Full comparisons and failure cases](reports/RESULTS.md). Scores are fractions, not percentages.", ""]
        before, after = readme_text.split(start_marker,1)
        _, after = after.split(end_marker,1)
        readme.write_text(before+start_marker+"\n".join(summary_lines)+end_marker+after,encoding="utf-8")
    guide = ROOT/"INTERVIEW_GUIDE.md"
    guide_text = guide.read_text(encoding="utf-8").split("\n<!-- GENERATED RESULTS -->")[0]
    guide.write_text(guide_text+"\n<!-- GENERATED RESULTS -->\n\n## 19. Final measured outcomes (automatically generated)\n\n"+
                     "\n".join(lines[2:])+"\n", encoding="utf-8")
    print("Wrote reports/RESULTS.md and verified source integrity")


if __name__ == "__main__":
    main()
