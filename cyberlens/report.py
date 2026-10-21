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
    robust_path = reports/"call_scam_robust.json"
    robust = json.loads(robust_path.read_text()) if robust_path.exists() else None
    regularized_path = reports/"complaint_regularized.json"
    regularized = json.loads(regularized_path.read_text()) if regularized_path.exists() else None
    margin_path = reports/"complaint_margin.json"
    margin = json.loads(margin_path.read_text()) if margin_path.exists() else None
    continued_path = reports/"complaint_continued.json"
    continued = json.loads(continued_path.read_text()) if continued_path.exists() else None
    semantic_experiments = {}
    for name in ("semantic","ensemble"):
        path = reports/f"complaint_{name}.json"
        if path.exists():
            semantic_experiments[name] = json.loads(path.read_text())
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
    if regularized and regularized["adopt"]:
        measured["complaint"]["regularized"] = regularized
        lines.append(f"| complaint | regularized {regularized['selected']} | {regularized['validation']['macro_f1']:.4f} | {regularized['test']['macro_f1']:.4f} | {regularized['test_typos']['macro_f1']:.4f} | {regularized['test']['accuracy']:.4f} | {regularized['seconds']:.1f} |")
    if margin and margin["adopt"]:
        measured["complaint"]["margin"] = margin
        lines.append(f"| complaint | margin {margin['selected']} | {margin['validation']['macro_f1']:.4f} | {margin['test']['macro_f1']:.4f} | {margin['test_typos']['macro_f1']:.4f} | {margin['test']['accuracy']:.4f} | {margin['seconds']:.1f} |")
    for name,r in semantic_experiments.items():
        if r["adopt"]:
            measured["complaint"][name] = r
            lines.append(f"| complaint | {name} | {r['validation']['macro_f1']:.4f} | {r['test']['macro_f1']:.4f} | {r['test_typos']['macro_f1']:.4f} | {r['test']['accuracy']:.4f} | {r['seconds']:.1f} |")
    if continued and continued["adopt"]:
        measured["complaint"]["continued"] = continued
        lines.append(f"| complaint | continued encoder | {continued['validation']['macro_f1']:.4f} | {continued['test']['macro_f1']:.4f} | {continued['test_typos']['macro_f1']:.4f} | {continued['test']['accuracy']:.4f} | {continued['seconds']:.1f} |")
    champions = {}
    for task, candidates in measured.items():
        best = max(candidates, key=lambda k:(candidates[k]["validation"]["macro_f1"] + candidates[k]["validation_typos"]["macro_f1"])/2)
        champions[task] = {"backend":best,"selection":"mean clean + synthetic-typo validation macro-F1",
                           "score":(candidates[best]["validation"]["macro_f1"]+candidates[best]["validation_typos"]["macro_f1"])/2}
    if robust and robust["adopt"]:
        champions["call_scam"] = {"backend":"robust","selection":"nested family CV with fixed uncertainty/promotion rule",
                                  "score":(robust["nested"]["macro_f1"]+robust["nested_typos"]["macro_f1"])/2}
    (ROOT/"models/champions.json").write_text(json.dumps(champions,indent=2),encoding="utf-8")
    if family:
        lines += ["", "## Harder call evaluation: template families", "",
                  f"I grouped the 743 messages into **{family['families']}** near-duplicate families before evaluating the frozen baseline with five-fold group cross-validation.",
                  f"Family-held-out macro-F1: **{family['oof']['macro_f1']:.4f}**; typo macro-F1: **{family['oof_typos']['macro_f1']:.4f}**; accuracy: **{family['oof']['accuracy']:.4f}**.",
                  "", "Each message is evaluated exactly once; no family crosses a fold's training and test sets. All four classes are represented in training for each recorded fold.",
                  "The 1.0000 call scores above came from the earlier exact-deduplicated split. They were too optimistic because related templates remained on both sides. I keep them for comparison, but use the family-held-out result in the README headline.",
                  "Family grouping uses a 0.80 character-TF-IDF cosine threshold and connected components. This is a heuristic stress test, not a human-verified semantic split or an independent real-world benchmark."]
    if robust:
        interval = robust["paired_uncertainty"]
        lines += ["", "## Bounded call improvement experiment", "",
                  "I compared six fixed candidates inside three inner family folds, then assessed the selection procedure on five outer family folds. The outer assessment messages never fitted a vocabulary, scaler or classifier in their fold.",
                  "", "| Model procedure | Clean macro-F1 | Typo macro-F1 | Accuracy |", "|---|---:|---:|---:|",
                  f"| Reference word baseline | {robust['reference']['macro_f1']:.4f} | {robust['reference_typos']['macro_f1']:.4f} | {robust['reference']['accuracy']:.4f} |",
                  f"| Nested candidate selection | {robust['nested']['macro_f1']:.4f} | {robust['nested_typos']['macro_f1']:.4f} | {robust['nested']['accuracy']:.4f} |",
                  "", f"The approximate paired family-bootstrap 95% interval for the **clean F1 difference** is **[{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]** ({interval['repetitions']} resamples). This is wide and includes zero.",
                  f"My fixed promotion rule was: {robust['adoption_rule']}. Passed: **{robust['adopt']}**.",
                  f"The optional `robust` artifact uses **{robust['deployment_variant']}**, selected with a separate grouped inner search and refitted on all {robust['deployment_training_rows']} development messages. Its training-set performance is not the reported nested score.",
                  ("The candidate passed the fixed promotion rule." if robust["adopt"] else "I kept the baseline as the default.")+" The encoder is frozen; only a small L2-regularized classification head learns labels. This reduces capacity but does not prove absence of overfitting.",
                  "This corpus was explored before this experiment, families are heuristic, and several classes have few independent families. Fresh human-labelled English/Hinglish messages are still needed for independent confirmation."]
    if regularized:
        lines += ["", "## Bounded complaint improvement experiment", "",
                  "I compared three fixed heads on the same training-only TF-IDF features. Selection uses only the mean clean/noisy validation macro-F1, with a fixed minimum gain of 0.02 before promotion.",
                  "", "| Head | Clean validation macro-F1 | Typo validation macro-F1 | Selection mean |", "|---|---:|---:|---:|"]
        for name,r in regularized["search"].items():
            lines.append(f"| {name} | {r['validation']['macro_f1']:.4f} | {r['validation_typos']['macro_f1']:.4f} | {r['selection_score']:.4f} |")
        lines += ["", f"Reference selection mean: **{regularized['reference_selection_score']:.4f}**. Selected head: **{regularized['selected']}**. Promotion rule passed: **{regularized['adopt']}**.",
                  regularized.get("test_policy","Only the selected candidate was evaluated on the supplied test after passing the validation criterion."),
                  "The validation set has been used in earlier development, so this remains development evidence. I did not change labels, remove difficult classes or choose settings from test scores."]
    if margin:
        lines += ["", "## Inner-selected complaint margin experiment", "",
                  f"I split only the original training partition into {margin['inner_train_rows']} inner-training and {margin['inner_validation_rows']} inner-validation messages. Its TF-IDF vocabulary was fitted on inner training only; normalized family overlap was {margin['inner_overlapping_groups']}.",
                  "I fixed three heads: averaged SGD hinge, averaged SGD modified Huber, and a balanced linear SVM at C=1. All use hybrid word/character features. Earlier word-only SVM checks were weaker; this comparison uses the combined representation.",
                  "", "| Head | Inner clean/noisy mean macro-F1 |", "|---|---:|"]
        for name,r in margin["search"].items():
            lines.append(f"| {name} | {r['selection_score']:.4f} |")
        lines += ["", f"Inner-selected head: **{margin['selected']}**. After refitting on the full original training split, original validation mean macro-F1 was **{margin['selection_score']:.4f}**, versus **{margin['reference_selection_score']:.4f}** for the baseline. Promotion rule passed: **{margin['adopt']}**.",
                  margin.get("test_policy","The selected candidate passed the validation rule before official-test evaluation."),
                  "The original validation corpus was used in earlier experiments. The inner split limits reuse during this search but does not make the existing validation/test corpus fresh independent evidence. Softmax-converted margins are ranking scores, not calibrated probabilities."]
    if semantic_experiments:
        lines += ["", "## Frozen complaint semantics and fixed ensemble", "",
                  "I also tested fixed pretrained multilingual representations. C=0.01 or 0.1 is selected inside the original training split, with scalers fitted on inner training only. The chosen head is refitted on full original training before the original validation check.",
                  "The additional ensemble is one predeclared 50/50 average with the baseline. I did not search blend weights. Components use the same class vocabulary with explicit column alignment.",
                  "", "| Candidate | Clean validation macro-F1 | Typo validation macro-F1 | Mean | Promotion passed |", "|---|---:|---:|---:|---|"]
        for name,r in semantic_experiments.items():
            lines.append(f"| {name} | {r['validation']['macro_f1']:.4f} | {r['validation_typos']['macro_f1']:.4f} | {r['selection_score']:.4f} | {r['adopt']} |")
        for name,r in semantic_experiments.items():
            lines += ["",f"{name}: "+r.get("test_policy","Validation criterion passed before test evaluation.")]
        lines += ["", "All of these runs use an already explored development corpus. The bounded search and frozen encoder reduce some overfitting risks; fresh independent labels are still needed to confirm transfer."]
    if continued:
        lines += ["", "## Bounded encoder continuation", "",
                  "I continued the existing one-epoch checkpoint for at most two further epochs, with learning rate 2e-5, weight decay 0.01, effective batch 32 and the same 192-token limit. I kept square-root inverse-frequency loss weights and training-only typo augmentation.",
                  "After each epoch I checked mean clean/noisy validation macro-F1. A gain below 0.005 stopped training; the best qualifying checkpoint was retained. The optimizer/scheduler restarted for this bounded continuation.",
                  "", "| Total epoch | Training loss | Clean/noisy validation mean F1 | Checkpoint improved |", "|---|---:|---:|---|"]
        for row in continued["history"]:
            lines.append(f"| {row['total_epoch']} | {row['training_loss']:.4f} | {row['selection_score']:.4f} | {row['checkpoint_improved']} |")
        lines += ["", f"Selected total epoch: **{continued['selected_total_epoch']}**. Validation mean: **{continued['selection_score']:.4f}**, versus baseline **{continued['reference_selection_score']:.4f}**. Promotion rule passed: **{continued['adopt']}**.",
                  continued.get("test_policy","Only the selected checkpoint was evaluated on test after passing the validation rule."),
                  "The original validation set has been reused; this bounded budget and stopping rule reduce some risks but do not establish absence of overfitting or independent generalisation."]
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
            if task == "call_scam" and info["backend"] == "robust":
                clean,noisy,protocol = robust["nested"],robust["nested_typos"],"Nested template-family CV"
            elif task == "call_scam" and family:
                clean,noisy,protocol = family["oof"],family["oof_typos"],"5-fold template-family CV (frozen baseline)"
            else:
                r = measured[task][info["backend"]]
                clean,noisy,protocol = r["test"],r["test_typos"],"Supplied test.csv"
            summary_lines.append(f"| {task} | {info['backend']} | {protocol} | {clean['macro_f1']:.4f} | {noisy['macro_f1']:.4f} | {clean['accuracy']:.4f} |")
        summary_lines += ["", "[Full comparisons and failure cases](reports/RESULTS.md). Scores are fractions, not percentages.", ""]
        if robust and not robust["adopt"]:
            summary_lines += [f"The optional `robust` call experiment reached **{robust['nested']['macro_f1']:.4f}** clean / **{robust['nested_typos']['macro_f1']:.4f}** typo macro-F1 in nested family CV. I kept it experimental: the approximate paired family-bootstrap interval for its clean improvement includes zero. These are development results, pending fresh independent data.", ""]
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
