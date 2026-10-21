"""Generate a human-readable experiment record from measured artifacts."""
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
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
    (ROOT / "requirements-tested.txt").write_text("# Dependency versions used for the recorded experiments.\n" +
        "\n".join(f"{k}=={v}" for k,v in environment["packages"].items()) + "\n", encoding="utf-8")
    lines = ["# Evaluation Results", "", "Metrics are generated from the saved experiment reports. Evaluation protocols are listed explicitly because the call and complaint datasets use different taxonomies and split strategies.", "",
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
    adjusted_path = reports/"complaint_adjusted.json"
    adjusted = json.loads(adjusted_path.read_text()) if adjusted_path.exists() else None
    improved_path = reports/"complaint_improved.json"
    improved = json.loads(improved_path.read_text()) if improved_path.exists() else None
    supervised_path = reports/"complaint_supervised.json"
    supervised = json.loads(supervised_path.read_text()) if supervised_path.exists() else None
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
    if improved and improved["adopt"]:
        measured["complaint"]["improved"] = improved
        lines.append(f"| complaint | confirmed {improved['selected']} | {improved['validation']['macro_f1']:.4f} | {improved['test']['macro_f1']:.4f} | {improved['test_typos']['macro_f1']:.4f} | {improved['test']['accuracy']:.4f} | {improved['seconds']:.1f} |")
    if supervised and supervised["adopt"]:
        measured["complaint"]["supervised"] = supervised
        lines.append(f"| complaint | supervised word/character network | {supervised['validation']['macro_f1']:.4f} | {supervised['test']['macro_f1']:.4f} | {supervised['test_typos']['macro_f1']:.4f} | {supervised['test']['accuracy']:.4f} | {supervised['seconds']:.1f} |")
    champions = {}
    for task, candidates in measured.items():
        best = max(candidates, key=lambda k:(candidates[k]["validation"]["macro_f1"] + candidates[k]["validation_typos"]["macro_f1"])/2)
        champions[task] = {"backend":best,"selection":"mean clean + synthetic-typo validation macro-F1",
                           "score":(candidates[best]["validation"]["macro_f1"]+candidates[best]["validation_typos"]["macro_f1"])/2}
    if robust and robust["adopt"]:
        champions["call_scam"] = {"backend":"robust","selection":"nested family CV with fixed uncertainty/promotion rule",
                                  "score":(robust["nested"]["macro_f1"]+robust["nested_typos"]["macro_f1"])/2}
    if adjusted and adjusted["adopt"]:
        # Its validation set comes from a different training pool, so it is promoted by its own
        # predeclared test rule rather than by comparing validation scores across protocols.
        champions["complaint"] = {"backend":"adjusted","selection":"official-train protocol; validation grid; predeclared adoption rule",
                                  "score":adjusted["selected_validation"]["selection"]}
    (ROOT/"models/champions.json").write_text(json.dumps(champions,indent=2),encoding="utf-8")
    if adjusted:
        old, ref, new = adjusted["default_slices"], adjusted["reference_refit"], adjusted["selected_refit"]
        cfg = adjusted["selected"]
        lines += ["", "## Official-train protocol and class-prior correction", "",
                  "The earlier preparation removed every training row whose normalized template also occurs in `test.csv`. That removal consults the test file and discards 12,778 rows (13.6% of `train.csv`); one subtype kept 8% of its rows. In the official-train protocol, training uses all of `train.csv` and `test.csv` is not consulted. Validation remains a template-disjoint 20% of the training pool. Conflicting training templates keep their plurality label when it covers at least 60% of rows; others are excluded.",
                  "Balanced class weights left rare subtypes with low precision. A post-hoc correction adds tau times the log training prior to log-probabilities. SGD alpha and tau were selected from a fixed 4 x 5 grid by mean clean/noisy validation macro-F1, then refitted on training plus validation.",
                  "", "| Alpha | Tau | Clean validation macro-F1 | Typo validation macro-F1 | Clean validation accuracy |", "|---:|---:|---:|---:|---:|"]
        for row in adjusted["grid"]:
            mark = " (selected)" if (row["alpha"], row["tau"]) == (cfg["alpha"], cfg["tau"]) else ""
            lines.append(f"| {row['alpha']:g} | {row['tau']:g}{mark} | {row['clean_macro_f1']:.4f} | {row['typo_macro_f1']:.4f} | {row['clean_accuracy']:.4f} |")
        lines += ["", "Validation sets differ between protocols, so these validation scores are not comparable with earlier sections. The test comparison below uses identical test rows.",
                  "", "| Test slice | Rows | Previous default macro-F1 | Same model, official-train | Selected macro-F1 | Previous accuracy | Selected accuracy |", "|---|---:|---:|---:|---:|---:|---:|"]
        names = {"test":"Full official test","test_typos":"Full test, synthetic typos","test_unique_examples":"Unique test templates","test_novel_templates":"Templates absent from train.csv"}
        for key,title in names.items():
            lines.append(f"| {title} | {old[key]['n']} | {old[key]['macro_f1']:.4f} | {ref[key]['macro_f1']:.4f} | {new[key]['macro_f1']:.4f} | {old[key]['accuracy']:.4f} | {new[key]['accuracy']:.4f} |")
        u = adjusted["uncertainty_vs_default"]
        lines += ["", f"Paired normalized-template bootstrap 95% intervals against the previous default ({u['test']['repetitions']} resamples): full-test macro-F1 gain [{u['test']['lower_95']:.4f}, {u['test']['upper_95']:.4f}], accuracy gain [{u['test']['accuracy_lower_95']:.4f}, {u['test']['accuracy_upper_95']:.4f}]; novel-template macro-F1 gain [{u['test_novel_templates']['lower_95']:.4f}, {u['test_novel_templates']['upper_95']:.4f}], accuracy gain [{u['test_novel_templates']['accuracy_lower_95']:.4f}, {u['test_novel_templates']['accuracy_upper_95']:.4f}].",
                  f"Adoption rule (declared before evaluation): {adjusted['adoption_rule']}. Passed: **{adjusted['adopt']}**.",
                  "Most of the full-test macro-F1 gain comes from subtypes dominated by recurring templates; macro-F1 on templates absent from `train.csv` is unchanged within uncertainty, while accuracy there improves through the prior correction. Some small subtypes decline. The test file has been inspected in earlier development and is not fresh independent evidence."]
        diffs = sorted(((k, old_f, new["test"]["per_class"][k]["f1-score"]) for k, old_f in
                        ((k, v["f1-score"]) for k, v in adjusted["default_slices_per_class"].items())), key=lambda x: x[2]-x[1])
        lines += ["", "| Subtype | Test support | Previous F1 | Selected F1 |", "|---|---:|---:|---:|"]
        for k, a, b in diffs[-5:][::-1]+diffs[:5]:
            lines.append(f"| {k} | {int(new['test']['per_class'][k]['support'])} | {a:.4f} | {b:.4f} |")
        lines += ["", "The table lists the five largest per-subtype gains and declines."]
    if family:
        lines += ["", "## Harder call evaluation: template families", "",
                  f"The 743 messages form **{family['families']}** near-duplicate families. The fixed baseline is evaluated with five-fold group cross-validation.",
                  f"Family-held-out macro-F1: **{family['oof']['macro_f1']:.4f}**; typo macro-F1: **{family['oof_typos']['macro_f1']:.4f}**; accuracy: **{family['oof']['accuracy']:.4f}**.",
                  "", "Each message is evaluated exactly once; no family crosses a fold's training and test sets. All four classes are represented in training for each recorded fold.",
                  "The 1.0000 call scores above use the earlier exact-deduplicated split. Related templates crossed that split, producing optimistic estimates. The primary call result uses family-held-out evaluation.",
                  "Family grouping uses a 0.80 character-TF-IDF cosine threshold and connected components. This is a heuristic stress test, not a human-verified semantic split or an independent real-world benchmark."]
    if robust:
        interval = robust["paired_uncertainty"]
        lines += ["", "## Bounded call improvement experiment", "",
                  "Six fixed candidates are compared within three inner family folds. Five outer family folds assess the selection procedure. Vocabularies, scalers and classifiers are fitted only on the corresponding training fold.",
                  "", "| Model procedure | Clean macro-F1 | Typo macro-F1 | Accuracy |", "|---|---:|---:|---:|",
                  f"| Reference word baseline | {robust['reference']['macro_f1']:.4f} | {robust['reference_typos']['macro_f1']:.4f} | {robust['reference']['accuracy']:.4f} |",
                  f"| Nested candidate selection | {robust['nested']['macro_f1']:.4f} | {robust['nested_typos']['macro_f1']:.4f} | {robust['nested']['accuracy']:.4f} |",
                  "", f"The approximate paired family-bootstrap 95% interval for the **clean F1 difference** is **[{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]** ({interval['repetitions']} resamples). This is wide and includes zero.",
                  f"Promotion criteria: {robust['adoption_rule']}. Passed: **{robust['adopt']}**.",
                  f"The optional `robust` artifact uses **{robust['deployment_variant']}**, selected with a separate grouped inner search and refitted on all {robust['deployment_training_rows']} development messages. Its training-set performance is not the reported nested score.",
                  ("The candidate passed the fixed promotion rule." if robust["adopt"] else "The baseline remains the default.")+" The encoder is frozen; only a small L2-regularized classification head learns labels. This reduces capacity but does not prove absence of overfitting.",
                  "This corpus was explored before this experiment, families are heuristic, and several classes have few independent families. Fresh human-labelled English/Hinglish messages are still needed for independent confirmation."]
    if regularized:
        lines += ["", "## Bounded complaint improvement experiment", "",
                  "Three fixed heads use the same training-only TF-IDF features. Selection uses mean clean/noisy validation macro-F1, with a minimum gain of 0.02 before promotion.",
                  "", "| Head | Clean validation macro-F1 | Typo validation macro-F1 | Selection mean |", "|---|---:|---:|---:|"]
        for name,r in regularized["search"].items():
            lines.append(f"| {name} | {r['validation']['macro_f1']:.4f} | {r['validation_typos']['macro_f1']:.4f} | {r['selection_score']:.4f} |")
        lines += ["", f"Reference selection mean: **{regularized['reference_selection_score']:.4f}**. Selected head: **{regularized['selected']}**. Promotion rule passed: **{regularized['adopt']}**.",
                  regularized.get("test_policy","Only the selected candidate was evaluated on the supplied test after passing the validation criterion."),
                  "The validation set has been used in earlier development, so these are development results. Labels and difficult classes are retained; test scores do not select settings."]
    if margin:
        lines += ["", "## Inner-selected complaint margin experiment", "",
                  f"The training partition contains an inner split of {margin['inner_train_rows']} training and {margin['inner_validation_rows']} validation messages. TF-IDF is fitted on inner training only; normalized group overlap is {margin['inner_overlapping_groups']}.",
                  "The candidates are averaged SGD hinge, averaged SGD modified Huber, and balanced linear SVM at C=1. All use hybrid word/character features. Earlier word-only SVM results were weaker; this comparison uses the combined representation.",
                  "", "| Head | Inner clean/noisy mean macro-F1 |", "|---|---:|"]
        for name,r in margin["search"].items():
            lines.append(f"| {name} | {r['selection_score']:.4f} |")
        lines += ["", f"Inner-selected head: **{margin['selected']}**. After refitting on the full original training split, original validation mean macro-F1 was **{margin['selection_score']:.4f}**, versus **{margin['reference_selection_score']:.4f}** for the baseline. Promotion rule passed: **{margin['adopt']}**.",
                  margin.get("test_policy","The selected candidate passed the validation rule before official-test evaluation."),
                  "The original validation corpus was used in earlier experiments. The inner split limits reuse during this search but does not make the existing validation/test corpus fresh independent evidence. Softmax-converted margins are ranking scores, not calibrated probabilities."]
    if semantic_experiments:
        lines += ["", "## Frozen complaint semantics and fixed ensemble", "",
                  "The semantic classifier uses fixed pretrained multilingual representations. C=0.01 or 0.1 is selected within the training partition, with scalers fitted on inner training only. The selected head is refitted on full training before validation.",
                  "The ensemble uses a predeclared 50/50 average with the baseline. Blend weights are not tuned. Component class columns are explicitly aligned.",
                  "", "| Candidate | Clean validation macro-F1 | Typo validation macro-F1 | Mean | Promotion passed |", "|---|---:|---:|---:|---|"]
        for name,r in semantic_experiments.items():
            lines.append(f"| {name} | {r['validation']['macro_f1']:.4f} | {r['validation_typos']['macro_f1']:.4f} | {r['selection_score']:.4f} | {r['adopt']} |")
        for name,r in semantic_experiments.items():
            lines += ["",f"{name}: "+r.get("test_policy","Validation criterion passed before test evaluation.")]
        lines += ["", "All of these runs use an already explored development corpus. The bounded search and frozen encoder reduce some overfitting risks; fresh independent labels are still needed to confirm transfer."]
    if continued:
        lines += ["", "## Bounded encoder continuation", "",
                  "The one-epoch checkpoint is continued for at most two further epochs, with learning rate 2e-5, weight decay 0.01, effective batch 32 and a 192-token limit. Training uses square-root inverse-frequency loss weights and typo augmentation.",
                  "Mean clean/noisy validation macro-F1 is checked after each epoch. A gain below 0.005 stops training; the best qualifying checkpoint is retained. The optimizer and scheduler restart for this continuation.",
                  "", "| Total epoch | Training loss | Clean/noisy validation mean F1 | Checkpoint improved |", "|---|---:|---:|---|"]
        for row in continued["history"]:
            lines.append(f"| {row['total_epoch']} | {row['training_loss']:.4f} | {row['selection_score']:.4f} | {row['checkpoint_improved']} |")
        lines += ["", f"Selected total epoch: **{continued['selected_total_epoch']}**. Validation mean: **{continued['selection_score']:.4f}**, versus baseline **{continued['reference_selection_score']:.4f}**. Promotion rule passed: **{continued['adopt']}**.",
                  continued.get("test_policy","Only the selected checkpoint was evaluated on test after passing the validation rule."),
                  "The original validation set has been reused; this bounded budget and stopping rule reduce some risks but do not establish absence of overfitting or independent generalisation."]
    if improved:
        lines += ["", "## Training-internal weighting and NB-ratio comparison", "",
                  "A predeclared 70/15/15 split of the training partition separates fitting, candidate selection and confirmation. The selected candidate is refitted before each subsequent assessment. Fitted vocabularies and class-specific feature ratios exclude their assessment partitions.",
                  "The candidates compare balanced SGD, normalized square-root class weighting, a fixed lower-regularization variant, and one-versus-rest linear SVMs with binary word/bigram NB log-count ratios.",
                  "", "| Candidate | Selection clean/noisy mean F1 | Selection accuracy |", "|---|---:|---:|"]
        for name,row in improved["search"].items():
            lines.append(f"| {name} | {row['mean']:.4f} | {row['clean']['accuracy']:.4f} |")
        lines += ["",f"Selected candidate: **{improved.get('selected','none')}**. Final status: **{improved['status']}**. Promotion passed: **{improved['adopt']}**."]
        if "confirmation" in improved:
            c = improved["confirmation"]
            lines += [f"Training-internal confirmation mean F1: candidate **{c['candidate']['mean']:.4f}**, reference **{c['reference']['mean']:.4f}**. Passed: **{c['passed']}**."]
        if "validation" in improved:
            lines += [f"Existing-validation mean F1: candidate **{improved['selection_score']:.4f}**, reference **{improved['reference_selection_score']:.4f}**."]
        if "test" in improved:
            interval = improved["test_uncertainty"]
            lines += [f"Official-test macro-F1 **{improved['test']['macro_f1']:.4f}**, typo macro-F1 **{improved['test_typos']['macro_f1']:.4f}**, accuracy **{improved['test']['accuracy']:.4f}**.",
                      f"Paired normalized-template bootstrap 95% interval for clean F1 gain: **[{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]**."]
        else:
            lines += ["No additional official-test evaluation was run because an earlier gate did not qualify the candidate."]
        lines += ["", "The revised criteria were recorded before training in `complaint_improved_plan.json`. Confirmation and existing-validation require a mean F1 gain of 0.01, nonnegative clean F1 gain, and accuracy/typo-F1 declines no larger than 0.005. Release additionally requires positive benchmark F1 gain and a positive paired-bootstrap lower bound. The benchmark remains development-exposed; independent confirmation is still required."]
    if supervised:
        lines += ["", "## Label-supervised word/character network", "",
                  "This experiment learns hashed word/bigram and character embeddings jointly with a nonlinear classifier. It uses a six-epoch cap, dropout 0.4, AdamW weight decay 0.01, inverse-frequency loss weights and training-only typo augmentation.",
                  "Epoch selection uses the fitting/selection partitions from the preceding experiment; a qualified network would require a fresh refit and separate confirmation before existing validation or test evaluation.",
                  "", "| Epoch | Training loss | Selection clean/noisy mean F1 | Selection accuracy |", "|---|---:|---:|---:|"]
        for row in supervised["history"]:
            lines.append(f"| {row['epoch']} | {row['training_loss']:.4f} | {row['assessment']['mean']:.4f} | {row['assessment']['clean']['accuracy']:.4f} |")
        lines += ["",f"Selected epoch: **{supervised['selected_epoch']}**. Final status: **{supervised['status']}**. Promotion passed: **{supervised['adopt']}**."]
        if "selection" in supervised:
            s = supervised["selection"]
            lines += [f"Selected-network mean F1: **{s['candidate']['mean']:.4f}**, compared with reference **{s['reference']['mean']:.4f}** on the same selection partition."]
        if "confirmation" in supervised:
            c = supervised["confirmation"]
            lines += [f"Confirmation mean F1: candidate **{c['candidate']['mean']:.4f}**, reference **{c['reference']['mean']:.4f}**. Passed: **{c['passed']}**."]
        if "validation" in supervised:
            lines += [f"Existing-validation mean F1: **{supervised['selection_score']:.4f}**, reference **{supervised['reference_selection_score']:.4f}**."]
        if "test" in supervised:
            interval = supervised["test_uncertainty"]
            lines += [f"Official-test macro-F1 **{supervised['test']['macro_f1']:.4f}**, typo macro-F1 **{supervised['test_typos']['macro_f1']:.4f}**, accuracy **{supervised['test']['accuracy']:.4f}**.",
                      f"Paired-template bootstrap 95% interval for F1 gain: **[{interval['lower_95']:.4f}, {interval['upper_95']:.4f}]**."]
        else:
            lines += ["No additional test evaluation was performed. Declining training loss without sufficient held-out improvement does not justify deployment."]
    lines += ["", "Default inference uses validation-selected backends:", ""]
    for task, info in champions.items():
        if task == "call_scam" and info["backend"] == "baseline" and family:
            lines.append(f"- {task}: **baseline** (primary family-held-out macro-F1 {family['oof']['macro_f1']:.4f}; legacy exact-template validation selection score {info['score']:.4f}).")
        else:
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
    lines += ["", "## Limitations", "", "- English and Hinglish label vocabularies differ; cross-language transfer is not independently validated.",
              "- Complaint classification has no benign class; call `none` does not prove safety.",
              "- Rule-based PII masking is incomplete; unstructured names, addresses and contextual identifiers may remain.",
              "- Confidence scores and review threshold are uncalibrated; unknown scams can receive high scores.",
              "- Final reports supersede internal-test exploratory artifacts under `superseded/`.",
              "- No RAG layer, live-call integration, or automatic complaint submission is included."]
    (reports / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    readme = ROOT/"README.md"
    readme_text = readme.read_text(encoding="utf-8")
    start_marker, end_marker = "<!-- RESULTS SUMMARY START -->", "<!-- RESULTS SUMMARY END -->"
    if start_marker in readme_text and end_marker in readme_text:
        summary_lines = ["", "Call classification uses template-family evaluation. Complaint classification uses the supplied test set. Scores from these protocols are not directly comparable.", "",
                         "| Task | Backend | Evaluation protocol | Macro-F1 | Typo macro-F1 | Accuracy |",
                         "|---|---|---|---:|---:|---:|"]
        for task,info in champions.items():
            if task == "call_scam" and info["backend"] == "robust":
                clean,noisy,protocol = robust["nested"],robust["nested_typos"],"Nested template-family CV"
            elif task == "call_scam" and family:
                clean,noisy,protocol = family["oof"],family["oof_typos"],"5-fold template-family CV (frozen baseline)"
            elif task == "complaint" and info["backend"] == "adjusted":
                r = adjusted["selected_refit"]
                clean,noisy,protocol = r["test"],r["test_typos"],"Supplied test.csv (official-train protocol)"
            else:
                r = measured[task][info["backend"]]
                clean,noisy,protocol = r["test"],r["test_typos"],"Supplied test.csv"
            summary_lines.append(f"| {task} | {info['backend']} | {protocol} | {clean['macro_f1']:.4f} | {noisy['macro_f1']:.4f} | {clean['accuracy']:.4f} |")
        summary_lines += ["", "[Full comparisons and failure cases](reports/RESULTS.md). Scores are fractions, not percentages.", ""]
        if adjusted and adjusted["adopt"]:
            old, new = adjusted["default_slices"], adjusted["selected_refit"]
            summary_lines += [f"On complaint test messages whose normalized template does not occur in `train.csv`, macro-F1 is **{new['test_novel_templates']['macro_f1']:.4f}** versus **{old['test_novel_templates']['macro_f1']:.4f}** for the previous default and accuracy **{new['test_novel_templates']['accuracy']:.4f}** versus **{old['test_novel_templates']['accuracy']:.4f}**. Most of the full-test macro-F1 gain comes from recurring complaint templates; see the results for the breakdown.", ""]
        if robust and not robust["adopt"]:
            summary_lines += [f"The experimental `robust` call model reached **{robust['nested']['macro_f1']:.4f}** clean / **{robust['nested_typos']['macro_f1']:.4f}** typo macro-F1 in nested family CV. It remains experimental because the paired family-bootstrap interval for clean improvement includes zero. Independent confirmation is required.", ""]
        before, after = readme_text.split(start_marker,1)
        _, after = after.split(end_marker,1)
        readme.write_text(before+start_marker+"\n".join(summary_lines)+end_marker+after,encoding="utf-8")
    guide = ROOT/"TECHNICAL_GUIDE.md"
    guide_text = guide.read_text(encoding="utf-8").split("\n<!-- GENERATED RESULTS -->")[0]
    guide.write_text(guide_text+"\n<!-- GENERATED RESULTS -->\n\n## Evaluation reference\n\n"+
                     "[Evaluation results](reports/RESULTS.md) contain the measured comparisons, class-level metrics and model-promotion decisions.\n", encoding="utf-8")
    print("Updated evaluation reports")


if __name__ == "__main__":
    main()
