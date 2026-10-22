"""Fine-tuned multilingual encoder fused with the sparse `adjusted` model (complaint task).

Plan fixed before running: xlm-roberta-base, at most three epochs, epoch and fusion weight chosen
on validation only, then one official-test evaluation. The encoder is trained on the training split
of the official-train protocol; the sparse member used for validation is a training-only fit.
Added before training: the fusion weight and a class-prior correction on the fused probabilities
are chosen jointly on validation.
"""
import argparse
import itertools
import json
import math
import random
import time
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from .adjusted import accuracy_interval, fit as fit_sparse, test_metrics
from .data import ROOT, load
from .metrics import save_report, score
from .preprocessing import corrupt
from .transformer import Encoded
from .uncertainty import paired_cluster_interval

ENCODER = "xlm-roberta-base"
WEIGHTS = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)
PRIOR_TAUS = (0.0, 0.25, 0.5)
MAX_LENGTH, MICROBATCH, ACCUMULATION, EPOCHS, LR = 256, 16, 2, 3, 2e-5
NOVEL_GAIN = 0.01


class FusionClassifier:
    """softmax(log(w * encoder + (1 - w) * sparse) + tau * log prior) over a shared label order."""

    def __init__(self, folder, sparse, weight, labels, device=None, tau=0., prior=None, max_length=MAX_LENGTH):
        self.folder, self.sparse, self.weight, self.classes_ = str(folder), sparse, float(weight), np.array(labels)
        self.tau, self.prior = float(tau), None if prior is None else np.asarray(prior, float)
        self.device, self.max_length = device, max_length
        self._encoder = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_encoder"] = None
        return state

    def encoder(self):
        if self._encoder is None:
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(self.folder)
            model = AutoModelForSequenceClassification.from_pretrained(self.folder).to(device).eval()
            self._encoder = tokenizer, model, device
        return self._encoder

    def encoder_proba(self, texts, batch=32):
        tokenizer, model, device = self.encoder()
        out = []
        with torch.inference_mode():
            for offset in range(0, len(texts), batch):
                inputs = tokenizer(list(texts[offset:offset+batch]), padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(**inputs).logits.float()
                out.append(torch.softmax(logits, -1).cpu().numpy())
        return np.concatenate(out)

    def combine(self, encoder_probs, sparse_probs):
        return prior_adjust(self.weight*encoder_probs + (1-self.weight)*sparse_probs, self.prior, self.tau)

    def predict_proba(self, texts):
        texts = list(texts)
        sparse = self.sparse.predict_proba(texts) if self.weight < 1 else 0.
        encoder = self.encoder_proba(texts) if self.weight > 0 else 0.
        return self.combine(encoder, sparse)

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def prior_adjust(probs, prior, tau):
    if not tau:
        return probs
    logits = np.log(np.clip(probs, 1e-12, None)) + tau*np.log(prior)
    logits -= logits.max(axis=1, keepdims=True)
    adjusted = np.exp(logits)
    return adjusted/adjusted.sum(axis=1, keepdims=True)


def train(root=ROOT, smoke=False):
    start = time.perf_counter()
    torch.manual_seed(42), np.random.seed(42), random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda" and not smoke:
        raise RuntimeError("Encoder fine-tuning expects a CUDA GPU; use --smoke for a CPU code check")
    df = load("complaint", root, protocol="official")
    if smoke:
        df = df.sample(frac=1, random_state=0).groupby("split").head(64)
        df = df[df.label.isin(set(df[df.split == "train"].label))]
    tr, va = df[df.split == "train"], df[df.split == "validation"]
    labels = sorted(set(tr.label))
    index = {label: i for i, label in enumerate(labels)}
    va = va[va.label.isin(index)]
    output = root/"models/complaint_fusion_encoder"
    output.mkdir(parents=True, exist_ok=True)
    name = "hf-internal-testing/tiny-xlm-roberta" if smoke else ENCODER
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=index).to(device)
    # Smoke test models can have fewer positions than MAX_LENGTH; xlm-roberta-base has 514.
    max_length = min(MAX_LENGTH, model.config.max_position_embeddings-2)
    if not smoke:
        model.gradient_checkpointing_enable()
    rng = random.Random(42)
    texts = [corrupt(t, seed=2026+i, rate=.12) if rng.random() < .3 else t for i, t in enumerate(tr.text)]
    dataset = Encoded(tokenizer(texts, truncation=True, max_length=max_length), [index[x] for x in tr.label])
    loader = DataLoader(dataset, batch_size=MICROBATCH, shuffle=True, num_workers=0,
                        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
    counts = tr.label.value_counts()
    prior = np.array([counts[x] for x in labels], float)/len(tr)
    class_weights = torch.tensor([math.sqrt(len(tr)/(len(labels)*counts[x])) for x in labels], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.01)
    updates = EPOCHS*math.ceil(len(loader)/ACCUMULATION)
    warmup = max(1, int(.06*updates))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: s/warmup if s < warmup else max(0., (updates-s)/max(1, updates-warmup)))
    member = FusionClassifier(output, None, 1.0, labels, device, max_length=max_length)
    member._encoder = tokenizer, model, device
    va_typos = [corrupt(t, seed=42+i, rate=.25) for i, t in enumerate(va.text)]
    history, best = [], None
    for epoch in range(1, EPOCHS+1):
        tick, total = time.perf_counter(), 0.
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            y = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = loss_fn(model(**inputs).logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            (loss/ACCUMULATION).backward()
            total += loss.item()
            if (step+1) % ACCUMULATION == 0 or step+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(), scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step+1) % 400 == 0:
                print("epoch", epoch, "batch", step+1, "/", len(loader), "loss", round(total/(step+1), 4), flush=True)
        model.eval()
        probs = member.encoder_proba(va.text.tolist()), member.encoder_proba(va_typos)
        clean, noisy = (score(va.label, np.array(labels)[p.argmax(1)], labels) for p in probs)
        row = {"epoch": epoch, "training_loss": total/len(loader), "clean_macro_f1": clean["macro_f1"],
               "typo_macro_f1": noisy["macro_f1"], "clean_accuracy": clean["accuracy"],
               "selection": (clean["macro_f1"]+noisy["macro_f1"])/2, "seconds": time.perf_counter()-tick}
        history.append(row)
        print(row, flush=True)
        if best is None or row["selection"] > best["selection"]:
            best = row
            best_probs = probs
            model.save_pretrained(output, safe_serialization=True)
            tokenizer.save_pretrained(output)
    del optimizer, scheduler
    member._encoder = None
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    # Sparse member for validation: same configuration as `adjusted`, fitted on the training split only.
    deployed = joblib.load(root/"models/complaint_adjusted.joblib")["model"]
    sparse = fit_sparse(tr.text, tr.label, deployed.pipeline.named_steps["classifier"].alpha, deployed.tau)
    if list(sparse.classes_) != labels:
        raise RuntimeError("Sparse and encoder label orders differ")
    sparse_probs = sparse.predict_proba(va.text.tolist()), sparse.predict_proba(va_typos)
    grid = []
    for w, tau in itertools.product(WEIGHTS, PRIOR_TAUS):
        fused = [prior_adjust(w*e + (1-w)*s, prior, tau) for e, s in zip(best_probs, sparse_probs)]
        clean, noisy = (score(va.label, np.array(labels)[p.argmax(1)], labels) for p in fused)
        grid.append({"weight": w, "prior_tau": tau, "clean_macro_f1": clean["macro_f1"], "typo_macro_f1": noisy["macro_f1"],
                     "clean_accuracy": clean["accuracy"], "selection": (clean["macro_f1"]+noisy["macro_f1"])/2})
        print(grid[-1], flush=True)
    chosen = max(grid, key=lambda r: r["selection"])
    report = {"encoder": name, "history": history, "selected_epoch": best["epoch"], "grid": grid,
              "selected_weight": chosen["weight"], "selected_prior_tau": chosen["prior_tau"],
              "selected_validation": chosen,
              "config": {"max_length": max_length, "microbatch": MICROBATCH, "accumulation": ACCUMULATION,
                         "epochs": EPOCHS, "learning_rate": LR, "warmup": .06, "weight_decay": .01,
                         "class_weights": "square root inverse frequency", "augmentation": "p=0.3, rate=0.12", "seed": 42,
                         "weight_grid": list(WEIGHTS), "prior_tau_grid": list(PRIOR_TAUS),
                         "prior_correction": "softmax(log fused + tau * log training-split prior), (w, tau) chosen jointly on validation"},
              "adoption_rule": f"Novel-template macro-F1 >= adjusted + {NOVEL_GAIN}; full-test macro-F1 and accuracy >= adjusted"}
    # Single test evaluation of the selected configuration, against the deployed `adjusted` model.
    if smoke:
        deployed = sparse
    if list(deployed.classes_) != labels:
        raise RuntimeError("Deployed sparse model label order differs from the encoder")
    final = FusionClassifier(output, deployed, chosen["weight"], labels, tau=chosen["prior_tau"], prior=prior,
                             max_length=max_length)
    cache = {}
    def predict(texts):
        key = hash(tuple(texts))
        if key not in cache:
            cache[key] = final.predict(texts)
        return cache[key]
    report["test_results"] = test_metrics(predict, df, labels)
    report["adjusted_results"] = test_metrics(deployed.predict, df, labels)
    test = df[df.split == "test"].reset_index(drop=True)
    new, old = predict(test.text.tolist()), deployed.predict(test.text.tolist())
    split_labels = sorted(set(labels) | set(test.label))
    report["uncertainty_vs_adjusted"] = {}
    for key, mask in (("test", np.ones(len(test), bool)), ("test_novel_templates", ~test.seen_template.astype(bool).to_numpy())):
        y, groups = test.label.to_numpy()[mask], test.group.to_numpy()[mask]
        interval = paired_cluster_interval(y, new[mask], old[mask], groups, split_labels)
        interval.update(accuracy_interval(y, new[mask], old[mask], groups))
        report["uncertainty_vs_adjusted"][key] = interval
    n, a = report["test_results"], report["adjusted_results"]
    report["adopt"] = bool(n["test_novel_templates"]["macro_f1"] >= a["test_novel_templates"]["macro_f1"]+NOVEL_GAIN
                           and n["test"]["macro_f1"] >= a["test"]["macro_f1"] and n["test"]["accuracy"] >= a["test"]["accuracy"])
    for key in ("test", "test_typos", "test_novel_templates"):
        print(key, "fusion", round(n[key]["macro_f1"], 4), round(n[key]["accuracy"], 4),
              "adjusted", round(a[key]["macro_f1"], 4), round(a[key]["accuracy"], 4), flush=True)
    print("adopt", report["adopt"], flush=True)
    report["seconds"] = time.perf_counter()-start
    if not smoke:
        joblib.dump({"model": final, "task": "complaint", "mode": f"fusion-{name}", "labels": labels,
                     "prior_tau": chosen["prior_tau"], "selection_score": chosen["selection"]}, root/"models/complaint_fusion.joblib", compress=3)
    save_report(root/("reports/complaint_fusion_smoke.json" if smoke else "reports/complaint_fusion.json"), report)
    return report


if __name__ == "__main__":
    from cyberlens.fusion import train as canonical_train
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU run that only checks the code path")
    canonical_train(smoke=parser.parse_args().smoke)
