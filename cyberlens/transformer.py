"""Local multilingual DistilBERT fine-tuning, with an 8 GB VRAM profile."""
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from .data import ROOT, TASKS, load
from .metrics import evaluate, save_report
from .preprocessing import corrupt

MODEL_ID = "distilbert/distilbert-base-multilingual-cased"


class Encoded(Dataset):
    def __init__(self, encoding, labels=None):
        self.encoding, self.labels = encoding, labels

    def __len__(self):
        return len(self.encoding["input_ids"])

    def __getitem__(self, i):
        item = {key: values[i] for key, values in self.encoding.items()}
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


class TransformerPredictor:
    def __init__(self, path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(self.device).eval()
        self.classes_ = np.array([self.model.config.id2label[i] for i in range(self.model.config.num_labels)])
        self.max_length = 192

    @torch.inference_mode()
    def predict_proba(self, texts):
        probabilities = []
        for start in range(0, len(texts), 24):
            inputs = self.tokenizer(texts[start:start+24], truncation=True, padding=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                logits = self.model(**inputs).logits
            probabilities.append(logits.float().softmax(-1).cpu().numpy())
        return np.concatenate(probabilities)

    def predict(self, texts):
        return self.classes_[self.predict_proba(list(texts)).argmax(-1)]


def train(tasks=TASKS, root=ROOT, epochs=1, batch_size=8, accumulation=4, max_length=192):
    if min(epochs, batch_size, accumulation, max_length) < 1:
        raise ValueError("Training sizes and epochs must be positive")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full transformer training expects CUDA; use the CPU sparse baseline on this machine instead.")
    for task in tasks:
        start = time.perf_counter()
        df = load(task, root)
        tr = df[df.split == "train"]
        labels = sorted(tr.label.unique())
        label_to_id = {label: i for i, label in enumerate(labels)}
        local = root/".cache/pretrained/distilmbert"
        source = str(local) if (local/"model.safetensors").exists() else MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=str(root/".cache/huggingface"), local_files_only=source != MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(source,
            cache_dir=str(root/".cache/huggingface"), local_files_only=source != MODEL_ID,
            use_safetensors=True, num_labels=len(labels),
            id2label=dict(enumerate(labels)), label2id=label_to_id).to(device)
        model.gradient_checkpointing_enable()
        rng = random.Random(42)
        texts = [corrupt(t, seed=i+42, rate=.12) if rng.random() < .3 else t for i,t in enumerate(tr.text)]
        encoding = tokenizer(texts, truncation=True, max_length=max_length)
        dataset = Encoded(encoding, [label_to_id[x] for x in tr.label])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                            collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
        counts = tr.label.value_counts()
        weights = torch.tensor([math.sqrt(len(tr)/(len(labels)*counts[x])) for x in labels], device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
        updates = epochs*math.ceil(len(loader)/accumulation)
        warmup = max(1, int(.1*updates))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
            step/warmup if step < warmup else max(0., (updates-step)/max(1,updates-warmup)))
        history = []
        output = root / "models" / f"{task}_transformer"
        output.mkdir(parents=True, exist_ok=True)
        print(f"Fine-tuning {task}: {len(tr)} examples, {epochs} epochs, batch={batch_size}, accumulation={accumulation}", flush=True)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            for step, batch in enumerate(loader):
                y = batch.pop("labels").to(device)
                inputs = {k:v.to(device) for k,v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**inputs).logits
                    loss = loss_fn(logits.float(), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite transformer loss")
                # Correctly scale the final accumulation group, which may be shorter.
                remaining_group = min(accumulation, len(loader)-(step//accumulation)*accumulation)
                (loss/remaining_group).backward()
                total_loss += loss.item()
                if (step+1)%accumulation == 0 or step+1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if (step+1)%200 == 0:
                    print(f"{task} epoch {epoch+1}, batch {step+1}/{len(loader)}, mean loss={total_loss/(step+1):.4f}", flush=True)
            history.append(total_loss/len(loader))
            print(f"Completed epoch {epoch+1}: loss={history[-1]:.4f}", flush=True)
        # Fixed epoch budget; no test-based checkpoint choice.
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        metadata = {"model_id": MODEL_ID, "revision": getattr(model.config,"_commit_hash",None),
                    "epochs": epochs, "batch_size":batch_size, "gradient_accumulation":accumulation,
                    "max_length":max_length,"precision":"bfloat16","gradient_checkpointing":True,
                    "training_rows":len(tr),"loss":history,"typo_augmentation_fraction":.3,
                    "class_weights":"square root inverse frequency", "seed":42}
        if (local/"source_metadata.json").exists():
            metadata["revision"] = json.loads((local/"source_metadata.json").read_text())["revision"]
        (output/"training_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        del optimizer, scheduler, model
        torch.cuda.empty_cache()
        predictor = TransformerPredictor(output)
        results = evaluate(predictor.predict, df, sorted(df.label.unique()))
        results.update({"config":metadata,"seconds":time.perf_counter()-start,
                        "cuda_peak_mb":torch.cuda.max_memory_allocated()/1024**2})
        save_report(root/"reports"/f"{task}_transformer.json",results)
        print("Transformer test macro F1:", results["test"]["macro_f1"], flush=True)
        del predictor
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
