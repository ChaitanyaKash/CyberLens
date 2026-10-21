"""Bounded continuation of the one-epoch complaint encoder; validation stopping."""
import json
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from .data import ROOT, load
from .metrics import score, evaluate, save_report
from .preprocessing import corrupt
from .transformer import Encoded, TransformerPredictor


def train(root=ROOT):
    start = time.perf_counter()
    torch.manual_seed(2026)
    np.random.seed(2026)
    random.seed(2026)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded continuation expects the local CUDA GPU")
    df = load("complaint",root)
    tr,va = df[df.split == "train"],df[df.split == "validation"]
    labels = sorted(tr.label.unique())
    source = root/"models/complaint_transformer"
    output = root/"models/complaint_continued"
    output.mkdir(parents=True,exist_ok=True)
    previous = json.loads((root/"reports/complaint_transformer.json").read_text())
    if previous["config"]["epochs"] != 1 or previous["config"]["training_rows"] != len(tr):
        raise RuntimeError("Continuation expects the documented one-epoch training artifact")
    tokenizer = AutoTokenizer.from_pretrained(source,local_files_only=True,fix_mistral_regex=False)
    model = AutoModelForSequenceClassification.from_pretrained(source,local_files_only=True).to("cuda")
    actual = [model.config.id2label[i] for i in range(len(labels))]
    if actual != labels:
        raise RuntimeError("Checkpoint class order does not match the prepared training labels")
    model.gradient_checkpointing_enable()
    label_ids = {label:i for i,label in enumerate(labels)}
    rng = random.Random(2026)
    texts = [corrupt(t,seed=2026+i,rate=.12) if rng.random()<.3 else t for i,t in enumerate(tr.text)]
    dataset = Encoded(tokenizer(texts,truncation=True,max_length=192),[label_ids[x] for x in tr.label])
    loader = DataLoader(dataset,batch_size=16,shuffle=True,num_workers=0,
                        collate_fn=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8))
    counts = tr.label.value_counts()
    weights = torch.tensor([math.sqrt(len(tr)/(len(labels)*counts[x])) for x in labels],device="cuda")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    updates = 2*math.ceil(len(loader)/2)
    warmup = max(1,int(.1*updates))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:step/warmup if step<warmup else max(0.,(updates-step)/max(1,updates-warmup)))
    best_clean,best_noisy = previous["validation"],previous["validation_typos"]
    best = (best_clean["macro_f1"]+best_noisy["macro_f1"])/2
    initial_score = best
    selected_epoch = 1
    model.save_pretrained(output,safe_serialization=True)
    tokenizer.save_pretrained(output)
    history = []

    def predict(values):
        model.eval()
        predictions = []
        with torch.inference_mode():
            for offset in range(0,len(values),24):
                inputs = tokenizer(list(values[offset:offset+24]),padding=True,truncation=True,max_length=192,return_tensors="pt").to("cuda")
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    logits = model(**inputs).logits
                predictions.extend(labels[i] for i in logits.argmax(-1).cpu().tolist())
        return predictions

    validation_typos = [corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(va.text)]
    for added_epoch in range(1,3):
        tick = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for step,batch in enumerate(loader):
            y = batch.pop("labels").to("cuda")
            inputs = {key:value.to("cuda") for key,value in batch.items()}
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                loss = loss_fn(model(**inputs).logits.float(),y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite continuation loss")
            group_size = min(2,len(loader)-(step//2)*2)
            (loss/group_size).backward()
            total += loss.item()
            if (step+1)%2 == 0 or step+1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step+1)%400 == 0:
                print("Total epoch",added_epoch+1,"batch",step+1,"/",len(loader),"mean loss",total/(step+1),flush=True)
        clean = score(va.label,predict(va.text.tolist()),labels)
        noisy = score(va.label,predict(validation_typos),labels)
        selection = (clean["macro_f1"]+noisy["macro_f1"])/2
        improved = selection >= best+.005
        history.append({"total_epoch":added_epoch+1,"training_loss":total/len(loader),"validation":clean,
                        "validation_typos":noisy,"selection_score":selection,"checkpoint_improved":improved,
                        "seconds":time.perf_counter()-tick})
        print("Total epoch",added_epoch+1,"validation mean",selection,"checkpoint improved",improved,flush=True)
        if not improved:
            print("Stopping: validation gain below fixed 0.005 minimum",flush=True)
            break
        best,best_clean,best_noisy,selected_epoch = selection,clean,noisy,added_epoch+1
        model.save_pretrained(output,safe_serialization=True)
    baseline = json.loads((root/"reports/complaint_baseline.json").read_text())
    baseline = baseline[baseline["selected"]]
    reference = (baseline["validation"]["macro_f1"]+baseline["validation_typos"]["macro_f1"])/2
    accepted = best >= reference+.02
    result = {"history":history,"selected_total_epoch":selected_epoch,"initial_validation_mean":initial_score,
              "validation":best_clean,"validation_typos":best_noisy,"selection_score":best,
              "reference_selection_score":reference,"adopt":bool(accepted),
              "protocol":"Continue existing one-epoch checkpoint for at most two additional epochs; stop on validation mean gain <0.005; keep best qualifying checkpoint",
              "adoption_rule":"Best clean/noisy validation mean macro-F1 exceeds sparse baseline by >=0.02",
              "config":{"learning_rate":2e-5,"batch_size":16,"accumulation":2,"max_length":192,
                        "class_weights":"square root inverse frequency","weight_decay":.01,"seed":2026},
              "limits":"Previously examined validation; early stopping limits but cannot eliminate overfitting. Test never selects checkpoints. Continuation resets optimizer/scheduler."}
    (output/"training_metadata.json").write_text(json.dumps({k:v for k,v in result.items() if k not in ("history","validation","validation_typos")},indent=2),encoding="utf-8")
    del optimizer,scheduler,model
    torch.cuda.empty_cache()
    if accepted:
        print("Validation criterion passed; evaluating selected checkpoint on test",flush=True)
        predictor = TransformerPredictor(output)
        result.update(evaluate(predictor.predict,df,sorted(df.label.unique())))
    else:
        result["test_policy"] = "Not evaluated: bounded continuation did not pass validation promotion rule."
    result["seconds"] = time.perf_counter()-start
    save_report(root/"reports/complaint_continued.json",result)
    print("Continuation selected epoch",selected_epoch,"validation mean",best,"adopt",accepted,flush=True)


if __name__ == "__main__":
    train()
