"""Fixed-budget supervised subword experiment with independent internal confirmation."""
import gc
import hashlib
import json
from datetime import datetime, timezone
import random
import time
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .data import ROOT, load
from .improve import validation_rule, fit_pipeline, assess
from .metrics import score, save_report
from .preprocessing import corrupt
from .supervised import CONFIG, HashBagNet, BagDataset, collate, encode_text, probabilities
from .uncertainty import paired_cluster_interval


def train(root=ROOT):
    started = time.perf_counter()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load("complaint",root)
    tr = df[df.split=="train"].reset_index(drop=True)
    labels = np.array(sorted(tr.label.unique()))
    ids = {label:i for i,label in enumerate(labels)}
    y = np.array([ids[label] for label in tr.label])
    fit,rest = train_test_split(np.arange(len(tr)),test_size=.3,stratify=y,random_state=20261021)
    selection,confirmation = train_test_split(rest,test_size=.5,stratify=y[rest],random_state=20261022)
    if tr.group.duplicated().any() or set(fit)&set(selection) or set(fit)&set(confirmation) or set(selection)&set(confirmation):
        raise RuntimeError("Training-internal partitions overlap")
    previous = json.loads((root/"reports/complaint_improved.json").read_text())
    reference = previous["search"]["reference"]
    expected_digest = previous["plan"]["partitions"]["fit"]["group_digest"]
    if hashlib.sha256("".join(sorted(tr.iloc[fit].group)).encode()).hexdigest() != expected_digest:
        raise RuntimeError("The reference uses a different training-internal split")
    plan = {"declared_utc":datetime.now(timezone.utc).isoformat(),"config":CONFIG,
            "training":{"max_selection_epochs":6,"patience":2,"min_delta":.005,"batch_size":128,
                        "learning_rate":.002,"weight_decay":.01,"class_weights":"inverse frequency within fitting partition",
                        "gradient_clip":1.,"augmentation_probability":.3,"augmentation_rate":.12,"seed":20261021},
            "partitions":previous["plan"]["partitions"],"split_seeds":[20261021,20261022],
            "selection_rule":"Mean clean/noisy F1 gain >=0.005, nonnegative clean F1 gain, accuracy/typo F1 drop <=0.005 against same-split reference",
            "confirmation_rule":"Refit fresh model on 85% for fixed selected epoch count; mean F1 gain >=0.01 and same nonregression limits against refitted reference",
            "validation_rule":"Fresh refit on full training for selected epoch count; same >=0.01 mean F1 and nonregression criteria against saved baseline",
            "test_rule":"Only a validation-qualified checkpoint is evaluated; release requires positive F1 gain, positive paired-template bootstrap lower bound and accuracy/typo declines <=0.005",
            "limits":"Previously explored development corpus; hash collisions and mean-pooling limits remain; original benchmark is not new external confirmation",
            "source":"https://arxiv.org/abs/1607.01759"}
    save_report(root/"reports/complaint_supervised_plan.json",plan)
    result = {"plan":plan,"adopt":False,"history":[],"status":"running"}
    notes = root/"local_notes/supervised_experiment.md"
    notes.parent.mkdir(parents=True,exist_ok=True)
    notes.write_text("# Supervised representation experiment\n\nI am testing joint label-supervised word/character representations because the earlier document embeddings were frozen. This model is small and regularized; that does not prove it cannot overfit. Separate selection, confirmation and existing validation gates remain required.\n\n"+json.dumps(plan,indent=2)+"\n",encoding="utf-8")

    def finish(status):
        result["status"],result["seconds"] = status,time.perf_counter()-started
        save_report(root/"reports/complaint_supervised.json",result)
        with notes.open("a",encoding="utf-8") as stream:
            stream.write(f"\nOutcome: {status}; adopted: {result['adopt']}.\n")
        print("Supervised outcome",status,"adopt",result["adopt"],flush=True)

    cache_dir = root/".cache/experiments"
    cache_dir.mkdir(parents=True,exist_ok=True)
    def encode_cached(texts,tag):
        digest = hashlib.sha256((json.dumps(texts)+json.dumps(CONFIG,sort_keys=True)).encode()).hexdigest()
        path = cache_dir/f"hashbags_{digest}.joblib"
        if path.exists():
            return joblib.load(path)
        encoded = []
        for i,text in enumerate(texts):
            encoded.append(encode_text(text))
            if (i+1)%10000 == 0:
                print(tag,"encoded",i+1,"/",len(texts),flush=True)
        joblib.dump(encoded,path)
        return encoded

    clean = encode_cached(tr.text.tolist(),"training")
    augmented = encode_cached([corrupt(t,seed=20261021+i,rate=.12) for i,t in enumerate(tr.text)],"augmentation")
    selection_noisy = encode_cached([corrupt(tr.text.iloc[i],seed=42+j,rate=.25) for j,i in enumerate(selection)],"selection typos")

    def measure(model,part_clean,part_indices,part_noisy,targets):
        a = labels[probabilities(model,part_clean,part_indices,device).argmax(axis=1)]
        b = labels[probabilities(model,part_noisy,np.arange(len(part_noisy)),device).argmax(axis=1)]
        a,b = score(targets,a,labels.tolist()),score(targets,b,labels.tolist())
        return {"clean":a,"typos":b,"mean":(a["macro_f1"]+b["macro_f1"])/2}

    def fit_model(indices,epochs,select_epochs=False):
        torch.manual_seed(20261021)
        np.random.seed(20261021)
        random.seed(20261021)
        model = HashBagNet(len(labels)).to(device)
        counts = np.bincount(y[indices],minlength=len(labels))
        if np.any(counts==0):
            raise RuntimeError("A fitting partition is missing a class")
        weights = torch.tensor(len(indices)/(len(labels)*counts),dtype=torch.float32,device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01,fused=device.type=="cuda")
        loader = DataLoader(BagDataset(clean,y,indices,augmented),batch_size=128,shuffle=True,collate_fn=collate,num_workers=0)
        best,best_epoch,stale,best_state = -1.,0,0,None
        for epoch in range(1,epochs+1):
            model.train()
            total = 0.
            for values,target in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    logits = model(*(value.to(device) for value in values))
                    loss = loss_fn(logits.float(),target.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite supervised training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                total += loss.item()
            print("Supervised fit rows",len(indices),"epoch",epoch,"loss",total/len(loader),flush=True)
            if select_epochs:
                measured = measure(model,clean,selection,selection_noisy,tr.label.iloc[selection])
                result["history"].append({"epoch":epoch,"training_loss":total/len(loader),"assessment":measured})
                print("Selection mean",measured["mean"],"accuracy",measured["clean"]["accuracy"],flush=True)
                if measured["mean"] >= best+.005:
                    best,best_epoch,stale = measured["mean"],epoch,0
                    best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
                else:
                    stale += 1
                if stale >= 2:
                    break
        del optimizer
        if select_epochs:
            model.load_state_dict(best_state)
        return model,best_epoch if select_epochs else epochs

    model,epoch = fit_model(fit,6,select_epochs=True)
    result["selected_epoch"] = epoch
    selected_assessment = result["history"][epoch-1]["assessment"]
    result["selection"] = {"candidate":selected_assessment,"reference":reference,"passed":validation_rule(selected_assessment,reference,gain=.005)}
    del model
    gc.collect()
    if device.type=="cuda":
        torch.cuda.empty_cache()
    if not result["selection"]["passed"]:
        finish("selection_gate_failed")
        return
    combined = np.array(sorted(set(fit)|set(selection)))
    model,_ = fit_model(combined,epoch)
    confirmation_noisy = encode_cached([corrupt(tr.text.iloc[i],seed=42+j,rate=.25) for j,i in enumerate(confirmation)],"confirmation typos")
    new = measure(model,clean,confirmation,confirmation_noisy,tr.label.iloc[confirmation])
    reference_model = fit_pipeline("reference",tr.iloc[combined])
    old = assess(reference_model,tr.iloc[confirmation],labels.tolist())
    result["confirmation"] = {"candidate":new,"reference":old,"passed":validation_rule(new,old)}
    print("Supervised confirmation means",new["mean"],old["mean"],"passed",result["confirmation"]["passed"],flush=True)
    del model,reference_model
    gc.collect()
    if device.type=="cuda":
        torch.cuda.empty_cache()
    if not result["confirmation"]["passed"]:
        finish("confirmation_gate_failed")
        return
    model,_ = fit_model(np.arange(len(tr)),epoch)
    va = df[df.split=="validation"]
    vc = encode_cached(va.text.tolist(),"validation")
    vn = encode_cached([corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(va.text)],"validation typos")
    new = measure(model,vc,np.arange(len(va)),vn,va.label)
    baseline = json.loads((root/"reports/complaint_baseline.json").read_text())
    baseline = baseline[baseline["selected"]]
    old = {"clean":baseline["validation"],"typos":baseline["validation_typos"],"mean":(baseline["validation"]["macro_f1"]+baseline["validation_typos"]["macro_f1"])/2}
    result.update({"validation":new["clean"],"validation_typos":new["typos"],"selection_score":new["mean"],"reference_selection_score":old["mean"],"validation_passed":validation_rule(new,old)})
    output = root/"models/complaint_supervised"
    output.mkdir(parents=True,exist_ok=True)
    torch.save({key:value.detach().cpu() for key,value in model.state_dict().items()},output/"weights.pt")
    save_report(output/"metadata.json",{"config":CONFIG,"labels":labels.tolist(),"epochs":epoch})
    print("Supervised validation means",new["mean"],old["mean"],"passed",result["validation_passed"],flush=True)
    if not result["validation_passed"]:
        finish("validation_gate_failed")
        return
    te = df[df.split=="test"]
    tc = encode_cached(te.text.tolist(),"test")
    tn = encode_cached([corrupt(t,seed=42+i,rate=.25) for i,t in enumerate(te.text)],"test typos")
    predicted = labels[probabilities(model,tc,np.arange(len(te)),device).argmax(axis=1)]
    test_labels = sorted(set(labels)|set(te.label))
    result["test"] = score(te.label,predicted,test_labels)
    result["test_typos"] = score(te.label,labels[probabilities(model,tn,np.arange(len(te)),device).argmax(axis=1)],test_labels)
    reference_model = joblib.load(root/"models/complaint_baseline.joblib")["model"]
    interval = paired_cluster_interval(te.label,predicted,reference_model.predict(te.text.tolist()),te.group,test_labels)
    result["test_uncertainty"] = interval
    result["adopt"] = bool(result["test"]["macro_f1"]>baseline["test"]["macro_f1"]
                           and result["test"]["accuracy"]>=baseline["test"]["accuracy"]-.005
                           and result["test_typos"]["macro_f1"]>=baseline["test_typos"]["macro_f1"]-.005
                           and interval["lower_95"]>0)
    finish("test_confirmed" if result["adopt"] else "test_confirmation_failed")


if __name__=="__main__":
    train()
