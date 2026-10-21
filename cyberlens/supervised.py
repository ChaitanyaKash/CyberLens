"""Supervised hashed word/character bags with a small nonlinear classifier.

Embedding-bag classification is inspired by https://arxiv.org/abs/1607.01759.
Separate character features and the nonlinear head are project-specific choices.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .preprocessing import tokens


CONFIG = {"word_buckets":65536,"char_buckets":32768,"dimensions":96,"hidden":128,
          "dropout":.4,"max_tokens":256,"max_char_grams":12,"feature_version":1}


@lru_cache(maxsize=400000)
def feature_id(value,buckets):
    return 1+int.from_bytes(hashlib.blake2b(value.encode(),digest_size=8,person=b"CyberLens").digest(),"little")%(buckets-1)


@lru_cache(maxsize=200000)
def char_ids(word):
    word = "^"+word[:64]+"$"
    grams = [word[i:i+n] for n in (3,4,5) for i in range(max(0,len(word)-n+1))]
    if len(grams)>CONFIG["max_char_grams"]:
        grams = [grams[i] for i in np.linspace(0,len(grams)-1,CONFIG["max_char_grams"],dtype=int)]
    return tuple(feature_id("c:"+gram,CONFIG["char_buckets"]) for gram in grams)


def encode_text(text):
    words = tokens(text)[:CONFIG["max_tokens"]]
    w = [feature_id("w:"+word,CONFIG["word_buckets"]) for word in words]
    w += [feature_id("b:"+a+" "+b,CONFIG["word_buckets"]) for a,b in zip(words,words[1:])]
    c = [value for word in words for value in char_ids(word)]
    return np.array(w or [0],dtype=np.int32),np.array(c or [0],dtype=np.int32)


def collate(rows):
    encoded,y = zip(*rows)
    values = []
    for channel in (0,1):
        arrays = [row[channel] for row in encoded]
        offsets = np.concatenate(([0],np.cumsum([len(a) for a in arrays])))
        values.extend((torch.from_numpy(np.concatenate(arrays).astype(np.int64)),torch.from_numpy(offsets)))
    return values,torch.tensor(y,dtype=torch.long)


class BagDataset(Dataset):
    def __init__(self,clean,labels,indices,augmented=None):
        self.clean,self.labels,self.indices,self.augmented = clean,labels,np.asarray(indices),augmented

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,i):
        j = self.indices[i]
        features = self.augmented[j] if self.augmented is not None and random.random()<.3 else self.clean[j]
        return features,int(self.labels[j])


class HashBagNet(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.word = nn.EmbeddingBag(CONFIG["word_buckets"],CONFIG["dimensions"],mode="mean",include_last_offset=True)
        self.char = nn.EmbeddingBag(CONFIG["char_buckets"],CONFIG["dimensions"],mode="mean",include_last_offset=True)
        nn.init.normal_(self.word.weight,std=.1)
        nn.init.normal_(self.char.weight,std=.1)
        self.head = nn.Sequential(nn.LayerNorm(2*CONFIG["dimensions"]),nn.Dropout(CONFIG["dropout"]),
                                  nn.Linear(2*CONFIG["dimensions"],CONFIG["hidden"]),nn.GELU(),
                                  nn.Dropout(CONFIG["dropout"]),nn.Linear(CONFIG["hidden"],classes))

    def forward(self,words,word_offsets,chars,char_offsets):
        return self.head(torch.cat((self.word(words,word_offsets),self.char(chars,char_offsets)),dim=1))


@torch.inference_mode()
def probabilities(model,encoded,indices,device):
    model.eval()
    loader = DataLoader(BagDataset(encoded,np.zeros(len(encoded)),indices),batch_size=256,collate_fn=collate,num_workers=0)
    result = []
    for values,_ in loader:
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            logits = model(*(value.to(device) for value in values))
        result.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(result)


class SupervisedPredictor:
    def __init__(self,path,device=None):
        path = Path(path)
        metadata = json.loads((path/"metadata.json").read_text())
        if metadata["config"] != CONFIG:
            raise ValueError("Saved feature configuration differs from the current implementation")
        self.classes_ = np.array(metadata["labels"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = HashBagNet(len(self.classes_)).to(self.device)
        self.model.load_state_dict(torch.load(path/"weights.pt",map_location=self.device,weights_only=True))
        self.model.eval()

    def predict_proba(self,texts):
        encoded = [encode_text(text) for text in texts]
        return probabilities(self.model,encoded,np.arange(len(encoded)),self.device)

    def predict(self,texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]
