"""Download only public pretrained assets, never dataset text; validate LFS SHA-256."""
import hashlib
import json
import shutil
import urllib.request
from .data import ROOT


def main():
    repo = "distilbert/distilbert-base-multilingual-cased"
    output = ROOT / ".cache/pretrained/distilmbert"
    output.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{repo}"
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=30) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    wanted = {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name not in wanted:
            continue
        target = output / name
        expected = item.get("lfs", {}).get("sha256")
        if target.exists() and (not expected or hashlib.sha256(target.read_bytes()).hexdigest() == expected):
            print("Already downloaded:", name, flush=True)
            continue
        temporary = output / (name + ".part")
        digest = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(f"{base}/resolve/{revision}/{name}", timeout=60) as response, temporary.open("wb") as stream:
            while True:
                block = response.read(1024*1024)
                if not block:
                    break
                stream.write(block)
                digest.update(block)
                total += len(block)
                if total % (50*1024*1024) == 0:
                    print(name, total//(1024*1024), "MiB", flush=True)
        if expected and digest.hexdigest() != expected:
            raise RuntimeError("Model checksum mismatch; partial file retained for diagnosis")
        temporary.replace(target)
        print("Verified", name, total, "bytes", flush=True)
    (output/"source_metadata.json").write_text(json.dumps({"repo":repo,"revision":revision},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
