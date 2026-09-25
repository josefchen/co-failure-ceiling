"""Download the released data at a PINNED Hugging Face revision into ./runs (and ./gpqa_open_v2).

The dataset is josefchen/co-failure-67-models. REVISION is the exact commit the paper's numbers were computed
from; later commits cannot change what this script fetches. Usage: python3 scripts/fetch_data.py [--responses]
(--responses also downloads the raw model responses, ~25 MB compressed, needed only to re-grade cells).
"""
import os, sys, shutil
from huggingface_hub import snapshot_download

REPO = "josefchen/co-failure-67-models"
REVISION = "f2dd304420cfe48bf8e5c10dcb6735ec3dd5252c"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

if __name__ == "__main__":
    patterns = ["runs/*", "gpqa_open_v2/*"] + (["responses/*"] if "--responses" in sys.argv else [])
    path = snapshot_download(REPO, repo_type="dataset", revision=REVISION, allow_patterns=patterns)
    for sub in ("runs", "gpqa_open_v2", "responses"):
        src = os.path.join(path, sub)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(ROOT, sub); os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        print(f"[fetch] {sub}/: {len(os.listdir(src))} files from {REPO}@{REVISION[:12]}")
