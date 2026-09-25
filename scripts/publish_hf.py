"""Publish release/hf/ (built by scripts/stage_release.py) as one commit to the HF dataset, then pin that commit in
scripts/fetch_data.py so the reproduction always downloads exactly these files (maintainers only; needs an HF login).
The first release stays available at the tag v1-2026-06. Usage: python3 scripts/publish_hf.py "<commit message>"
"""
import os, re, sys
from huggingface_hub import HfApi

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REPO = "josefchen/co-failure-67-models"

if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "v2 release"
    api = HfApi()
    info = api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=os.path.join(ROOT, "release", "hf"),
                             commit_message=msg,
                             delete_patterns=["*.json", "*.csv", "*.md", "*.gz"])   # remote files not in this release
    sha = info.oid
    p = os.path.join(ROOT, "scripts", "fetch_data.py")
    s = open(p).read()
    s = re.sub(r'REVISION = "[^"]*"', f'REVISION = "{sha}"', s)
    open(p, "w").write(s)
    print(f"[hf] {REPO}@{sha}\n[hf] pinned in scripts/fetch_data.py")
