"""Download only MiniLM inference files from its pinned official HF revision."""
import os
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import snapshot_download
from evrag.config import ROOT

if __name__ == '__main__':
    path = snapshot_download(
        repo_id='sentence-transformers/all-MiniLM-L6-v2',
        revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
        local_dir=ROOT/'models/all-MiniLM-L6-v2',
        allow_patterns=['*.json', 'vocab.txt', 'model.safetensors'],
        max_workers=2,
    )
    print(path)
