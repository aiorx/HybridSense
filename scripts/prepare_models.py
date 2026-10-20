"""Download the exact pretrained encoder used in the original training scripts."""
import sys
import json
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.settings import EMBEDDING_PATH
from huggingface_hub import snapshot_download, HfApi

if __name__ == "__main__":
    repo = "huggingface/CodeBERTa-small-v1"
    revision = HfApi().model_info(repo).sha
    snapshot_download(repo, revision=revision, local_dir=str(EMBEDDING_PATH),
        allow_patterns=["config.json", "pytorch_model.bin", "model.safetensors", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json"])
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in EMBEDDING_PATH.iterdir() if p.is_file() and p.name != "hybridsense-manifest.json"}
    (EMBEDDING_PATH / "hybridsense-manifest.json").write_text(json.dumps({"repo":repo,"revision":revision,"sha256":files},indent=2),encoding="utf-8")
    print(f"CodeBERTa ready: {EMBEDDING_PATH}")
