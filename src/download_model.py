"""Download and verify the fixed Qwen snapshot without loading the model."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import huggingface_hub
from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "Qwen2.5-1.5B"
SOURCE_PATH = PROJECT_ROOT / "docs" / "model_source.json"
FILES = [
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def verify_file(path, remote):
    """Verify LFS SHA-256 or the Git blob hash, and record a local SHA-256."""
    size = path.stat().st_size
    if size != remote.size:
        raise RuntimeError(f"Size mismatch for {path.name}: {size} != {remote.size}")
    sha256 = hashlib.sha256()
    git_hash = None
    if remote.lfs is None:
        git_hash = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            sha256.update(chunk)
            if git_hash is not None:
                git_hash.update(chunk)
    digest = sha256.hexdigest()
    if remote.lfs is not None:
        expected = remote.lfs.sha256
        actual = digest
        algorithm = "sha256"
    else:
        expected = remote.blob_id
        actual = git_hash.hexdigest()
        algorithm = "git-blob-sha1"
    if actual != expected:
        raise RuntimeError(f"Remote hash mismatch for {path.name}")
    return {
        "path": path.name,
        "size_bytes": size,
        "sha256": digest,
        "remote_hash_algorithm": algorithm,
        "remote_hash": expected,
        "remote_hash_verified": True,
    }


def main(model_dir=MODEL_DIR):
    model_dir = model_dir.expanduser().absolute()
    info = HfApi(token=False).model_info(
        MODEL_ID, revision=REVISION, files_metadata=True
    )
    if info.sha != REVISION:
        raise RuntimeError(f"Unexpected resolved revision: {info.sha}")
    remote_files = {entry.rfilename: entry for entry in info.siblings}
    for name in FILES:
        if name not in remote_files or remote_files[name].size is None:
            raise RuntimeError(f"Required file or size metadata missing: {name}")
    total_bytes = sum(remote_files[name].size for name in FILES)
    missing_bytes = sum(
        remote_files[name].size
        for name in FILES
        if not (model_dir / name).is_file()
        or (model_dir / name).stat().st_size != remote_files[name].size
    )
    # Recreate the target if a temporary-storage symlink survives a restart.
    model_dir.resolve().mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(model_dir).free < missing_bytes + 64 * 1024**2:
        raise RuntimeError("Insufficient disk space for the selected snapshot")
    print(f"Model: {MODEL_ID}\nRevision: {REVISION}", flush=True)
    print(f"Selected: {len(FILES)} files, {total_bytes:,} bytes", flush=True)
    print(f"Destination: {model_dir}", flush=True)
    print(f"Storage location: {model_dir.resolve()}", flush=True)
    # local_dir stores payloads here directly, with only small local HF metadata.
    # It avoids another copy of the weights in the global Hugging Face cache.
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=model_dir,
        allow_patterns=FILES,
        token=False,
        max_workers=2,
    )
    verified = []
    for name in FILES:
        print(f"Verifying {name} ...", flush=True)
        verified.append(verify_file(model_dir / name, remote_files[name]))
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "qwen2":
        raise RuntimeError(f"Unexpected model type: {config.get('model_type')}")
    source = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "resolved_revision": info.sha,
        "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
        "local_path": (
            model_dir.relative_to(PROJECT_ROOT).as_posix()
            if model_dir.is_relative_to(PROJECT_ROOT)
            else str(model_dir)
        ),
        "storage_path": str(model_dir.resolve()),
        "storage_is_temporary": model_dir.resolve().is_relative_to(Path("/tmp")),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "huggingface_hub_version": huggingface_hub.__version__,
        "download_method": "snapshot_download with local_dir and explicit allow_patterns",
        "total_size_bytes": total_bytes,
        "files": verified,
        "validation": {
            "exact_revision": True,
            "all_sizes_and_remote_hashes_match": True,
            "model_type": config["model_type"],
            "model_loaded": False,
        },
    }
    SOURCE_PATH.write_text(json.dumps(source, indent=2) + "\n")
    print(f"Verified all {len(FILES)} files. Source metadata: {SOURCE_PATH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    main(parser.parse_args().model_dir)
