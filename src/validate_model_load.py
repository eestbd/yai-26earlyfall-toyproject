"""Validate local Qwen tokenizer/model loading; do not generate any tokens."""

import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

# This validation must not download anything, including tokenizer assets.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "docs" / "model_source.json"
RESULT_PATH = PROJECT_ROOT / "results" / "model_load.json"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    source_bytes = SOURCE_PATH.read_bytes()
    source = json.loads(source_bytes)
    require(source["model_id"] == MODEL_ID, "Unexpected model ID")
    require(source["revision"] == REVISION, "Unexpected model revision")
    require(source["resolved_revision"] == REVISION, "Unexpected resolved revision")
    model_path = PROJECT_ROOT / source["local_path"]
    require(
        model_path.is_dir(),
        "Local snapshot is missing. Restore it with src/download_model.py first.",
    )
    for entry in source["files"]:
        path = model_path / entry["path"]
        require(path.is_file(), f"Missing snapshot file: {entry['path']}")
        require(
            path.stat().st_size == entry["size_bytes"],
            f"Snapshot file size changed: {entry['path']}",
        )
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.is_bf16_supported(), "Current GPU/runtime does not support BF16")
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False

    print(f"Loading local tokenizer: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    print("Loading model with BF16 parameters and eager attention ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map=None,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    # Explicit placement keeps the complete model on one GPU for inference.
    model = model.to(device)
    model.eval()
    torch.cuda.synchronize()

    parameter_count = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    parameter_devices = sorted({str(p.device) for p in model.parameters()})
    parameter_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    attention_classes = sorted({type(layer.self_attn).__name__ for layer in model.model.layers})
    require(parameter_count > 0, "Model has no parameters")
    require(parameter_devices == [str(device)], "Some parameters are not on cuda:0")
    require(parameter_dtypes == ["torch.bfloat16"], "Some parameters are not BF16")
    require(not any(module.training for module in model.modules()), "Model is not fully in eval mode")
    require(model.config._attn_implementation == "eager", "Unexpected attention backend")
    require(attention_classes == ["Qwen2Attention"], "Unexpected attention layer implementation")
    require(not getattr(model, "is_quantized", False), "Unexpected quantization")
    require(len(tokenizer) <= model.get_input_embeddings().num_embeddings, "Tokenizer exceeds embedding vocabulary")

    package_names = ["torch", "transformers", "accelerate", "huggingface-hub", "safetensors", "numpy"]
    result = {
        "status": "passed",
        "scope": "step_6_local_tokenizer_and_model_loading_only",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "local_path": source["local_path"],
        "storage_path": str(model_path.resolve()),
        "storage_is_temporary": model_path.resolve().is_relative_to(Path("/tmp")),
        "source_metadata_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer),
        "embedding_vocabulary_size": model.get_input_embeddings().num_embeddings,
        "model_class": type(model).__name__,
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "parameter_devices": parameter_devices,
        "parameter_dtypes": parameter_dtypes,
        "buffer_dtypes": sorted({str(b.dtype) for b in model.buffers()}),
        "eval_mode": not model.training,
        "attention_implementation": model.config._attn_implementation,
        "attention_classes": attention_classes,
        "quantized": bool(getattr(model, "is_quantized", False)),
        "seed": 42,
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "local_files_only": True,
        "generation_performed": False,
        "limitations": "This validates loading only; forward, generation and KV Cache behavior are not yet tested.",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Tokenizer: {result['tokenizer_class']}, size={len(tokenizer):,}")
    print(f"Model: {result['model_class']}, parameters={parameter_count:,}")
    print(f"Device: {parameter_devices}; dtype: {parameter_dtypes}")
    print(f"Evaluation mode: {result['eval_mode']}; attention: {attention_classes}")
    print(f"PASS: Model loading validated. Result: {RESULT_PATH}")


if __name__ == "__main__":
    main()
