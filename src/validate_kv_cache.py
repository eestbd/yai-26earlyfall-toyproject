"""Inspect real KV Cache behavior on a short input, without benchmarking."""

import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "docs" / "model_source.json"
RESULT_PATH = PROJECT_ROOT / "results" / "kv_cache_validation.json"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
PROMPT = "The purpose of a key-value cache in autoregressive language models is"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def cache_layers(cache):
    if isinstance(cache, DynamicCache):
        return cache.to_legacy_cache()
    require(isinstance(cache, (tuple, list)), f"Unexpected cache type: {type(cache)}")
    return cache


def inspect_cache(cache, config, sequence_length, device):
    require(cache is not None, "Cache ON returned no past_key_values")
    layers = cache_layers(cache)
    require(len(layers) == config.num_hidden_layers, "Unexpected cache layer count")
    shape = [1, config.num_key_value_heads, sequence_length, config.hidden_size // config.num_attention_heads]
    records = []
    for index, pair in enumerate(layers):
        require(len(pair) == 2, f"Layer {index}: expected K and V")
        key, value = pair
        require(list(key.shape) == shape and list(value.shape) == shape, f"Layer {index}: unexpected K/V shape")
        require(key.dtype == value.dtype == torch.bfloat16, f"Layer {index}: cache dtype is not BF16")
        require(key.device == value.device == device, f"Layer {index}: cache is not on cuda:0")
        require(torch.isfinite(key).all().item() and torch.isfinite(value).all().item(), f"Layer {index}: non-finite cache values")
        records.append({
            "layer": index,
            "key_shape": list(key.shape),
            "value_shape": list(value.shape),
            "key_dtype": str(key.dtype),
            "value_dtype": str(value.dtype),
            "key_device": str(key.device),
            "value_device": str(value.device),
            "all_values_finite": True,
        })
    return records


def last_logits(output, label):
    require(torch.isfinite(output.logits).all().item(), f"{label}: non-finite logits")
    return output.logits[:, -1, :].float()


def main():
    source_bytes = SOURCE_PATH.read_bytes()
    source = json.loads(source_bytes)
    require(source["model_id"] == MODEL_ID, "Unexpected model ID")
    require(source["revision"] == source["resolved_revision"] == REVISION, "Unexpected model revision")
    model_path = PROJECT_ROOT / source["local_path"]
    require(model_path.is_dir(), "Local snapshot is missing; run src/download_model.py first")
    for entry in source["files"]:
        path = model_path / entry["path"]
        require(path.is_file(), f"Missing snapshot file: {entry['path']}")
        require(path.stat().st_size == entry["size_bytes"], f"File size changed: {entry['path']}")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.is_bf16_supported(), "Current GPU/runtime does not support BF16")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    print("Loading local BF16/eager model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
        low_cpu_mem_usage=True, device_map=None, local_files_only=True,
        trust_remote_code=False, use_safetensors=True,
    ).to(device).eval()
    require(all(p.device == device and p.dtype == torch.bfloat16 for p in model.parameters()), "Unexpected parameter device or dtype")
    require(model.config._attn_implementation == "eager", "Unexpected attention backend")
    require(not any(module.training for module in model.modules()), "Model is not in eval mode")
    require(not getattr(model, "is_quantized", False), "Unexpected quantization")
    inputs = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        # The prefill calls differ only in use_cache; both start without a cache.
        off = model(**inputs, use_cache=False, return_dict=True)
        on = model(**inputs, use_cache=True, return_dict=True)
        require(off.past_key_values is None, "Cache OFF unexpectedly returned past_key_values")
        prefill_records = inspect_cache(on.past_key_values, model.config, prompt_length, device)
        off_logits = last_logits(off, "OFF prefill")
        on_logits = last_logits(on, "ON prefill")
        next_token = on_logits.argmax(dim=-1, keepdim=True)
        require(torch.equal(next_token, off_logits.argmax(dim=-1, keepdim=True)), "ON/OFF prefill next-token predictions differ")

        # Keep the original tensors so we can check that the stored prefix survives.
        original_layers = cache_layers(on.past_key_values)
        cache = on.past_key_values
        if not isinstance(cache, DynamicCache):
            cache = DynamicCache.from_legacy_cache(cache)
        require(cache.get_seq_length() == prompt_length, "Unexpected initial cache length")
        extended_ids = torch.cat([inputs["input_ids"], next_token], dim=1)
        extended_mask = torch.cat([inputs["attention_mask"], torch.ones_like(next_token)], dim=1)
        cached_step = model(
            input_ids=next_token,
            attention_mask=extended_mask,
            position_ids=torch.tensor([[prompt_length]], device=device),
            cache_position=torch.tensor([prompt_length], device=device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        grown_records = inspect_cache(cached_step.past_key_values, model.config, prompt_length + 1, device)
        require(cached_step.past_key_values is cache, "The supplied DynamicCache was not reused")
        require(cache.get_seq_length() == prompt_length + 1, "Cache did not grow by one token")
        for index, ((old_k, old_v), (new_k, new_v)) in enumerate(zip(original_layers, cache_layers(cache))):
            require(torch.equal(old_k, new_k[:, :, :prompt_length, :]), f"Layer {index}: cached key prefix changed")
            require(torch.equal(old_v, new_v[:, :, :prompt_length, :]), f"Layer {index}: cached value prefix changed")

        full_step = model(input_ids=extended_ids, attention_mask=extended_mask, use_cache=False, return_dict=True)
        require(full_step.past_key_values is None, "Extended OFF input unexpectedly returned a cache")
        cached_logits = last_logits(cached_step, "ON incremental step")
        full_logits = last_logits(full_step, "OFF full-prefix recomputation")
        require(torch.equal(cached_logits.argmax(dim=-1), full_logits.argmax(dim=-1)), "Cached and recomputed next-token predictions differ")
        prefill_difference = (on_logits - off_logits).abs().max().item()
        incremental_difference = (cached_logits - full_logits).abs().max().item()
    torch.cuda.synchronize()

    package_names = ["torch", "transformers", "accelerate", "huggingface-hub", "safetensors", "numpy"]
    result = {
        "status": "passed",
        "scope": "step_8_short_input_cache_behavior_validation",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "source_metadata_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "local_path": source["local_path"],
        "storage_path": str(model_path.resolve()),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "dtype": str(model.dtype),
        "device": str(model.device),
        "attention_implementation": model.config._attn_implementation,
        "seed": 42,
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "eval_mode": not model.training,
        "inference_mode_used": True,
        "local_files_only": True,
        "prompt": PROMPT,
        "input_purpose": "temporary functional check; not the fixed benchmark input",
        "tokenizer_add_special_tokens": False,
        "input_token_ids": inputs["input_ids"][0].cpu().tolist(),
        "prompt_tokens": prompt_length,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "head_dim": model.config.hidden_size // model.config.num_attention_heads,
        "shape_axes": ["batch", "kv_heads", "cached_tokens", "head_dim"],
        "prefill": {
            "off_past_key_values_is_none": off.past_key_values is None,
            "on_past_key_values_is_none": on.past_key_values is None,
            "on_cache_type": type(on.past_key_values).__name__,
            "layers": prefill_records,
            "all_logits_finite": True,
            "next_token_ids_match": True,
            "next_token_id": next_token.item(),
            "max_abs_logit_difference": prefill_difference,
        },
        "incremental_step": {
            "on_input_tokens": 1,
            "off_input_tokens": prompt_length + 1,
            "cache_length_before": prompt_length,
            "cache_length_after": cache.get_seq_length(),
            "cache_type": type(cache).__name__,
            "same_cache_object_reused": True,
            "cached_prefix_preserved_exactly": True,
            "off_past_key_values_is_none": full_step.past_key_values is None,
            "layers": grown_records,
            "all_logits_finite": True,
            "next_token_ids_match": True,
            "next_token_id": cached_logits.argmax(dim=-1).item(),
            "max_abs_logit_difference": incremental_difference,
        },
        "benchmark_performed": False,
        "limitations": "Only prefill and one incremental step were checked. Full 128-token ON/OFF output equality and performance measurement are still pending.",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"OFF: past_key_values is None = {off.past_key_values is None}")
    print(f"ON: past_key_values is None = {on.past_key_values is None}; layers = {len(prefill_records)}")
    print(f"K/V shape in every layer: {prefill_records[0]['key_shape']} -> {grown_records[0]['key_shape']}")
    print(f"Cache length: {prompt_length} -> {cache.get_seq_length()}; prefix preserved exactly")
    print(f"Next-token predictions match; max logit differences: prefill={prefill_difference}, incremental={incremental_difference}")
    print(f"PASS: Real cache behavior verified. Result: {RESULT_PATH}")


if __name__ == "__main__":
    main()
