"""Run one local greedy generation check, without benchmarking or cache comparison."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "docs" / "model_source.json"
RESULT_PATH = PROJECT_ROOT / "results" / "basic_generation.json"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
# This is a temporary functional-check prompt, not the fixed benchmark input.
PROMPT = "The purpose of a key-value cache in autoregressive language models is"
NEW_TOKENS = 128
SEED = 42


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
    require(model_path.is_dir(), "Local snapshot is missing; run src/download_model.py first")
    for entry in source["files"]:
        path = model_path / entry["path"]
        require(path.is_file(), f"Missing snapshot file: {entry['path']}")
        require(path.stat().st_size == entry["size_bytes"], f"File size changed: {entry['path']}")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.is_bf16_supported(), "Current GPU/runtime does not support BF16")
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")

    print("Loading local tokenizer and BF16/eager model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map=None,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device)
    model.eval()
    require(all(p.device == device and p.dtype == torch.bfloat16 for p in model.parameters()), "Unexpected parameter device or dtype")
    require(model.config._attn_implementation == "eager", "Unexpected attention backend")
    require(not any(module.training for module in model.modules()), "Model is not in eval mode")
    require(not getattr(model, "is_quantized", False), "Unexpected quantization")
    require(tokenizer.eos_token_id is not None, "Expected a tokenizer EOS token")

    inputs = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_tokens = inputs["input_ids"].shape[1]
    # Equal min/max new-token limits suppress EOS until the requested length.
    # A fresh config avoids inheriting sampling parameters from the checkpoint.
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        min_new_tokens=NEW_TOKENS,
        max_new_tokens=NEW_TOKENS,
        num_return_sequences=1,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_logits=True,
    )
    print(f"Generating exactly {NEW_TOKENS} new tokens from {prompt_tokens} input tokens ...", flush=True)
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=generation_config)
    torch.cuda.synchronize()
    sequences = output.sequences
    require(sequences.shape[0] == 1, "Unexpected output batch size")
    require(torch.equal(sequences[:, :prompt_tokens], inputs["input_ids"]), "Output does not preserve the input prefix")
    generated_ids = sequences[0, prompt_tokens:].cpu().tolist()
    require(len(generated_ids) == NEW_TOKENS, f"Expected {NEW_TOKENS} new tokens, got {len(generated_ids)}")
    require(all(0 <= token < model.config.vocab_size for token in generated_ids), "Generated token outside model vocabulary")
    require(len(output.logits) == len(generated_ids), "Missing per-step raw logits")
    # Raw logits must be finite; processed scores may contain -inf for suppressed EOS.
    nonfinite_logit_steps = [
        step for step, logits in enumerate(output.logits, start=1)
        if not torch.isfinite(logits).all().item()
    ]
    generated_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    package_names = ["torch", "transformers", "accelerate", "huggingface-hub", "safetensors", "numpy"]
    result = {
        "status": "failed" if nonfinite_logit_steps else "passed",
        "scope": "step_7_basic_greedy_generation_only",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "local_path": source["local_path"],
        "storage_path": str(model_path.resolve()),
        "source_metadata_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "dtype": str(model.dtype),
        "device": str(model.device),
        "attention_implementation": model.config._attn_implementation,
        "batch_size": 1,
        "seed": SEED,
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "eval_mode": not model.training,
        "inference_mode_used": True,
        "local_files_only": True,
        "prompt": PROMPT,
        "input_purpose": "temporary functional check; not the fixed benchmark input",
        "tokenizer_add_special_tokens": False,
        "input_token_ids": inputs["input_ids"][0].cpu().tolist(),
        "prompt_tokens": prompt_tokens,
        "requested_generated_tokens": NEW_TOKENS,
        "actual_generated_tokens": len(generated_ids),
        "output_total_tokens": sequences.shape[1],
        "input_prefix_preserved": True,
        "raw_logits_all_finite": not nonfinite_logit_steps,
        "nonfinite_logit_steps_1based": nonfinite_logit_steps,
        "generation_config": generation_config.to_dict(),
        "eos_policy": "min_new_tokens=max_new_tokens=128; EOS is suppressed before that many new tokens",
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "kv_cache_behavior_validated": False,
        "benchmark_performed": False,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Prompt: {PROMPT}")
    print(f"Generated text:\n{generated_text}")
    print(f"Result: {RESULT_PATH}")
    require(
        not nonfinite_logit_steps,
        f"Non-finite raw logits in {len(nonfinite_logit_steps)} generation steps; "
        "token count alone does not establish correct generation. See the saved result.",
    )
    print(f"PASS: {len(generated_ids)} new tokens; all per-step raw logits are finite.")


if __name__ == "__main__":
    main()
