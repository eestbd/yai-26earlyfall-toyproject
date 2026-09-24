"""Load and validate fixed pilot settings without loading or running the model."""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
from transformers import GenerationConfig

from validate_fixed_input import EXPECTED_SHA256, MODEL_ID, REVISION, load_fixed_input


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "docs" / "pilot_config.json"
RESULT_PATH = PROJECT_ROOT / "results" / "pilot_config_validation.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_fields(data, expected, label):
    for name, value in expected.items():
        actual = data.get(name)
        require(name in data and type(actual) is type(value) and actual == value, f"{label}.{name}: expected {value!r}, got {actual!r}")


def load_pilot_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    require_fields(config, {
        "model_id": MODEL_ID, "model_revision": REVISION,
        "model_source_path": "docs/model_source.json",
        "input_path": "data/shared_5_1_input.json", "input_sha256": EXPECTED_SHA256,
        "prompt_tokens": 128, "batch_size": 1, "tokenizer_add_special_tokens": False,
        "dtype": "bfloat16", "device": "cuda:0", "attention_implementation": "eager",
        "seed": 42, "allow_tf32": False, "model_eval": True, "inference_mode": True,
        "quantization": None, "offload": False, "torch_compile": False,
        "local_files_only": True, "trust_remote_code": False,
        "warmup_runs_per_condition": 2, "warmup_in_statistics": False,
        "measurement_repeats_per_condition": 5,
    }, "pilot")
    # JSON comparison distinguishes actual booleans from integer 0/1.
    require(json.dumps(config["use_cache_values"]) == "[false, true]", "Expected OFF and ON modes")
    expected_order = [[False, True], [True, False], [False, True], [True, False], [False, True]]
    require(json.dumps(config["measurement_order"]) == json.dumps(expected_order), "Unexpected measurement order")
    require(platform.python_version() == config["python_version"], "Python version differs from pilot settings")
    for name, version in config["package_versions"].items():
        require(importlib.metadata.version(name) == version, f"Package version mismatch: {name}")

    fixed = load_fixed_input(PROJECT_ROOT / config["input_path"])
    require(len(fixed["token_ids"]) == config["prompt_tokens"], "Input length mismatch")
    require(fixed["sha256"] == config["input_sha256"], "Input hash mismatch")
    source = json.loads((PROJECT_ROOT / config["model_source_path"]).read_text())
    require(source["model_id"] == MODEL_ID and source["revision"] == source["resolved_revision"] == REVISION, "Model source mismatch")
    model_path = PROJECT_ROOT / source["local_path"]
    require(model_path.is_dir(), "Local model snapshot is missing")
    model_config = json.loads((model_path / "config.json").read_text())
    require(max(fixed["token_ids"]) < model_config["vocab_size"], "Input token exceeds model vocabulary")
    require_fields(config["generation"], {
        "min_new_tokens": 128, "max_new_tokens": 128,
        "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
        "bos_token_id": model_config["bos_token_id"],
        "eos_token_id": model_config["eos_token_id"],
        "pad_token_id": model_config["eos_token_id"],
        "return_dict_in_generate": False, "output_scores": False,
        "output_logits": False, "output_attentions": False, "output_hidden_states": False,
        "remove_invalid_values": False,
    }, "generation")
    require("use_cache" not in config["generation"], "use_cache must be supplied separately per condition")
    generation_config_for(config, False)
    generation_config_for(config, True)
    return config


def generation_config_for(config, use_cache):
    """Fresh configs prevent checkpoint sampling defaults or ON/OFF mutation leaks."""
    require(type(use_cache) is bool, "use_cache must be a boolean")
    generation = GenerationConfig(**config["generation"], use_cache=use_cache)
    generation.validate()
    return generation


def model_loading_kwargs(config):
    """Arguments for future local model loading; device placement stays explicit."""
    require(config["dtype"] == "bfloat16", "Pilot requires BF16")
    return {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": config["attention_implementation"],
        "low_cpu_mem_usage": True,
        "device_map": None,
        "local_files_only": config["local_files_only"],
        "trust_remote_code": config["trust_remote_code"],
        "use_safetensors": True,
    }


def configure_runtime(config):
    """Apply seeds and TF32 policy before a future model load or inference run."""
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]


def main():
    config = load_pilot_config()
    configure_runtime(config)
    off = generation_config_for(config, False).to_dict()
    on = generation_config_for(config, True).to_dict()
    differences = sorted(key for key in off.keys() | on.keys() if off.get(key) != on.get(key))
    require(differences == ["use_cache"], f"ON/OFF differ in unexpected settings: {differences}")
    require(torch.initial_seed() == config["seed"], "Torch seed was not applied")
    require(torch.backends.cuda.matmul.allow_tf32 is False, "TF32 is not disabled")
    fixed = load_fixed_input(PROJECT_ROOT / config["input_path"])
    # Construct only a tiny CPU integer tensor; no GPU transfer or model execution.
    input_ids = torch.tensor([fixed["token_ids"]], dtype=torch.long)
    require(list(input_ids.shape) == [1, 128], "Unexpected input shape")
    kwargs = model_loading_kwargs(config)
    result = {
        "status": "passed",
        "scope": "step_10_pilot_configuration_validation_only",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "config_file_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": config,
        "input_shape": list(input_ids.shape),
        "input_tensor_dtype": str(input_ids.dtype),
        "input_tensor_device": str(input_ids.device),
        "input_hash_verified": True,
        "runtime_seed": torch.initial_seed(),
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "model_loading_kwargs": {key: str(value) if key == "torch_dtype" else value for key, value in kwargs.items()},
        "generation_config_off": off,
        "generation_config_on": on,
        "on_off_differing_fields": differences,
        "python_and_package_versions_match": True,
        "retokenization_performed": False,
        "model_loaded": False,
        "generation_performed": False,
        "warmup_performed": False,
        "benchmark_performed": False,
        "limitations": "Settings only: actual model eval/inference mode, GPU inputs and generated token counts must be checked when executing future runs.",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: BF16/eager pilot configuration matches the fixed model, input and installed versions.")
    print(f"Input shape: {list(input_ids.shape)}; dtype: {input_ids.dtype}; device: {input_ids.device}")
    print(f"ON/OFF differing fields: {differences}")
    print("New tokens: min=max=128; greedy; batch=1; seed=42; TF32 off.")
    print("Planned: warm-up OFF/ON 2 each; measurement OFF/ON 5 each, alternating order.")
    print("No model load, generation, warm-up or benchmark was executed.")
    print(f"Result: {RESULT_PATH}")


if __name__ == "__main__":
    main()
