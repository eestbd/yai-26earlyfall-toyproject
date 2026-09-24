"""Validate the fixed token IDs using only the Python standard library."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "shared_5_1_input.json"
RESULT_PATH = PROJECT_ROOT / "results" / "fixed_input_validation.json"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
PROMPT_TOKENS = 128
EXPECTED_SHA256 = "5fcc0f299f56df0dcefe9385960670e79276e2727011ef96bb0654b06186fa34"
HASH_SERIALIZATION = "json.dumps(token_ids).encode('utf-8')"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_ids_sha256(token_ids):
    # Explicit separators reproduce json.dumps defaults, including comma spaces.
    # Hash the ordered list only, with no trailing newline or surrounding metadata.
    payload = json.dumps(token_ids, ensure_ascii=True, separators=(", ", ": "))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_fixed_input(path=INPUT_PATH):
    """Read and validate IDs without loading a model or running a tokenizer."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(data, dict), "Input JSON must be an object")
    require(data.get("model_id") == MODEL_ID, "Unexpected model ID")
    require(data.get("model_revision") == REVISION, "Unexpected model revision")
    require(data.get("tokenizer_add_special_tokens") is False, "Special tokens must not be added")
    require(type(data.get("prompt_tokens")) is int and data["prompt_tokens"] == PROMPT_TOKENS, "Unexpected prompt_tokens metadata")
    ids = data.get("token_ids")
    require(isinstance(ids, list), "token_ids must be a list")
    require(len(ids) == PROMPT_TOKENS, f"Expected exactly {PROMPT_TOKENS} token IDs")
    require(all(type(token) is int and token >= 0 for token in ids), "Token IDs must be nonnegative integers")
    require(data.get("sha256_serialization") == HASH_SERIALIZATION, "Unexpected hash serialization")
    require(data.get("sha256") == EXPECTED_SHA256, "Stored hash differs from the reference hash")
    computed = token_ids_sha256(ids)
    require(computed == EXPECTED_SHA256, f"Token ID hash mismatch: {computed}")
    return data


def main():
    data = load_fixed_input()
    result = {
        "status": "passed",
        "scope": "step_9_fixed_input_validation_only",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "input_path": INPUT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "input_file_sha256": hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest(),
        "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_id": data["model_id"],
        "model_revision": data["model_revision"],
        "tokenizer_add_special_tokens": data["tokenizer_add_special_tokens"],
        "prompt_tokens": len(data["token_ids"]),
        "min_token_id": min(data["token_ids"]),
        "max_token_id": max(data["token_ids"]),
        "expected_token_ids_sha256": EXPECTED_SHA256,
        "computed_token_ids_sha256": token_ids_sha256(data["token_ids"]),
        "sha256_serialization": HASH_SERIALIZATION,
        "reference_hash_matches": True,
        "retokenization_performed": False,
        "model_loaded": False,
        "benchmark_performed": False,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(data['token_ids'])} fixed token IDs; model ID and revision match.")
    print(f"Token ID SHA-256: {result['computed_token_ids_sha256']}")
    print(f"Hash serialization: {HASH_SERIALIZATION}")
    print("No tokenizer or model was loaded.")
    print(f"Result: {RESULT_PATH}")


if __name__ == "__main__":
    main()
