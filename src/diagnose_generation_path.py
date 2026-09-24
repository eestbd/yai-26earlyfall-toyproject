"""Trace actual generate forwards without changing or timing the pilot benchmark."""

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pilot_config import PROJECT_ROOT, configure_runtime, generation_config_for, load_pilot_config, model_loading_kwargs, require
from validate_fixed_input import load_fixed_input

import torch
from transformers import AutoModelForCausalLM


def cache_length(cache):
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2]) if len(cache) else 0


def main():
    config = load_pilot_config()
    configure_runtime(config)
    source = json.loads((PROJECT_ROOT / config["model_source_path"]).read_text())
    model = AutoModelForCausalLM.from_pretrained(
        PROJECT_ROOT / source["local_path"], **model_loading_kwargs(config),
    ).to(config["device"]).eval()
    ids = load_fixed_input()["token_ids"]
    input_ids = torch.tensor([ids], device=config["device"], dtype=torch.long)
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    reference_path = PROJECT_ROOT / "results" / "pilot_memory.json"
    reference = json.loads(reference_path.read_text())
    result = {
        "scope": "actual_generate_cache_path_diagnostic", "status": "running",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "performance_measurement": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(), "runs": [],
    }
    for use_cache in [False, True]:
        run = {"use_cache": use_cache, "forward_steps": []}
        result["runs"].append(run)

        def before(_model, args, kwargs):
            run["forward_steps"].append({
                "step": len(run["forward_steps"]),
                "input_tokens": int(kwargs["input_ids"].shape[1]),
                "forward_use_cache": kwargs.get("use_cache"),
                "incoming_cache_length": cache_length(kwargs.get("past_key_values")),
            })

        def after(_model, args, kwargs, output):
            cache = output.past_key_values
            run["forward_steps"][-1].update(
                output_cache_present=cache is not None,
                output_cache_length=cache_length(cache),
            )
            if len(run["forward_steps"]) == config["generation"]["max_new_tokens"]:
                layers = list(cache) if cache is not None else []
                run["final_cache_layers"] = len(layers)
                run["final_first_key_shape"] = list(layers[0][0].shape) if layers else None
                run["final_cache_tensor_bytes"] = sum(t.numel() * t.element_size() for pair in layers for t in pair)

        pre = model.register_forward_pre_hook(before, with_kwargs=True)
        post = model.register_forward_hook(after, with_kwargs=True)
        try:
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=generation_config_for(config, use_cache))
            torch.cuda.synchronize()
            generated = output[0, len(ids):].tolist()
        finally:
            pre.remove()
            post.remove()
        steps = run["forward_steps"]
        require(len(steps) == 128 and len(generated) == 128, "Unexpected generation length")
        expected_inputs = [128] + ([1] * 127 if use_cache else list(range(129, 256)))
        require([s["input_tokens"] for s in steps] == expected_inputs, "Unexpected forward input lengths")
        require(all(s["forward_use_cache"] is use_cache and s["output_cache_present"] is use_cache for s in steps),
                "Actual cache usage differs from requested mode")
        expected_incoming = [0] + (list(range(128, 255)) if use_cache else [0] * 127)
        require([s["incoming_cache_length"] for s in steps] == expected_incoming, "Incoming cache does not grow as expected")
        require([s["output_cache_length"] for s in steps] == (list(range(128, 256)) if use_cache else [0] * 128),
                "Output cache does not grow as expected")
        ref = next(r for r in reference["repeat_runs"] if r["use_cache"] is use_cache)
        run["matches_pilot_output"] = generated == ref["generated_token_ids"]
        require(run["matches_pilot_output"], "Diagnostic output differs from pilot")
        run["status"] = "passed"
        print(f"{'ON' if use_cache else 'OFF'}: inputs {[s['input_tokens'] for s in steps[:4]]} ... "
              f"{steps[-1]['input_tokens']}; final cache {run['final_cache_tensor_bytes']} bytes; pilot output matches", flush=True)
        del output
        gc.collect()
    result["status"] = "passed"
    path = PROJECT_ROOT / "results" / "generation_path_diagnostic.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Result: {path}")


if __name__ == "__main__":
    main()
