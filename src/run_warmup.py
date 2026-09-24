"""Run four unmeasured warm-ups; reuse this function in the benchmark process."""

import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pilot_config import (
    CONFIG_PATH, PROJECT_ROOT, configure_runtime, generation_config_for,
    load_pilot_config, model_loading_kwargs, require,
)
from validate_fixed_input import load_fixed_input, token_ids_sha256

import torch
from transformers import AutoModelForCausalLM


RESULT_PATH = PROJECT_ROOT / "results" / "warmup_validation.json"


def run_warmup(model, inputs, config, records=None):
    """Warm the supplied model in this process; return CPU-only audit records.

    Temporary logits checks are removed before returning, including on failure.
    These runs must not contribute to performance statistics.
    """
    if records is None:
        records = []
    device = torch.device(config["device"])
    require(not any(module.training for module in model.modules()), "Model must be in eval mode")
    require(all(p.dtype == torch.bfloat16 and p.device == device for p in model.parameters()),
            "All parameters must be BF16 on the configured GPU")
    require(model.config._attn_implementation == config["attention_implementation"], "Attention mismatch")
    require(not getattr(model, "is_quantized", False), "Quantization is disabled")
    require(torch.backends.cuda.matmul.allow_tf32 == config["allow_tf32"], "TF32 mismatch")
    fixed = load_fixed_input(PROJECT_ROOT / config["input_path"])
    expected_ids = torch.tensor([fixed["token_ids"]], dtype=torch.long, device=device)
    require(set(inputs) == {"input_ids", "attention_mask"}, "Unexpected model inputs")
    require(inputs["input_ids"].device == device and inputs["input_ids"].dtype == torch.long,
            "Input IDs must be int64 on the configured GPU")
    require(torch.equal(inputs["input_ids"], expected_ids), "Input differs from fixed token IDs")
    require(inputs["attention_mask"].device == device
            and torch.equal(inputs["attention_mask"], torch.ones_like(expected_ids)), "Invalid attention mask")
    del expected_ids
    off = generation_config_for(config, False).to_dict()
    on = generation_config_for(config, True).to_dict()
    require([key for key in off if off[key] != on[key]] == ["use_cache"], "Unexpected ON/OFF differences")

    for use_cache in config["use_cache_values"]:
        for repeat in range(config["warmup_runs_per_condition"]):
            record = {
                "use_cache": use_cache, "repeat": repeat, "warmup": True,
                "included_in_statistics": False, "status": "running",
                "prompt_tokens": config["prompt_tokens"],
                "requested_new_tokens": config["generation"]["max_new_tokens"],
                "finite_logits_steps": 0,
            }
            records.append(record)

            def check_logits(_module, _args, output):
                require(torch.is_inference_mode_enabled(), "Inference mode is required")
                # Inspect only next-token raw logits; retain no output or KV tensors.
                require(torch.isfinite(output.logits[:, -1, :]).all().item(), "Non-finite raw logits")
                record["finite_logits_steps"] += 1

            gc.collect()
            torch.cuda.synchronize(device)
            hook = model.register_forward_hook(check_logits)
            outputs = None
            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs, generation_config=generation_config_for(config, use_cache),
                    )
                torch.cuda.synchronize(device)
                actual = outputs.shape[1] - config["prompt_tokens"]
                require(list(outputs.shape) == [config["batch_size"], config["prompt_tokens"] + record["requested_new_tokens"]],
                        "Unexpected generation shape")
                require(torch.equal(outputs[:, :config["prompt_tokens"]], inputs["input_ids"]), "Prompt prefix changed")
                require(inputs["input_ids"].tolist() == [fixed["token_ids"]], "Input IDs were mutated")
                require(record["finite_logits_steps"] == record["requested_new_tokens"], "Unexpected forward-step count")
                ids = outputs[0, config["prompt_tokens"]:].tolist()
                record.update({
                    "status": "passed", "actual_generated_tokens": actual,
                    "all_next_token_logits_finite": True,
                    "generated_token_ids": ids,
                    "generated_token_ids_sha256": token_ids_sha256(ids),
                })
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                hook.remove()
                record["diagnostic_hook_removed"] = True
                del outputs
            print(f"PASS warm-up {'ON' if use_cache else 'OFF'} {repeat + 1}/2: "
                  f"{actual} new tokens; {record['finite_logits_steps']} finite logits steps", flush=True)
    gc.collect()
    torch.cuda.synchronize(device)
    return records


def main():
    report = {
        "status": "running", "scope": "step_11_warmup_only",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_file_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "included_in_statistics": False, "benchmark_performed": False,
        "performance_metrics_recorded": False,
        "rerun_in_same_process_before_measurement": True, "runs": [],
    }
    try:
        config = load_pilot_config()
        configure_runtime(config)
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 is required")
        source = json.loads((PROJECT_ROOT / config["model_source_path"]).read_text())
        model_path = PROJECT_ROOT / source["local_path"]
        report.update(config=config, gpu_name=torch.cuda.get_device_name(0), input_hash_verified=True)
        print("Loading local BF16/eager model for OFF x2, then ON x2 warm-up.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_loading_kwargs(config))
        model.to(config["device"]).eval()
        fixed = load_fixed_input(PROJECT_ROOT / config["input_path"])
        input_ids = torch.tensor([fixed["token_ids"]], dtype=torch.long, device=config["device"])
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        run_warmup(model, inputs, config, records=report["runs"])
        require(len(report["runs"]) == 4, "Expected four warm-up runs")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Result: {RESULT_PATH}", flush=True)
    print("Warm-up validation passed. No performance measurements were taken.", flush=True)


if __name__ == "__main__":
    main()
