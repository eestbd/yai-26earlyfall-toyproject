"""Measure generation latency, throughput and CUDA allocator memory."""

import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from pilot_config import (
    CONFIG_PATH, PROJECT_ROOT, configure_runtime, generation_config_for,
    load_pilot_config, model_loading_kwargs, require,
)
from run_warmup import run_warmup
from calculate_throughput import THROUGHPUT_DEFINITION, calculate_throughput
from validate_fixed_input import load_fixed_input, token_ids_sha256

import torch
from transformers import AutoModelForCausalLM


RESULT_PATH = PROJECT_ROOT / "results" / "pilot_memory.json"


def run_pilot_repeats(model, inputs, config, records=None):
    """Run the fixed ten-generation schedule after same-process warm-up.

    Time only generate plus its completion synchronization, with GPU inputs ready.
    Records contain CPU lists and scalars, never GPU outputs or KV caches.
    """
    if records is None:
        records = []
    require(not records, "Repeat records must start empty")
    require(not any(module.training for module in model.modules()), "Model must be in eval mode")
    require(not model._forward_hooks, "Remove warm-up diagnostic hooks before repeats")
    device = torch.device(config["device"])
    prompt_tokens = config["prompt_tokens"]
    requested = config["generation"]["max_new_tokens"]
    fixed_ids = load_fixed_input(PROJECT_ROOT / config["input_path"])["token_ids"]
    require(inputs["input_ids"].tolist() == [fixed_ids], "Input must match the fixed token IDs")
    for repeat, order in enumerate(config["measurement_order"]):
        for position, use_cache in enumerate(order):
            record = {
                "execution_index": len(records), "repeat": repeat,
                "position_in_repeat": position, "use_cache": use_cache,
                "warmup": False, "status": "running",
                "included_in_statistics": False,
                "performance_metrics_recorded": False,
                "prompt_tokens": prompt_tokens, "requested_new_tokens": requested,
            }
            records.append(record)
            generation = generation_config_for(config, use_cache)
            outputs = None
            try:
                gc.collect()
                with torch.inference_mode():
                    torch.cuda.synchronize(device)
                    baseline_allocated_bytes = torch.cuda.memory_allocated(device)
                    baseline_reserved_bytes = torch.cuda.memory_reserved(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    outputs = model.generate(**inputs, generation_config=generation)
                    torch.cuda.synchronize(device)
                    total_latency_seconds = time.perf_counter() - started
                    # Read peaks before validation creates any additional GPU tensors.
                    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
                    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
                require(0 < baseline_allocated_bytes <= peak_allocated_bytes <= peak_reserved_bytes,
                        "Invalid allocated/peak memory readings")
                require(baseline_allocated_bytes <= baseline_reserved_bytes <= peak_reserved_bytes,
                        "Invalid reserved memory readings")
                require(math.isfinite(total_latency_seconds) and total_latency_seconds > 0,
                        "Latency must be finite and positive")
                require(list(outputs.shape) == [config["batch_size"], prompt_tokens + requested],
                        "Unexpected generation shape or new-token count")
                require(torch.equal(outputs[:, :prompt_tokens], inputs["input_ids"]), "Prompt prefix changed")
                require(inputs["input_ids"].tolist() == [fixed_ids], "Input IDs were mutated")
                require(torch.equal(inputs["attention_mask"], torch.ones_like(inputs["input_ids"])),
                        "Attention mask changed")
                ids = outputs[0, prompt_tokens:].tolist()
                record.update(
                    status="passed", actual_generated_tokens=len(ids),
                    total_latency_seconds=total_latency_seconds,
                    baseline_allocated_bytes=baseline_allocated_bytes,
                    baseline_reserved_bytes=baseline_reserved_bytes,
                    peak_allocated_bytes=peak_allocated_bytes,
                    peak_reserved_bytes=peak_reserved_bytes,
                    peak_allocated_above_baseline_bytes=peak_allocated_bytes - baseline_allocated_bytes,
                    throughput_tokens_per_second=calculate_throughput(len(ids), total_latency_seconds),
                    performance_metrics_recorded=True, included_in_statistics=True,
                    generated_token_ids=ids, generated_token_ids_sha256=token_ids_sha256(ids),
                )
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                del outputs
            print(f"PASS repeat {repeat}, position {position}: "
                  f"{'ON' if use_cache else 'OFF'}, {len(ids)} new tokens, "
                  f"{total_latency_seconds:.6f} s, "
                  f"{record['throughput_tokens_per_second']:.6f} tokens/s, "
                  f"peak allocated/reserved {peak_allocated_bytes / 2**20:.2f}/"
                  f"{peak_reserved_bytes / 2**20:.2f} MiB", flush=True)
    expected = config["measurement_repeats_per_condition"]
    require(len(records) == 2 * expected, "Unexpected total run count")
    for mode in config["use_cache_values"]:
        require(sum(r["use_cache"] is mode for r in records) == expected, "Unexpected runs per condition")
    gc.collect()
    torch.cuda.synchronize(device)
    return records


def main():
    report = {
        "status": "running", "scope": "step_15_latency_throughput_and_memory",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / "src" / name).read_bytes()).hexdigest()
            for name in ("run_pilot.py", "run_warmup.py", "pilot_config.py", "validate_fixed_input.py", "calculate_throughput.py")
        },
        "performance_metrics_recorded": False, "statistics_computed": False,
        "measured_metrics": ["total_latency_seconds", "baseline_allocated_bytes",
                             "baseline_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"],
        "derived_metrics": ["throughput_tokens_per_second", "peak_allocated_above_baseline_bytes"],
        "memory_measurement": {
            "unit": "bytes", "display_mib_divisor": 2**20,
            "scope": "PyTorch CUDA allocator in this process",
            "baseline": "after gc.collect and CUDA synchronization, with model and inputs resident",
            "peak_reset_each_run": True,
            "peaks_read_before_output_validation": True,
            "empty_cache_between_runs": False,
            "reserved_includes_reusable_allocator_cache": True,
            "allocated_delta_is_kv_cache_size": False,
            "includes_non_pytorch_cuda_allocations": False,
        },
        "throughput_definition": THROUGHPUT_DEFINITION,
        "timing": {
            "clock": "time.perf_counter", "unit": "seconds",
            "scope": "model.generate plus completion torch.cuda.synchronize",
            "synchronize_before_start": True, "synchronize_before_stop": True,
            "inputs_already_on_gpu": True,
            "excluded": ["model_load", "input_transfer", "generation_config_construction",
                         "warmup", "gc_collect", "output_validation", "cpu_output_copy",
                         "tokenizer_and_decoding", "throughput_calculation", "result_serialization",
                         "baseline_memory_reads", "peak_memory_reset", "peak_memory_reads"],
        },
        "warmup_runs": [], "repeat_runs": [],
    }
    try:
        config = load_pilot_config()
        configure_runtime(config)
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 is required")
        source = json.loads((PROJECT_ROOT / config["model_source_path"]).read_text())
        report.update(config=config, gpu_name=torch.cuda.get_device_name(0), input_hash_verified=True)
        print("Loading local BF16/eager model; warm-up x4 then alternating repeats x10.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            PROJECT_ROOT / source["local_path"], **model_loading_kwargs(config),
        ).to(config["device"]).eval()
        fixed = load_fixed_input(PROJECT_ROOT / config["input_path"])
        input_ids = torch.tensor([fixed["token_ids"]], dtype=torch.long, device=config["device"])
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        run_warmup(model, inputs, config, records=report["warmup_runs"])
        require(len(report["warmup_runs"]) == 4 and all(r["status"] == "passed" for r in report["warmup_runs"]),
                "All four warm-ups must pass before repeats")
        require(not model._forward_hooks, "Warm-up diagnostic hook is still installed")
        report.update(warmup_same_model_and_process=True, warmup_diagnostic_hooks_removed=True)
        run_pilot_repeats(model, inputs, config, records=report["repeat_runs"])
        report.update(status="passed", performance_metrics_recorded=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["performance_metrics_recorded"] = any(
            run.get("performance_metrics_recorded", False) for run in report["repeat_runs"]
        )
        RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Result: {RESULT_PATH}", flush=True)
    print("PASS: latency, throughput and memory recorded for OFF/ON five runs each; warm-up excluded.", flush=True)


if __name__ == "__main__":
    main()
