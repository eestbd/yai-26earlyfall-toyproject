"""Validate step-15 samples and export raw CSV plus descriptive statistics."""

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from calculate_throughput import calculate_throughput
from validate_fixed_input import load_fixed_input, token_ids_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "pilot_memory.json"
CSV_PATH = ROOT / "results" / "pilot_raw.csv"
SUMMARY_PATH = ROOT / "results" / "pilot_summary.json"
MEMORY_FIELDS = [
    "baseline_allocated_bytes", "baseline_reserved_bytes", "peak_allocated_bytes",
    "peak_reserved_bytes", "peak_allocated_above_baseline_bytes",
]
METRICS = ["total_latency_seconds", "throughput_tokens_per_second", *MEMORY_FIELDS]
CSV_FIELDS = [
    "execution_index", "repeat", "position_in_repeat", "use_cache", "warmup",
    "status", "included_in_statistics", "prompt_tokens", "requested_new_tokens",
    "actual_generated_tokens", *METRICS, "generated_token_ids_sha256", "generated_token_ids",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_tokens(left, right):
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a != b:
            return {"equal": False, "first_mismatch": {
                "generated_token_index_zero_based": index, "left_token_id": a, "right_token_id": b,
            }}
    return {"equal": True, "first_mismatch": None}


def analyze(source):
    require(source["status"] == "passed" and source["scope"] == "step_15_latency_throughput_and_memory",
            "Expected successful step-15 measurements")
    config = source["config"]
    runs = source["repeat_runs"]
    warmups = source["warmup_runs"]
    require(len(runs) == 10 and config["measurement_repeats_per_condition"] == 5, "Expected ten samples")
    expected_order = [[False, True], [True, False], [False, True], [True, False], [False, True]]
    require(config["measurement_order"] == expected_order, "Unexpected schedule configuration")
    expected = [(i, j, mode) for i, order in enumerate(expected_order) for j, mode in enumerate(order)]
    require([(r["repeat"], r["position_in_repeat"], r["use_cache"]) for r in runs] == expected,
            "Recorded schedule mismatch")
    require([r["execution_index"] for r in runs] == list(range(10)), "Execution index mismatch")
    require([r["use_cache"] for r in warmups] == [False, False, True, True], "Warm-up schedule mismatch")
    require(all(r["status"] == "passed" and r["warmup"] is True
                and r["included_in_statistics"] is False and r["diagnostic_hook_removed"]
                for r in warmups), "Invalid warm-up record")
    require(source["warmup_same_model_and_process"] and source["warmup_diagnostic_hooks_removed"],
            "Warm-up execution requirements were not met")
    for r in runs:
        require(type(r["use_cache"]) is bool and r["status"] == "passed"
                and r["warmup"] is False and r["included_in_statistics"] is True,
                "Invalid measurement sample")
        ids = r["generated_token_ids"]
        require(all(type(token) is int and token >= 0 for token in ids), "Invalid token IDs")
        require(r["prompt_tokens"] == config["prompt_tokens"] == 128, "Prompt count mismatch")
        require(len(ids) == r["actual_generated_tokens"] == r["requested_new_tokens"]
                == config["generation"]["max_new_tokens"] == 128, "Generated-token count mismatch")
        require(token_ids_sha256(ids) == r["generated_token_ids_sha256"], "Output token hash mismatch")
        rate = calculate_throughput(r["actual_generated_tokens"], r["total_latency_seconds"])
        require(math.isclose(rate, r["throughput_tokens_per_second"], rel_tol=1e-12), "Throughput mismatch")
        require(all(type(r[k]) is int and r[k] >= 0 for k in MEMORY_FIELDS), "Invalid memory values")
        require(0 < r["baseline_allocated_bytes"] <= r["peak_allocated_bytes"] <= r["peak_reserved_bytes"],
                "Allocated memory ordering mismatch")
        require(r["baseline_allocated_bytes"] <= r["baseline_reserved_bytes"] <= r["peak_reserved_bytes"],
                "Reserved memory ordering mismatch")
        require(r["peak_allocated_above_baseline_bytes"] == r["peak_allocated_bytes"] - r["baseline_allocated_bytes"],
                "Memory delta mismatch")
    groups = {name: [r for r in runs if r["use_cache"] is mode] for name, mode in [("OFF", False), ("ON", True)]}
    stats = {}
    for name, group in groups.items():
        require(len(group) == 5, f"Expected five {name} samples")
        stats[name] = {metric: {
            "n": len(group), "mean": statistics.mean(r[metric] for r in group),
            "std": statistics.stdev(r[metric] for r in group),
            "unit": "seconds" if metric == "total_latency_seconds" else
                    "tokens/second" if metric == "throughput_tokens_per_second" else "bytes",
        } for metric in METRICS}
    off = groups["OFF"][0]
    on = groups["ON"][0]
    pairs = [{"repeat": i, **compare_tokens(groups["OFF"][i]["generated_token_ids"],
                                          groups["ON"][i]["generated_token_ids"])} for i in range(5)]
    all_runs = [{"execution_index": r["execution_index"], "use_cache": r["use_cache"],
                 **compare_tokens(off["generated_token_ids"], r["generated_token_ids"])} for r in runs]
    equal = all(r["equal"] for r in all_runs)
    return {
        "status": "passed" if equal else "failed_output_equality",
        "validation": {
            "samples_valid": True, "measurement_runs": 10, "warmup_runs_excluded": 4,
            "actual_generated_tokens_each": 128, "all_generated_token_ids_equal": equal,
            "first_off_on_comparison": compare_tokens(off["generated_token_ids"], on["generated_token_ids"]),
            "paired_off_on_comparisons": pairs, "comparisons_to_first_off": all_runs,
            "first_valid_outputs": {name: {
                "execution_index": group[0]["execution_index"],
                "generated_token_ids": group[0]["generated_token_ids"],
                "generated_token_ids_sha256": group[0]["generated_token_ids_sha256"],
            } for name, group in groups.items()},
        },
        "statistics": stats,
        "statistics_definition": {
            "std": "sample standard deviation", "ddof": 1,
            "sample_source": "step-15 repeat_runs only", "outliers_removed": False,
            "throughput_mean": "arithmetic mean of per-run throughputs",
            "memory_values_are_bytes": True, "mib_divisor": 2**20,
        },
        "speedup": {
            "formula": "mean_OFF_latency / mean_ON_latency",
            "value": stats["OFF"]["total_latency_seconds"]["mean"] / stats["ON"]["total_latency_seconds"]["mean"],
            "greater_than_one_means": "ON is faster", "descriptive_only": True,
        },
    }


def main():
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    config_path = ROOT / "docs" / "pilot_config.json"
    require(hashlib.sha256(config_path.read_bytes()).hexdigest() == source["config_file_sha256"], "Config file hash mismatch")
    require(json.loads(config_path.read_text()) == source["config"], "Embedded config mismatch")
    fixed = load_fixed_input(ROOT / source["config"]["input_path"])
    require(fixed["sha256"] == source["config"]["input_sha256"], "Input hash mismatch")
    source_checks = {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() == digest
                     for name, digest in source["source_sha256"].items()}
    require(all(source_checks.values()), "Measurement code differs from recorded source hashes")
    report = analyze(source)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for run in source["repeat_runs"]:
            row = {key: run[key] for key in CSV_FIELDS}
            row["generated_token_ids"] = json.dumps(row["generated_token_ids"])
            writer.writerow(row)
    report.update({
        "scope": "step_16_validation_and_summary", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_recorded_at_utc": source["recorded_at_utc"],
        "config": source["config"], "gpu_name": source["gpu_name"],
        "provenance": {
            "raw_json_path": SOURCE.relative_to(ROOT).as_posix(),
            "raw_json_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "raw_csv_path": CSV_PATH.relative_to(ROOT).as_posix(),
            "raw_csv_sha256": hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
            "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "measurement_source_hashes_match": source_checks,
            "config_and_input_verified": True, "gpu_generation_reexecuted": False,
        },
    })
    SUMMARY_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Output equality: {report['validation']['all_generated_token_ids_equal']}")
    for name, stats in report["statistics"].items():
        print(name)
        for metric, values in stats.items():
            divisor = 2**20 if values["unit"] == "bytes" else 1
            unit = "MiB" if values["unit"] == "bytes" else values["unit"]
            print(f"  {metric}: {values['mean']/divisor:.6f} +/- {values['std']/divisor:.6f} {unit}")
    print(f"Latency speedup: {report['speedup']['value']:.6f}x")
    print(f"Raw CSV: {CSV_PATH}\nSummary: {SUMMARY_PATH}")
    require(report["status"] == "passed", "Output mismatch: see saved comparisons and first mismatch positions")


if __name__ == "__main__":
    main()
