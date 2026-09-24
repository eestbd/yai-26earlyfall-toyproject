"""Derive generated-token throughput from saved latency samples without a GPU run."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "results" / "pilot_latency.json"
RESULT_PATH = PROJECT_ROOT / "results" / "pilot_throughput.json"
THROUGHPUT_DEFINITION = {
    "formula": "actual_generated_tokens / total_latency_seconds",
    "unit": "tokens/second",
    "prompt_tokens_in_numerator": False,
    "latency_includes_prefill_and_decode": True,
}


def calculate_throughput(actual_generated_tokens, total_latency_seconds):
    """Use the actual new-token count, never prompt length or requested length."""
    if type(actual_generated_tokens) is not int or actual_generated_tokens <= 0:
        raise ValueError("Actual generated tokens must be a positive integer")
    if (type(total_latency_seconds) not in (int, float)
            or not math.isfinite(total_latency_seconds) or total_latency_seconds <= 0):
        raise ValueError("Latency must be finite and positive, in seconds")
    throughput = actual_generated_tokens / total_latency_seconds
    if not math.isfinite(throughput) or throughput <= 0:
        raise ValueError("Throughput must be finite and positive")
    return throughput


def main():
    source_bytes = SOURCE_PATH.read_bytes()
    report = json.loads(source_bytes)
    if report["status"] != "passed" or report["scope"] != "step_13_total_generation_latency":
        raise ValueError("Expected successful step-13 latency measurements")
    runs = report["repeat_runs"]
    config = report["config"]
    expected = [(repeat, position, mode)
                for repeat, order in enumerate(config["measurement_order"])
                for position, mode in enumerate(order)]
    if [(r["repeat"], r["position_in_repeat"], r["use_cache"]) for r in runs] != expected:
        raise ValueError("Measurement schedule mismatch")
    for run in runs:
        if run["status"] != "passed" or run["warmup"] or not run["included_in_statistics"]:
            raise ValueError("Throughput requires a valid non-warm-up sample")
        if run["actual_generated_tokens"] != len(run["generated_token_ids"]):
            raise ValueError("Actual generated-token count does not match saved output")
        run["throughput_tokens_per_second"] = calculate_throughput(
            run["actual_generated_tokens"], run["total_latency_seconds"],
        )
    if any(run["included_in_statistics"] for run in report["warmup_runs"]):
        raise ValueError("Warm-up must remain excluded from statistics")
    # Preserve measurement timestamps and source hashes as provenance of the GPU run.
    report["scope"] = "step_14_throughput_from_step_13_latency"
    report["derived_metrics"] = ["throughput_tokens_per_second"]
    report["throughput_definition"] = THROUGHPUT_DEFINITION
    report["derivation"] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": SOURCE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "source_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generation_reexecuted": False,
        "source_sha256_describes": "original step-13 measurement code",
    }
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for run in runs:
        print(f"repeat {run['repeat']} {'ON' if run['use_cache'] else 'OFF'}: "
              f"{run['actual_generated_tokens']} / {run['total_latency_seconds']:.6f} s = "
              f"{run['throughput_tokens_per_second']:.6f} tokens/s")
    print(f"PASS: throughput derived for {len(runs)} samples; no GPU rerun. Result: {RESULT_PATH}")


if __name__ == "__main__":
    main()
