"""Plot validated pilot samples without loading a model or rerunning inference."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Keep Matplotlib's small font/config cache away from the inode-constrained /root.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kv-cache-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "pilot_summary.json"
RAW = ROOT / "results" / "pilot_memory.json"


def main():
    summary = json.loads(SUMMARY.read_text())
    raw = json.loads(RAW.read_text())
    if summary["status"] != "passed" or not summary["validation"]["all_generated_token_ids_equal"]:
        raise ValueError("Plot requires validated, matching outputs")
    if hashlib.sha256(RAW.read_bytes()).hexdigest() != summary["provenance"]["raw_json_sha256"]:
        raise ValueError("Raw measurements differ from the summary source")
    modes = [("OFF", False, "#52657A"), ("ON", True, "#168A7C")]
    metrics = [
        ("total_latency_seconds", "Total generation latency", "Seconds (lower is better)", 1),
        ("throughput_tokens_per_second", "Generated-token throughput", "Tokens/s (higher is better)", 1),
        ("peak_allocated_bytes", "Peak allocated memory", "MiB (includes resident model)", 2**20),
        ("peak_allocated_above_baseline_bytes", "Allocated peak above baseline", "MiB (not KV-cache size)", 2**20),
    ]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.7))
    for ax, (metric, title, ylabel, divisor) in zip(axes.flat, metrics):
        upper = 0
        annotations = []
        for position, (name, use_cache, color) in enumerate(modes):
            samples = [r[metric] / divisor for r in raw["repeat_runs"] if r["use_cache"] is use_cache]
            if len(samples) != 5:
                raise ValueError("Expected five samples per mode")
            stat = summary["statistics"][name][metric]
            mean, std = stat["mean"] / divisor, stat["std"] / divisor
            ax.bar(position, mean, width=0.5, color=color, alpha=0.18, zorder=2)
            ax.scatter([position + shift for shift in [-0.12, -0.06, 0, 0.06, 0.12]],
                       samples, color=color, s=28, zorder=4)
            ax.errorbar(position, mean, yerr=std, fmt="D", color="#162A3B", markersize=5,
                        capsize=6, linewidth=1.5, zorder=5)
            upper = max(upper, max(samples), mean + std)
            annotations.append((position, max(max(samples), mean + std), mean, std))
        for position, top, mean, std in annotations:
            ax.text(position, top + upper * 0.055, f"{mean:.3f} ± {std:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set(xticks=[0, 1], xticklabels=["Cache OFF", "Cache ON"], ylabel=ylabel,
               title=title, ylim=(0, upper * 1.23), xlim=(-0.5, 1.5))
        ax.grid(axis="y", alpha=0.18, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("KV Cache Pilot | Qwen2.5-1.5B", fontsize=20, fontweight="bold", y=0.985)
    fig.text(0.5, 0.936, "RTX 3090 · BF16 · eager · batch 1 · prompt 128 · generation 128 · greedy",
             ha="center", fontsize=11)
    baseline = summary["statistics"]["OFF"]["baseline_allocated_bytes"]["mean"] / 2**20
    reserved = summary["statistics"]["OFF"]["peak_reserved_bytes"]["mean"] / 2**20
    speedup = summary["speedup"]["value"]
    fig.text(0.5, 0.061, "Dots: 5 measured runs per mode. Diamond/error bars: mean ± sample SD (ddof=1). All axes start at zero.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.038, f"Both modes: allocated baseline {baseline:.2f} MiB; peak reserved {reserved:.0f} MiB. Warm-up excluded; outputs identical.",
             ha="center", fontsize=9)
    fig.text(0.5, 0.015, f"Latency speedup (OFF mean / ON mean): {speedup:.4f}×. No speed gain observed in this pilot; descriptive results only.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=0.875, bottom=0.15, left=0.09, right=0.975, hspace=0.42, wspace=0.3)
    paths = [ROOT / "plots" / "pilot_results.png", ROOT / "plots" / "pilot_results.pdf"]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    metadata = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "matplotlib_version": matplotlib.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [SUMMARY, RAW]},
        "outputs": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "gpu_generation_reexecuted": False, "error_bars": "sample standard deviation, ddof=1",
    }
    (ROOT / "results" / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Plots saved: {', '.join(str(p) for p in paths)}")
    print(f"Used installed Matplotlib {matplotlib.__version__}; no packages installed.")


if __name__ == "__main__":
    main()
