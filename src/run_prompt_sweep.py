"""Fixed-length prompt sweep, preserving the original pilot and its source hashes."""

import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from pilot_config import PROJECT_ROOT, load_pilot_config, configure_runtime, generation_config_for, model_loading_kwargs, require
from validate_fixed_input import load_fixed_input, token_ids_sha256
from calculate_throughput import calculate_throughput
from summarize_pilot import compare_tokens

import torch
from transformers import AutoModelForCausalLM

SETTINGS = PROJECT_ROOT / "docs" / "prompt_sweep_config.json"
RESULT = PROJECT_ROOT / "results" / "prompt_sweep_raw.json"


def run_one(model, inputs, config, record):
    device = torch.device(config["device"])
    prompt = record["prompt_tokens"]
    expected_ids = inputs["input_ids"].tolist()
    mode = record["use_cache"]
    warmup = record["warmup"]
    generation = generation_config_for(config, mode)
    outputs = None
    hook = None
    steps = 0

    def validate_forward(_model, args, kwargs, output):
        nonlocal steps
        expected_length = prompt if steps == 0 else (1 if mode else prompt + steps)
        require(kwargs["input_ids"].shape[1] == expected_length, "Unexpected cache execution path")
        require(kwargs.get("use_cache") is mode and (output.past_key_values is not None) is mode,
                "Actual cache usage mismatch")
        require(torch.isfinite(output.logits[:, -1, :]).all().item(), "Non-finite next-token logits")
        steps += 1

    try:
        gc.collect()
        torch.cuda.synchronize(device)
        if warmup:
            hook = model.register_forward_hook(validate_forward, with_kwargs=True)
        else:
            require(not model._forward_hooks, "Diagnostic hook remains installed")
        with torch.inference_mode():
            if not warmup:
                baseline = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
            outputs = model.generate(**inputs, generation_config=generation)
            torch.cuda.synchronize(device)
            if not warmup:
                latency = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)
        require(list(outputs.shape) == [1, prompt + 128], "Expected exactly 128 new tokens")
        require(torch.equal(outputs[:, :prompt], inputs["input_ids"]), "Prompt prefix changed")
        require(inputs["input_ids"].tolist() == expected_ids, "Input changed")
        require(torch.equal(inputs["attention_mask"], torch.ones_like(inputs["input_ids"])), "Mask changed")
        ids = outputs[0, prompt:].tolist()
        record.update(status="passed", actual_generated_tokens=len(ids), generated_token_ids=ids,
                      generated_token_ids_sha256=token_ids_sha256(ids))
        if warmup:
            require(steps == 128, "Unexpected warm-up step count")
            record.update(finite_logits_steps=steps, cache_path_verified=True)
        else:
            require(math.isfinite(latency) and latency > 0, "Invalid latency")
            require(0 < baseline <= peak <= peak_reserved and baseline <= reserved <= peak_reserved,
                    "Invalid allocator readings")
            record.update(total_latency_seconds=latency,
                          throughput_tokens_per_second=calculate_throughput(len(ids), latency),
                          baseline_allocated_bytes=baseline, baseline_reserved_bytes=reserved,
                          peak_allocated_bytes=peak, peak_reserved_bytes=peak_reserved,
                          peak_allocated_above_baseline_bytes=peak-baseline)
    except Exception as exc:
        record.update(status="failed", included_in_statistics=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if hook is not None:
            hook.remove()
            record["diagnostic_hook_removed"] = True
        del outputs
    suffix = "finite logits / cache path OK" if warmup else f"{latency:.4f} s, {record['throughput_tokens_per_second']:.2f} tokens/s"
    print(f"P={prompt} {'warm-up' if warmup else 'measured'} {record['repeat']} "
          f"{'ON' if mode else 'OFF'}: {suffix}", flush=True)


def main():
    base = load_pilot_config()
    settings = json.loads(SETTINGS.read_text())
    require(settings["prompt_lengths"] == [128,256,512,1024] and settings["generation_tokens"] == 128,
            "Unexpected sweep conditions")
    require(settings["warmup_runs_per_condition_per_length"] == 2
            and settings["measurement_repeats_per_condition_per_length"] == 5, "Unexpected counts")
    require(settings["length_order"] == "ascending" and settings["synthetic_repeated_input"] is True
            and settings["empty_cache_between_runs_or_lengths"] is False, "Unexpected execution policy")
    base_path = PROJECT_ROOT / settings["base_config_path"]
    require(hashlib.sha256(base_path.read_bytes()).hexdigest() == settings["base_config_sha256"], "Base config hash mismatch")
    manifest_path = PROJECT_ROOT / settings["input_manifest_path"]
    require(hashlib.sha256(manifest_path.read_bytes()).hexdigest() == settings["input_manifest_sha256"], "Input manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    fixed = load_fixed_input()
    require(manifest["source_token_ids_sha256"] == fixed["sha256"], "Source input hash mismatch")
    require([entry["prompt_tokens"] for entry in manifest["inputs"]] == settings["prompt_lengths"], "Input lengths mismatch")
    for entry in manifest["inputs"]:
        require(entry["token_ids"] == fixed["token_ids"] * (entry["prompt_tokens"] // 128), "Unexpected synthetic input")
        require(token_ids_sha256(entry["token_ids"]) == entry["sha256"], "Input token hash mismatch")
    configure_runtime(base)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
    source = json.loads((PROJECT_ROOT / base["model_source_path"]).read_text())
    report = {
        "status":"running", "scope":"prompt_length_sweep_generation_128",
        "recorded_at_utc":datetime.now(timezone.utc).isoformat(), "base_config":base,
        "sweep_settings":settings, "gpu_name":torch.cuda.get_device_name(0),
        "source_sha256":{name:hashlib.sha256((PROJECT_ROOT/'src'/name).read_bytes()).hexdigest()
                         for name in ['run_prompt_sweep.py','pilot_config.py','validate_fixed_input.py','calculate_throughput.py','summarize_pilot.py']},
        "sweep_settings_sha256":hashlib.sha256(SETTINGS.read_bytes()).hexdigest(),
        "timing":"synchronize -> perf_counter -> generate -> synchronize -> perf_counter; memory reads and validation outside timer",
        "memory_policy":"per-run peaks reset; allocator cache retained across runs and ascending lengths",
        "conditions":[],
    }
    try:
        print('Loading one BF16/eager model for all four lengths.',flush=True)
        model = AutoModelForCausalLM.from_pretrained(PROJECT_ROOT/source['local_path'], **model_loading_kwargs(base)).to(base['device']).eval()
        require(all(p.device == torch.device(base['device']) and p.dtype == torch.bfloat16 for p in model.parameters()), "Parameter placement mismatch")
        require(not any(m.training for m in model.modules()) and model.config._attn_implementation == 'eager', "Model mode mismatch")
        require(not getattr(model,'is_quantized',False), "Quantization disabled")
        for entry in manifest['inputs']:
            configure_runtime(base)
            prompt = entry['prompt_tokens']
            require(prompt+128 <= model.config.max_position_embeddings, "Context exceeds model limit")
            condition = {'prompt_tokens':prompt, 'input_sha256':entry['sha256'], 'warmup_runs':[], 'repeat_runs':[]}
            report['conditions'].append(condition)
            ids = torch.tensor([entry['token_ids']],dtype=torch.long,device=base['device'])
            inputs = {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
            schedule = [(True,rep,pos,mode) for pos,mode in enumerate([False,True]) for rep in range(2)]
            schedule += [(False,rep,pos,mode) for rep,order in enumerate(base['measurement_order']) for pos,mode in enumerate(order)]
            for warm,rep,pos,mode in schedule:
                records = condition['warmup_runs' if warm else 'repeat_runs']
                record = {'execution_index':len(records),'prompt_tokens':prompt,'requested_new_tokens':128,
                          'repeat':rep,'position_in_repeat':pos,'use_cache':mode,'warmup':warm,
                          'included_in_statistics':not warm,'status':'running'}
                records.append(record)
                run_one(model, inputs, base, record)
            reference = condition['repeat_runs'][0]['generated_token_ids']
            condition['output_comparisons'] = [dict(execution_index=r['execution_index'],**compare_tokens(reference,r['generated_token_ids'])) for r in condition['repeat_runs']]
            condition['all_generated_token_ids_equal'] = all(r['equal'] for r in condition['output_comparisons'])
            del inputs,ids
            gc.collect()
            torch.cuda.synchronize()
            RESULT.write_text(json.dumps(report,indent=2)+'\n')
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        RESULT.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Completed 16 warm-ups and 40 measurements: {RESULT}',flush=True)


if __name__ == '__main__':
    main()
