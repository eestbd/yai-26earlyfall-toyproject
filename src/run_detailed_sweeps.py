"""Measure prefill, decode, total, decode throughput, peak allocation and actual KV bytes."""
import gc
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from pilot_config import PROJECT_ROOT, configure_runtime, load_pilot_config, model_loading_kwargs, require
from validate_fixed_input import load_fixed_input, token_ids_sha256
from summarize_pilot import compare_tokens

import torch
from transformers import AutoModelForCausalLM, DynamicCache

SETTINGS = PROJECT_ROOT/'docs/detailed_sweeps_config.json'
RESULT = PROJECT_ROOT/'results/detailed_sweeps_raw.json'


def advance(model, seq, mask, token, cache, mode, eos, diagnostic, step, prompt):
    model_input = seq if not mode or step == 0 else token
    output = model(input_ids=model_input, attention_mask=mask, past_key_values=cache,
                   use_cache=mode, num_logits_to_keep=1, return_dict=True,
                   output_attentions=False, output_hidden_states=False)
    scores = output.logits[:, -1, :].clone().float()
    if diagnostic:
        require(torch.is_inference_mode_enabled(), 'Inference mode required')
        require(model_input.shape[1] == (prompt + step if not mode else prompt if step == 0 else 1), 'Wrong cache path')
        require((output.past_key_values is not None) is mode, 'Unexpected cache presence')
        require(torch.isfinite(scores).all().item(), 'Non-finite logits')
    scores[:, eos] = -float('inf')
    next_token = torch.argmax(scores, dim=-1).unsqueeze(-1)
    next_cache = output.past_key_values
    seq = torch.cat([seq, next_token], dim=-1)
    mask = torch.cat([mask, mask.new_ones((1, 1))], dim=-1)
    return seq, mask, next_token, next_cache


def inspect_cache(cache, model, prompt, generated, mode):
    layers = list(cache) if cache is not None else []
    if not mode:
        require(not layers, 'OFF returned a cache')
        return {'actual_kv_tensor_bytes':0, 'kv_cache_layers':0, 'kv_cached_tokens':0,
                'first_key_shape':None, 'first_value_shape':None}
    config = model.config
    require(len(layers) == config.num_hidden_layers, 'Incorrect cache layer count')
    shape = [1, config.num_key_value_heads, prompt + generated - 1,
             config.hidden_size // config.num_attention_heads]
    for key, value in layers:
        require(list(key.shape) == list(value.shape) == shape, 'Incorrect cache shape')
        require(key.dtype == value.dtype == torch.bfloat16 and key.device == value.device == model.device,
                'Incorrect cache dtype/device')
    size = sum(t.numel()*t.element_size() for pair in layers for t in pair)
    return {'actual_kv_tensor_bytes':size, 'kv_cache_layers':len(layers),
            'kv_cached_tokens':shape[2], 'first_key_shape':shape, 'first_value_shape':shape}


def run_one(model, input_ids, attention_mask, base, record, reference):
    mode, warm = record['use_cache'], record['warmup']
    prompt, generated = record['prompt_tokens'], record['requested_new_tokens']
    device = input_ids.device
    cache = DynamicCache() if mode else None
    seq, mask, token = input_ids, attention_mask, None
    try:
        gc.collect()
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        if not warm:
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            if not warm:
                start = time.perf_counter()
            seq, mask, token, cache = advance(model, seq, mask, token, cache, mode,
                                             base['generation']['eos_token_id'], warm, 0, prompt)
            torch.cuda.synchronize(device)
            if not warm:
                boundary = time.perf_counter()
            for step in range(1, generated):
                seq, mask, token, cache = advance(model, seq, mask, token, cache, mode,
                                                 base['generation']['eos_token_id'], warm, step, prompt)
            torch.cuda.synchronize(device)
            if not warm:
                end = time.perf_counter()
                peak = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)
        # Read sizes from actual retained tensors after timers and peak reads.
        cache_info = inspect_cache(cache, model, prompt, generated, mode)
        require(list(seq.shape) == [1,prompt+generated], 'Generated count mismatch')
        require(torch.equal(seq[:,:prompt], input_ids), 'Input prefix changed')
        ids = seq[0,prompt:].tolist()
        comparison = compare_tokens(reference, ids)
        record.update(actual_generated_tokens=len(ids), decode_tokens=len(ids)-1,
                      generated_token_ids=ids, generated_token_ids_sha256=token_ids_sha256(ids),
                      generate_reference_comparison=comparison, **cache_info)
        require(comparison['equal'], 'Manual output differs from prior generate reference; see mismatch record')
        if warm:
            record.update(finite_logits_steps=generated, cache_path_verified=True)
        else:
            prefill, decode, total = boundary-start, end-boundary, end-start
            require(all(math.isfinite(v) and v>0 for v in [prefill,decode,total]), 'Invalid timing')
            require(0<baseline<=peak<=peak_reserved and baseline<=reserved<=peak_reserved, 'Invalid memory readings')
            record.update(prefill_latency_ms=prefill*1000, decode_latency_ms=decode*1000,
                          total_latency_ms=total*1000, decode_throughput_tokens_per_second=(len(ids)-1)/decode,
                          peak_allocated_bytes=peak, peak_allocated_mib=peak/2**20,
                          actual_kv_tensor_mib=cache_info['actual_kv_tensor_bytes']/2**20,
                          baseline_allocated_bytes=baseline, baseline_reserved_bytes=reserved,
                          peak_reserved_bytes=peak_reserved)
        record['status']='passed'
    except Exception as exc:
        record.update(status='failed',included_in_statistics=False,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        del seq,mask,token,cache
    detail = 'logits/cache/reference OK' if warm else f"prefill={record['prefill_latency_ms']:.2f}ms decode={record['decode_latency_ms']:.2f}ms KV={record['actual_kv_tensor_mib']:.3f}MiB"
    print(f"{record['sweep']} P={prompt} G={generated} {'warm-up' if warm else 'measured'} "
          f"{record['repeat']} {'ON' if mode else 'OFF'}: {detail}",flush=True)


def main():
    settings=json.loads(SETTINGS.read_text())
    base=load_pilot_config()
    require(settings['prompt_lengths']==[128,256,512,1024] and settings['generation_lengths']==[32,64,128,256], 'Length settings mismatch')
    require(settings['prompt_sweep_generation_tokens']==settings['generation_sweep_prompt_tokens']==128, 'Fixed lengths mismatch')
    require(settings['warmup_runs_per_condition']==2 and settings['measurement_repeats_per_condition']==5, 'Repeat counts mismatch')
    require(settings['num_logits_to_keep']==1 and settings['empty_cache_between_runs'] is False, 'Runtime policy mismatch')
    require(settings['sweep_order']==['prompt','generation'] and settings['length_order']=='ascending', 'Execution order mismatch')
    for path,digest in settings['input_file_sha256'].items():
        require(hashlib.sha256((PROJECT_ROOT/path).read_bytes()).hexdigest()==digest, f'Input hash mismatch: {path}')
    fixed=load_fixed_input(PROJECT_ROOT/settings['fixed_input_path'])
    manifest=json.loads((PROJECT_ROOT/settings['prompt_input_manifest_path']).read_text())
    inputs_by_length={e['prompt_tokens']:e for e in manifest['inputs']}
    for n in settings['prompt_lengths']:
        e=inputs_by_length[n]
        require(e['token_ids']==fixed['token_ids']*(n//128) and token_ids_sha256(e['token_ids'])==e['sha256'], 'Prompt input mismatch')
    references={name:json.loads((PROJECT_ROOT/path).read_text()) for name,path in settings['reference_results'].items()}
    require(all(r['status']=='passed' for r in references.values()), 'Prior reference incomplete')
    configure_runtime(base)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 required')
    source=json.loads((PROJECT_ROOT/base['model_source_path']).read_text())
    report={'status':'running','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'six_metric_manual_sweeps','base_config':base,'settings':settings,
            'settings_sha256':hashlib.sha256(SETTINGS.read_bytes()).hexdigest(),
            'gpu_name':torch.cuda.get_device_name(0),
            'source_sha256':{name:hashlib.sha256((PROJECT_ROOT/'src'/name).read_bytes()).hexdigest()
                             for name in ['run_detailed_sweeps.py','pilot_config.py','validate_fixed_input.py','summarize_pilot.py','calculate_throughput.py']},
            'sweeps':{'prompt':{'conditions':[]},'generation':{'conditions':[]}}}
    try:
        print('Loading one BF16/eager model for both six-metric sweeps.',flush=True)
        model=AutoModelForCausalLM.from_pretrained(PROJECT_ROOT/source['local_path'],**model_loading_kwargs(base)).to(base['device']).eval()
        require(all(p.dtype==torch.bfloat16 and p.device==torch.device(base['device']) for p in model.parameters()),'Parameter mismatch')
        require(model.config._attn_implementation=='eager' and not any(m.training for m in model.modules()),'Model mode mismatch')
        require(not getattr(model,'is_quantized',False),'Quantization disabled')
        for sweep in settings['sweep_order']:
            lengths=settings['prompt_lengths' if sweep=='prompt' else 'generation_lengths']
            for length in lengths:
                configure_runtime(base)
                prompt,generated=(length,128) if sweep=='prompt' else (128,length)
                e=inputs_by_length[prompt]
                reference_condition=next(c for c in references[sweep]['conditions'] if c['prompt_tokens']==prompt and
                                         (c.get('generation_tokens',128)==generated))
                reference_ids={mode:next(r['generated_token_ids'] for r in reference_condition['repeat_runs'] if r['use_cache'] is mode) for mode in [False,True]}
                c={'prompt_tokens':prompt,'generation_tokens':generated,'input_sha256':e['sha256'],'warmup_runs':[],'repeat_runs':[]}
                report['sweeps'][sweep]['conditions'].append(c)
                input_ids=torch.tensor([e['token_ids']],dtype=torch.long,device=base['device'])
                mask=torch.ones_like(input_ids)
                schedule=[(True,rep,pos,mode) for pos,mode in enumerate([False,True]) for rep in range(2)]
                schedule += [(False,rep,pos,mode) for rep,order in enumerate(base['measurement_order']) for pos,mode in enumerate(order)]
                for warm,rep,pos,mode in schedule:
                    records=c['warmup_runs' if warm else 'repeat_runs']
                    record={'sweep':sweep,'execution_index':len(records),'prompt_tokens':prompt,'requested_new_tokens':generated,
                            'repeat':rep,'position_in_repeat':pos,'use_cache':mode,'warmup':warm,'included_in_statistics':not warm,'status':'running'}
                    records.append(record)
                    run_one(model,input_ids,mask,base,record,reference_ids[mode])
                first=c['repeat_runs'][0]['generated_token_ids']
                c['output_comparisons']=[dict(execution_index=r['execution_index'],**compare_tokens(first,r['generated_token_ids'])) for r in c['repeat_runs']]
                c['all_generated_token_ids_equal']=all(x['equal'] for x in c['output_comparisons'])
                del input_ids,mask
                gc.collect(); torch.cuda.synchronize()
                RESULT.write_text(json.dumps(report,indent=2)+'\n')
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        RESULT.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Completed 32 warm-ups and 80 measured runs: {RESULT}',flush=True)


if __name__=='__main__':
    main()
