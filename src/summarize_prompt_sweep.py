"""Validate and summarize the prompt sweep without GPU execution."""
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from calculate_throughput import calculate_throughput
from summarize_pilot import CSV_FIELDS, METRICS, compare_tokens, require
from validate_fixed_input import token_ids_sha256

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results'/'prompt_sweep_raw.json'
CSV=ROOT/'results'/'prompt_sweep_raw.csv'
SUMMARY=ROOT/'results'/'prompt_sweep_summary.json'


def main():
    raw=json.loads(RAW.read_text())
    require(raw['status']=='passed','Sweep did not complete')
    settings=raw['sweep_settings']
    require([c['prompt_tokens'] for c in raw['conditions']]==[128,256,512,1024],'Missing conditions')
    require(hashlib.sha256((ROOT/'docs/prompt_sweep_config.json').read_bytes()).hexdigest()==raw['sweep_settings_sha256'],'Sweep config hash mismatch')
    for path,key in [(settings['base_config_path'],'base_config_sha256'),(settings['input_manifest_path'],'input_manifest_sha256')]:
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==settings[key],f'Hash mismatch: {path}')
    for name,digest in raw['source_sha256'].items():
        require(hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==digest,f'Source changed: {name}')
    manifest=json.loads((ROOT/settings['input_manifest_path']).read_text())
    report={'status':'passed','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'raw_path':RAW.relative_to(ROOT).as_posix(),'raw_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'sample_std_ddof':1,'outliers_removed':False,'warmups_excluded':16,'measured_runs':40,
            'synthetic_repeated_input':True,'generation_tokens':128,'conditions':[]}
    rows=[]
    expected=[(rep,pos,mode) for rep,order in enumerate([[False,True],[True,False],[False,True],[True,False],[False,True]]) for pos,mode in enumerate(order)]
    for c,entry in zip(raw['conditions'],manifest['inputs']):
        require(c['input_sha256']==entry['sha256']==token_ids_sha256(entry['token_ids']),'Input hash mismatch')
        warm=c['warmup_runs']; runs=c['repeat_runs']
        require(len(warm)==4 and [r['use_cache'] for r in warm]==[False,False,True,True],'Warm-up mismatch')
        require(all(r['status']=='passed' and r['warmup'] and not r['included_in_statistics'] and r['diagnostic_hook_removed'] and r['finite_logits_steps']==128 and r['cache_path_verified'] for r in warm),'Warm-up validation failed')
        require(len(runs)==10 and [(r['repeat'],r['position_in_repeat'],r['use_cache']) for r in runs]==expected,'Measurement schedule mismatch')
        require([r['execution_index'] for r in runs]==list(range(10)),'Execution indices mismatch')
        for r in warm+runs:
            require(r['status']=='passed' and r['actual_generated_tokens']==len(r['generated_token_ids'])==r['requested_new_tokens']==128,'Token count mismatch')
            require(r['prompt_tokens']==c['prompt_tokens'],'Prompt count mismatch')
            require(token_ids_sha256(r['generated_token_ids'])==r['generated_token_ids_sha256'],'Output hash mismatch')
        for r in runs:
            require(not r['warmup'] and r['included_in_statistics'],'Invalid sample selection')
            require(math.isclose(calculate_throughput(128,r['total_latency_seconds']),r['throughput_tokens_per_second'],rel_tol=1e-12),'Throughput mismatch')
            require(0<r['baseline_allocated_bytes']<=r['peak_allocated_bytes']<=r['peak_reserved_bytes'],'Invalid allocated readings')
            require(r['baseline_allocated_bytes']<=r['baseline_reserved_bytes']<=r['peak_reserved_bytes'],'Invalid reserved readings')
            require(r['peak_allocated_above_baseline_bytes']==r['peak_allocated_bytes']-r['baseline_allocated_bytes'],'Invalid delta')
            rows.append(r)
        stats={name:{metric:{'n':5,'mean':statistics.mean(r[metric] for r in runs if r['use_cache'] is mode),'std':statistics.stdev(r[metric] for r in runs if r['use_cache'] is mode)} for metric in METRICS} for name,mode in [('OFF',False),('ON',True)]}
        comparisons=[dict(execution_index=r['execution_index'],**compare_tokens(runs[0]['generated_token_ids'],r['generated_token_ids'])) for r in runs]
        equal=all(x['equal'] for x in comparisons)
        require(comparisons==c['output_comparisons'] and equal==c['all_generated_token_ids_equal'],'Output comparison mismatch')
        result={'prompt_tokens':c['prompt_tokens'],'input_sha256':c['input_sha256'],'statistics':stats,
                'all_generated_token_ids_equal':equal,'output_comparisons':comparisons,
                'speedup':stats['OFF']['total_latency_seconds']['mean']/stats['ON']['total_latency_seconds']['mean']}
        report['conditions'].append(result)
    report['all_conditions_outputs_equal']=all(c['all_generated_token_ids_equal'] for c in report['conditions'])
    report['comparison_caveat']=None if report['all_conditions_outputs_equal'] else 'Some cache modes generated different token IDs; see first mismatch diagnostics before interpreting timings.'
    with CSV.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=CSV_FIELDS,lineterminator='\n'); writer.writeheader()
        for r in rows:
            row={k:r[k] for k in CSV_FIELDS}; row['generated_token_ids']=json.dumps(row['generated_token_ids']); writer.writerow(row)
    report['csv_sha256']=hashlib.sha256(CSV.read_bytes()).hexdigest()
    SUMMARY.write_text(json.dumps(report,indent=2)+'\n')
    for c in report['conditions']:
        print(f"P={c['prompt_tokens']}: OFF {c['statistics']['OFF']['total_latency_seconds']['mean']:.4f}s; ON {c['statistics']['ON']['total_latency_seconds']['mean']:.4f}s; speedup {c['speedup']:.3f}x; identical outputs={c['all_generated_token_ids_equal']}")
    print(f'Validated 40 samples; summary: {SUMMARY}')


if __name__=='__main__':
    main()
