"""Validate both six-metric sweeps and export per-sweep raw/summary CSV files."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from summarize_pilot import compare_tokens, require
from validate_fixed_input import token_ids_sha256

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/detailed_sweeps_raw.json'
SUMMARY=ROOT/'results/detailed_sweeps_summary.json'
METRICS={
    'prefill_latency_ms':'ms', 'decode_latency_ms':'ms', 'total_latency_ms':'ms',
    'decode_throughput_tokens_per_second':'tokens/s',
    'peak_allocated_mib':'MiB', 'actual_kv_tensor_mib':'MiB',
}
RAW_FIELDS=['sweep','execution_index','repeat','position_in_repeat','use_cache','warmup','included_in_statistics',
            'prompt_tokens','requested_new_tokens','actual_generated_tokens','decode_tokens',*METRICS,
            'baseline_allocated_bytes','baseline_reserved_bytes','peak_allocated_bytes','peak_reserved_bytes',
            'actual_kv_tensor_bytes','kv_cached_tokens','kv_cache_layers','generated_token_ids_sha256','generated_token_ids']


def summarize(raw):
    require(raw['status']=='passed','Incomplete measurement')
    expected=[(rep,pos,mode) for rep,order in enumerate([[False,True],[True,False],[False,True],[True,False],[False,True]]) for pos,mode in enumerate(order)]
    result={}
    for sweep in ['prompt','generation']:
        conditions=raw['sweeps'][sweep]['conditions']
        expected_shapes=[(p,128) for p in [128,256,512,1024]] if sweep=='prompt' else [(128,n) for n in [32,64,128,256]]
        require([(c['prompt_tokens'],c['generation_tokens']) for c in conditions]==expected_shapes,'Condition list mismatch')
        result[sweep]={'measured_runs':40,'warmup_runs_excluded':16,'conditions':[]}
        for c in conditions:
            p,n=c['prompt_tokens'],c['generation_tokens']
            warm,runs=c['warmup_runs'],c['repeat_runs']
            require(len(warm)==4 and [r['use_cache'] for r in warm]==[False,False,True,True],'Warm-up schedule mismatch')
            require(all(r['warmup'] is True and r['included_in_statistics'] is False and r['finite_logits_steps']==n and r['cache_path_verified'] for r in warm),'Warm-up invalid')
            require(len(runs)==10 and [(r['repeat'],r['position_in_repeat'],r['use_cache']) for r in runs]==expected,'Measurement schedule mismatch')
            require([r['execution_index'] for r in runs]==list(range(10)),'Execution index mismatch')
            for r in warm+runs:
                require(r['status']=='passed' and r['sweep']==sweep and r['prompt_tokens']==p,'Invalid run metadata')
                require(r['requested_new_tokens']==r['actual_generated_tokens']==len(r['generated_token_ids'])==n,'Generated count mismatch')
                require(r['decode_tokens']==n-1 and r['generate_reference_comparison']['equal'],'Reference output mismatch')
                require(token_ids_sha256(r['generated_token_ids'])==r['generated_token_ids_sha256'],'Output hash mismatch')
                if r['use_cache']:
                    shape=r['first_key_shape']
                    require(shape==r['first_value_shape'] and shape[2]==p+n-1==r['kv_cached_tokens'],'KV sequence shape mismatch')
                    require(r['kv_cache_layers']==28 and shape[:2]==[1,2] and shape[3]==128,'Qwen cache architecture mismatch')
                    require(r['actual_kv_tensor_bytes']==2*r['kv_cache_layers']*math.prod(shape)*2,'Actual KV bytes mismatch')
                else:
                    require(r['actual_kv_tensor_bytes']==r['kv_cached_tokens']==r['kv_cache_layers']==0,'OFF KV must be absent')
            for r in runs:
                require(r['warmup'] is False and r['included_in_statistics'] is True,'Wrong sample selection')
                require(all(math.isfinite(r[k]) and r[k]>0 for k in list(METRICS)[:-1]),'Invalid positive metric')
                require(math.isfinite(r['actual_kv_tensor_mib']) and r['actual_kv_tensor_mib']>=0,'Invalid KV MiB')
                require(math.isclose(r['prefill_latency_ms']+r['decode_latency_ms'],r['total_latency_ms'],rel_tol=1e-12),'Phase timing does not sum to total')
                require(math.isclose(r['decode_throughput_tokens_per_second']*r['decode_latency_ms']/1000,n-1,rel_tol=1e-12),'Decode throughput numerator mismatch')
                require(r['peak_allocated_mib']==r['peak_allocated_bytes']/2**20 and r['actual_kv_tensor_mib']==r['actual_kv_tensor_bytes']/2**20,'MiB conversion mismatch')
                require(0<r['baseline_allocated_bytes']<=r['peak_allocated_bytes']<=r['peak_reserved_bytes'],'Invalid peak allocated memory')
                require(r['baseline_allocated_bytes']<=r['baseline_reserved_bytes']<=r['peak_reserved_bytes'],'Invalid reserved memory')
            comparisons=[dict(execution_index=r['execution_index'],**compare_tokens(runs[0]['generated_token_ids'],r['generated_token_ids'])) for r in runs]
            equal=all(x['equal'] for x in comparisons)
            require(comparisons==c['output_comparisons'] and equal==c['all_generated_token_ids_equal'],'Equality metadata mismatch')
            stats={name:{metric:{'n':5,'mean':statistics.mean(r[metric] for r in runs if r['use_cache'] is mode),
                                'std':statistics.stdev(r[metric] for r in runs if r['use_cache'] is mode),'unit':unit}
                         for metric,unit in METRICS.items()} for name,mode in [('OFF',False),('ON',True)]}
            result[sweep]['conditions'].append({'prompt_tokens':p,'generation_tokens':n,'input_sha256':c['input_sha256'],
                'all_generated_token_ids_equal':equal,'all_match_generate_reference':True,'output_comparisons':comparisons,
                'statistics':stats,'total_latency_speedup':stats['OFF']['total_latency_ms']['mean']/stats['ON']['total_latency_ms']['mean']})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    output_dir=args.output_dir.resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    summary_path=output_dir/'detailed_sweeps_summary.json'
    raw=json.loads(RAW.read_text())
    settings_path=ROOT/'docs/detailed_sweeps_config.json'
    require(hashlib.sha256(settings_path.read_bytes()).hexdigest()==raw['settings_sha256'],'Settings hash mismatch')
    require(json.loads(settings_path.read_text())==raw['settings'],'Embedded settings mismatch')
    for name,digest in raw['source_sha256'].items():
        require(hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==digest,f'Measurement source changed: {name}')
    for path,digest in raw['settings']['input_file_sha256'].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,f'Measurement input changed: {path}')
    manifest=json.loads((ROOT/raw['settings']['prompt_input_manifest_path']).read_text())
    inputs={e['prompt_tokens']:e for e in manifest['inputs']}
    refs={name:json.loads((ROOT/path).read_text()) for name,path in raw['settings']['reference_results'].items()}
    for sweep in ['prompt','generation']:
        for c in raw['sweeps'][sweep]['conditions']:
            p,n=c['prompt_tokens'],c['generation_tokens']; entry=inputs[p]
            require(c['input_sha256']==entry['sha256']==token_ids_sha256(entry['token_ids']),'Input token hash mismatch')
            ref=next(x for x in refs[sweep]['conditions'] if x['prompt_tokens']==p and x.get('generation_tokens',128)==n)
            for r in c['warmup_runs']+c['repeat_runs']:
                tokens=next(x['generated_token_ids'] for x in ref['repeat_runs'] if x['use_cache'] is r['use_cache'])
                require(compare_tokens(tokens,r['generated_token_ids'])==r['generate_reference_comparison'],'Reference comparison metadata mismatch')
    report={'status':'passed','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'measurement_recorded_at_utc':raw['recorded_at_utc'],'base_config':raw['base_config'],'settings':raw['settings'],
            'engine':'manual greedy forward loop; new measurements, not reused generate timings',
            'sample_std_ddof':1,'outliers_removed':False,'sweeps':summarize(raw),
            'raw_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'csv_sha256':{}}
    report['all_outputs_equal']=all(c['all_generated_token_ids_equal'] for sw in report['sweeps'].values() for c in sw['conditions'])
    for sweep in ['prompt','generation']:
        path=output_dir/f'{sweep}_sweep_detailed_raw.csv'
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=RAW_FIELDS,lineterminator='\n');writer.writeheader()
            for c in raw['sweeps'][sweep]['conditions']:
                for r in c['repeat_runs']:
                    row={k:r[k] for k in RAW_FIELDS};row['generated_token_ids']=json.dumps(row['generated_token_ids']);writer.writerow(row)
        report['csv_sha256'][path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        path=output_dir/f'{sweep}_sweep_detailed_summary.csv'
        fields=['prompt_tokens','generation_tokens','use_cache','n']+[f'{k}_{stat}' for k in METRICS for stat in ['mean','std']]
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for c in report['sweeps'][sweep]['conditions']:
                for mode in ['OFF','ON']:
                    row={'prompt_tokens':c['prompt_tokens'],'generation_tokens':c['generation_tokens'],'use_cache':mode=='ON','n':5}
                    row.update({f'{k}_{stat}':c['statistics'][mode][k][stat] for k in METRICS for stat in ['mean','std']});writer.writerow(row)
        report['csv_sha256'][path.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        for c in report['sweeps'][sweep]['conditions']:
            print(f"{sweep} P={c['prompt_tokens']} G={c['generation_tokens']}: speedup={c['total_latency_speedup']:.3f}x; identical={c['all_generated_token_ids_equal']}")
    summary_path.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Validated 80 measurements and 32 warm-ups: {summary_path}')


if __name__=='__main__':
    main()
