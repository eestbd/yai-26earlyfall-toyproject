"""Create two six-panel scientific figures from validated split-phase measurements."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/kv-cache-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SUMMARY=ROOT/'results/detailed_sweeps_summary.json'
PANELS=[('prefill_latency_ms','Prefill latency','Time (ms)'),
        ('decode_latency_ms','Decode latency','Time (ms)'),
        ('total_latency_ms','Total latency','Time (ms)'),
        ('decode_throughput_tokens_per_second','Decode throughput','Tokens/s'),
        ('peak_allocated_mib','Peak allocated GPU memory','MiB (zoomed axis)'),
        ('actual_kv_tensor_mib','Actual KV tensor size','MiB')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary',type=Path,default=SUMMARY)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'plots')
    args=parser.parse_args()
    summary_path=args.summary.resolve()
    output_dir=args.output_dir.resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    summary=json.loads(summary_path.read_text())
    if summary['status']!='passed' or hashlib.sha256((ROOT/'results/detailed_sweeps_raw.json').read_bytes()).hexdigest()!=summary['raw_sha256']:
        raise ValueError('Invalid summary provenance')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    paths=[]
    for sweep in ['prompt','generation']:
        conditions=summary['sweeps'][sweep]['conditions']
        xkey='prompt_tokens' if sweep=='prompt' else 'generation_tokens'
        xlabel='Prompt length (tokens)' if sweep=='prompt' else 'Generation length (tokens)'
        xs=[c[xkey] for c in conditions]
        fig,axes=plt.subplots(2,3,figsize=(16,9))
        for ax,(metric,title,ylabel) in zip(axes.flat,PANELS):
            for mode,color,fmt in [('ON','#167BA8','o-'),('OFF','#D36716','s--')]:
                means=[c['statistics'][mode][metric]['mean'] for c in conditions]
                stds=[c['statistics'][mode][metric]['std'] for c in conditions]
                ax.errorbar(xs,means,yerr=stds,fmt=fmt,color=color,capsize=4,linewidth=1.8,markersize=5,label='KV Cache '+mode)
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xticks=xs)
            if metric!='peak_allocated_mib': ax.set_ylim(bottom=0)
            ax.grid(alpha=.22)
        fixed='generation length = 128 tokens' if sweep=='prompt' else 'prompt length = 128 tokens'
        fig.suptitle(f'KV Cache ON/OFF: {sweep} length sweep ({fixed})',fontsize=18,fontweight='bold',y=.98)
        fig.text(.5,.93,'Qwen2.5-1.5B | RTX 3090 | BF16 | eager | batch 1 | manual greedy loop',ha='center',fontsize=11)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.905),ncol=2,frameon=False)
        fig.text(.5,.075,'Points = means; error bars = sample SD; 5 measured runs per condition and mode. Warm-up excluded.',ha='center',fontsize=10)
        fig.text(.5,.051,'Prefill includes first-token selection. Decode covers the remaining N - 1 tokens; decode throughput = (N - 1) / decode time.',ha='center',fontsize=10)
        fig.text(.5,.027,'1 MiB = 2^20 bytes. Peak includes model and temporary tensors (zoomed axis); actual KV sums final K/V tensors for P + N - 1 positions.',ha='center',fontsize=9)
        if not all(c['all_generated_token_ids_equal'] for c in conditions):
            fig.text(.5,.007,'WARNING: ON/OFF generated token IDs differ; see summary diagnostics.',ha='center',color='red',fontsize=9)
        fig.subplots_adjust(top=.81,bottom=.17,left=.06,right=.975,hspace=.5,wspace=.32)
        for suffix in ['png','pdf']:
            path=output_dir/f'{sweep}_sweep_detailed.{suffix}';fig.savefig(path,dpi=180,facecolor='white');paths.append(path)
        plt.close(fig)
    meta={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'matplotlib_version':matplotlib.__version__,
          'summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'outputs':{p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (summary_path.parent/'detailed_sweeps_plot_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Saved two six-panel PNG figures and matching PDFs.')


if __name__=='__main__':
    main()
