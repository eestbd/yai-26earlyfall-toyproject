"""Plot the validated generation sweep with per-mode sample standard deviations."""
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
SUMMARY=ROOT/'results/generation_sweep_summary.json'
RAW=ROOT/'results/generation_sweep_raw.json'


def main():
    s=json.loads(SUMMARY.read_text())
    if s['status']!='passed' or hashlib.sha256(RAW.read_bytes()).hexdigest()!=s['raw_sha256']:
        raise ValueError('Invalid summary provenance')
    lengths=[c['generation_tokens'] for c in s['conditions']]
    panels=[('total_latency_seconds','Total generation latency','Seconds',1),
            ('throughput_tokens_per_second','Generated-token throughput','Tokens/s (includes prefill)',1),
            ('peak_allocated_bytes','Peak allocated memory','MiB (zoomed axis)',2**20),
            ('peak_allocated_above_baseline_bytes','Allocated peak above baseline','MiB (not KV-cache size)',2**20)]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    for ax,(metric,title,ylabel,divisor) in zip(axes.flat,panels):
        for name,color,fmt in [('ON','#168A7C','o-'),('OFF','#BA5D21','s--')]:
            means=[c['statistics'][name][metric]['mean']/divisor for c in s['conditions']]
            stds=[c['statistics'][name][metric]['std']/divisor for c in s['conditions']]
            ax.errorbar(lengths,means,yerr=stds,fmt=fmt,color=color,capsize=4,label='Cache '+name)
        ax.set(title=title,xlabel='Generation length (tokens)',ylabel=ylabel,xticks=lengths)
        if metric!='peak_allocated_bytes':
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.9),ncol=2,frameon=False)
    fig.suptitle('KV Cache ON/OFF: generation length sweep',fontsize=20,fontweight='bold',y=.985)
    fig.text(.5,.935,'Qwen2.5-1.5B | RTX 3090 | BF16 | eager | prompt 128 | batch 1',ha='center',fontsize=11)
    speeds='; '.join(f"{c['generation_tokens']}: {c['speedup']:.2f}x" for c in s['conditions'])
    fig.text(.5,.075,'Means ± sample SD (n=5, ddof=1). Warm-up excluded. The original 128 input token IDs are fixed.',ha='center',fontsize=9)
    fig.text(.5,.05,'Latency speedup (OFF / ON) by generation length: '+speeds,ha='center',fontsize=9)
    text='Outputs match across all ten measured runs at each length.' if s['all_conditions_outputs_equal'] else 'WARNING: Some ON/OFF outputs differ; see summary mismatch diagnostics.'
    fig.text(.5,.025,text+' Peak memory includes temporary tensors.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.82,bottom=.16,left=.09,right=.97,hspace=.45,wspace=.3)
    paths=[ROOT/'plots/generation_sweep.png',ROOT/'plots/generation_sweep.pdf']
    for p in paths: fig.savefig(p,dpi=180,facecolor='white')
    plt.close(fig)
    meta={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'matplotlib_version':matplotlib.__version__,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'summary_sha256':hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
          'outputs':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT/'results/generation_sweep_plot_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Saved generation sweep PNG/PDF.')


if __name__=='__main__':
    main()
