"""Python figure contract: does more training improve CALIB under independent schedules?
Quantitative grid; all 2 seeds and 4 budgets, all 64 per-epoch observations retained.
Internal research figure, 183 x 120 mm, editable SVG/PDF and PNG, source CSV; no p-values.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *


def main():
    out=HERE/'E1/figures';out.mkdir(exist_ok=False)
    rows=[]
    for seed in (42,43):
        for epoch in range(1,65):
            base=HERE/'runs'/f'E1B_E64_s{seed}'
            x=read_json(base/f'calib_epoch{epoch:02}.json');t=read_json(base/f'epoch{epoch:02}.json')
            rows.append(dict(seed=seed,epoch=epoch,CALIB_minADE=x['overall']['minADE'],CALIB_change_coverage_pp=100*x['change']['coverage'],FIT85_loss=t['mean_train_loss']))
    write_csv(out/'learning_curve_source.csv',rows)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig,axs=plt.subplots(2,2,figsize=(7.204724,4.724409),layout='constrained')
    for seed,color,ls in [(42,'#3B6C8E','-'),(43,'#A46E42','--')]:
        rs=[r for r in rows if r['seed']==seed];xx=[r['epoch'] for r in rs]
        for ax,key in zip(axs.flat,['CALIB_minADE','CALIB_change_coverage_pp','FIT85_loss']):
            ax.plot(xx,[r[key] for r in rs],color=color,ls=ls,lw=1,label=f'Seed {seed}')
        budgets=[8,16,32,64];vals=[]
        for E in budgets:
            z=read_json(HERE/'runs'/f'E1B_E{E}_s{seed}'/'calib/summary_v2.json')
            vals.append(z['overall']['minADE'])
        axs[1,1].plot(budgets,vals,marker='o',ms=3,color=color,ls=ls,lw=1,label=f'Seed {seed}')
    labels=['CALIB minADE (m)','CALIB change coverage (%)','FIT85 mean training loss','CALIB final minADE (m)']
    for letter,ax,label in zip('abcd',axs.flat,labels):
        ax.set_xlabel('Epoch' if letter!='d' else 'Independent training budget (epochs)');ax.set_ylabel(label)
        ax.text(-.16,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=8);ax.legend(fontsize=6)
        ax.tick_params(direction='out',width=.6,length=2)
    axs[1,1].set_xscale('log',base=2);axs[1,1].set_xticks([8,16,32,64],[8,16,32,64])
    fig.savefig(out/'calibration_learning_curves.svg')
    fig.savefig(out/'calibration_learning_curves.pdf')
    fig.savefig(out/'calibration_learning_curves.png',dpi=600)
    plt.close(fig)
    write_json(out/'QA.json',dict(backend='Python matplotlib',archetype='quantitative grid',seeds=[42,43],
        curve_rows=len(rows),excluded_rows=0,all64_epochs=True,independent_budget_runs=8,
        uncertainty='Two seed trajectories shown individually, no inferential interval',
        split='Whole-track FIT85/CALIB within historical FIT; dev never used for E* selection',
        visual_review='PENDING_HUMAN_OR_MODEL_RENDER_INSPECTION',source_sha=sha(out/'learning_curve_source.csv')))


if __name__=='__main__':main()
