"""Python quantitative figures: all observations, all final lost/rescue cases, no sample picking."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import attribution_core as c

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.spines.right':False,'axes.spines.top':False,
    'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False,'axes.linewidth':.7})
COLORS=['#67788a','#c29b70','#867ba8','#507a8b','#809f8b']

def save(fig,root):
    for ext in ('svg','pdf','png'):fig.savefig(str(root)+'.'+ext,dpi=300,bbox_inches='tight')

def main():
    out=c.ROOT/'figures';out.mkdir(exist_ok=False)
    cases=c.ROOT/'cases';cases.mkdir(exist_ok=False)
    z=c.load_npz(c.ROOT/'derived_ports.npz');ports=z['paths'];cov=z['good'].any(2);native=z['good'][0]
    m=native.sum(1);st=np.where(native,np.where(m[:,None]==1,0,1),np.where(m[:,None]>0,2,3))
    dm=z['initial_error']-z['final_error'];dj=z['J'][0]-z['J'][3]
    # Quantitative grid: stage hero + Native-frozen candidate strata, full population.
    fig=plt.figure(figsize=(183/25.4,140/25.4),layout='constrained')
    gs=fig.add_gridspec(2,2);ax=fig.add_subplot(gs[0,:])
    for j,key in enumerate(c.PORTS):
        ax.bar(j,cov[j].sum(),color=COLORS[j]);ax.text(j,cov[j].sum()+1,str(cov[j].sum()),ha='center')
    ax.set(xticks=range(5),xticklabels=['Native','Continuous\n(diagnostic)','Quantized','Guarded','Oracle\n(GT)'],ylabel='Covered windows / 128',ylim=(0,128))
    ax.set_title('a  Mean candidate improvement does not guarantee Native support retention',loc='left',fontweight='bold')
    for col,(values,title) in enumerate([(dm,'Parameter MSE reduction'),(dj,'Native minus guarded J')]):
        ax=fig.add_subplot(gs[1,col]);means=[values[st==j].mean() if (st==j).any() else np.nan for j in range(4)]
        ax.bar(range(4),means,color=COLORS[:4]);ax.axhline(0,color='#444444',lw=.7)
        ax.set(xticks=range(4),xticklabels=['Unique\nsupport','Multiple\nsupport','Bad slot\nin good set','Bad\nwindow'],ylabel=title)
        ax.set_title(chr(98+col)+'  Native-frozen strata',loc='left',fontweight='bold')
        for j,mean in enumerate(means):ax.text(j,mean,f'n={(st==j).sum()}',ha='center',va='bottom' if mean>=0 else 'top',fontsize=6)
    save(fig,out/'stage_and_gain');plt.close(fig)
    u=c.load_npz(c.ROOT/'neighbor_utility_metrics.npz')
    fig,axes=plt.subplots(1,3,figsize=(183/25.4,62/25.4),layout='constrained')
    metrics=[('distance_squared','target_distance','H squared input distance','Teacher parameter distance'),
             ('target_distance','executed_gap','Teacher parameter distance','Cross-action executed J-gap'),
             ('distance_squared','executed_gap','H squared input distance','Cross-action executed J-gap')]
    for j,(xn,yn,xlabel,ylabel) in enumerate(metrics):
        xx=u['H_main_'+xn].ravel();yy=u['H_main_'+yn].ravel();ok=np.isfinite(xx)&np.isfinite(yy)
        axes[j].hexbin(xx[ok],yy[ok],gridsize=28,mincnt=1,cmap='Blues',rasterized=True)
        axes[j].set(xlabel=xlabel,ylabel=ylabel);axes[j].set_title(f'{chr(97+j)}  All {ok.sum()} neighbor pairs',loc='left',fontsize=7)
    save(fig,out/'neighbor_functional_compatibility');plt.close(fig)
    indices=np.flatnonzero(cov[0]!=cov[3]);assert len(indices)==15
    index=['# All final lost and rescue case pages','','All 128 windows were analyzed. These are every final event, not selected examples. C is unprojected diagnostic; O uses GT. All20 remain in saved arrays.','']
    with PdfPages(cases/'ALL_15_CASES.pdf') as pdf:
        for i in indices:
            sid=int(z['sample_id'][i]);ev='LOST' if cov[0,i] else 'RESCUE'
            fig=plt.figure(figsize=(183/25.4,230/25.4),layout='constrained');gs=fig.add_gridspec(4,3,height_ratios=[1,1,.55,1.15])
            for j,key in enumerate(c.PORTS):
                ax=fig.add_subplot(gs[j//3,j%3]);p=ports[j,i]
                for k in range(20):
                    support=bool(native[i,k]);ax.plot(p[k,:,0],p[k,:,1],color=COLORS[j],alpha=.85 if support else .16,lw=1.1 if support else .5)
                    if support:ax.text(p[k,-1,0],p[k,-1,1],str(k),fontsize=5)
                ax.plot(z['past'][i,:,0],z['past'][i,:,1],'k.--',ms=2,lw=.7,label='past')
                ax.plot(z['gt'][i,:,0],z['gt'][i,:,1],color='#b86c61',lw=1.3,marker='.',ms=2,label='GT')
                ax.set_title(f'{key}: good={int(z["good"][j,i].sum())}, minJ={z["J"][j,i].min():.3f}',fontsize=7)
                ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='local x (m)',ylabel='local y (m)')
            ax=fig.add_subplot(gs[1,2]);ax.axis('off')
            ax.text(0,1,f'{ev}: ID {sid}\nTrack {z["track_key"][i]}\nNative support m={m[i]}\nStage N/C/Q/S:\n'+
                ' -> '.join(str(int(cov[j,i])) for j in range(4))+'\n\nFaint: all other slots\nBold / endpoint labels:\nall Native-good slots\nDashed black: past\nMuted red: GT\n\nC: no radial clip\nO: GT oracle',va='top',fontsize=7)
            ax=fig.add_subplot(gs[2:,:]);ax.axis('off')
            header='slot  teacher  J(N)  J(C)  J(Q)  J(S)  J(O)   good N/C/Q/S/O   failed C / Q / S'
            lines=[header]
            for k in range(20):
                marker='*' if native[i,k] else ' '
                js=' '.join(f'{z["J"][j,i,k]:5.2f}' for j in range(5))
                bits=' / '.join(''.join(str(int(a)) for a in z['failed'][j,i,k]) for j in (1,2,3))
                gg=''.join(str(int(z['good'][j,i,k])) for j in range(5))
                lines.append(f'{k:2d}{marker}   {int(z["teacher_action"][i,k]):2d}    {js}       {gg}           {bits}')
            ax.text(0,1,'\n'.join(lines),va='top',fontfamily='DejaVu Sans Mono',fontsize=5.1,linespacing=1.65)
            ax.text(0,.04,'* Native-good support. Failure bits: speed p95, acceleration mean/p95, jerk mean/p95, junction, nonfinite.\nPhysical slot is not semantic mode identity. Training subset, one tiny H42; no generalization/causality claim.',fontsize=6)
            fig.suptitle(f'Complete event audit: {ev}, sample {sid}',fontsize=10,fontweight='bold')
            save(fig,cases/f'{ev}_{sid}');pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
            index.append(f'- {ev} sample {sid}: [{ev}_{sid}.pdf](cases/{ev}_{sid}.pdf)')
    (c.ROOT/'CASE_INDEX.md').open('x',encoding='utf-8').write('\n'.join(index)+'\n')
    c.write_json(c.ROOT/'FIGURE_QA.json',dict(status='RENDERED_VISUAL_REVIEW_REQUIRED',
        backend='Python/matplotlib',case_pages=15,lost=7,rescue=8,source='derived_ports.npz + neighbor_utility_metrics.npz',
        all_observations_used=True,finite_pair_filter='Only missing-neighbor NaN has no plotted pair; missing counts in neighbor_summary.csv',
        no_journal_submission_claim=True,no_AI_generated_data=True,editable_svg_text=True))
    print('Rendered all15 complete event pages and two summary grids',flush=True)

if __name__=='__main__':main()
