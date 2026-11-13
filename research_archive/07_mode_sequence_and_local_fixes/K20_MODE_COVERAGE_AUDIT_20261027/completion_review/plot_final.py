from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from morphology import COARSE
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
COLORS=['#808080','#9E8ABD','#4C78A8','#4E9B81','#D89A58']
def export(fig,folder,name):
    folder.mkdir(exist_ok=True,parents=True)
    # Explicit exports and visible bin-count scales; old versions remain untouched.
    for ax in list(fig.axes):
        for coll in ax.collections:
            if coll.__class__.__name__=='PolyCollection' and coll.get_array() is not None:
                cb=fig.colorbar(coll,ax=ax,pad=.02,fraction=.045);cb.set_label('Bin count',fontsize=5);cb.ax.tick_params(labelsize=5)
    if name=='E1_distribution':
        h,l=fig.axes[2].get_legend_handles_labels();fig.axes[2].get_legend().remove();fig.legend(h,l,loc='outside lower center',ncols=5,fontsize=6)
    if name=='E2_coverage_oracle':
        h,l=fig.axes[0].get_legend_handles_labels();fig.axes[0].get_legend().remove();fig.legend(h,l,loc='outside lower center',ncols=4,fontsize=6)
    if name.endswith('cases'):
        handles=[plt.Line2D([0],[0],color=c,lw=1,label=n) for c,n in zip(COLORS,COARSE)];handles.append(plt.Line2D([0],[0],color='black',label='observed history'))
        if name.startswith('E2'):
            handles.extend([plt.Line2D([0],[0],color='black',ls='--',label='GT (posthoc)'),plt.Line2D([0],[0],color='black',marker='o',mfc='none',ls='none',label='quality winner endpoint')])
            for ax in fig.axes:
                if not ax.axison:ax.set_title('absent_shape_failure: no qualifying coarse20 hold case',fontsize=6)
        fig.legend(handles=handles,loc='outside lower center',ncols=4,fontsize=6)
    fig.savefig(folder/(name+'.svg'),bbox_inches='tight',facecolor='white')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(folder/(name+'.png'),bbox_inches='tight',dpi=600,facecolor='white')
    plt.close(fig)
def label(axes):
    for i,ax in enumerate(np.ravel(axes)):ax.text(-.12,1.04,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=9)
def hashed(indices,ids,tag,k):
    return sorted(indices,key=lambda i:hashlib.sha256((tag+'|'+str(int(ids[i]))).encode()).hexdigest())[:k]
def draw_case(ax,x,m,i,gt=None,important=None):
    for k in range(20):
        p=np.vstack((x['past'][i,-1],x['native'][i,k]));ax.plot(p[:,0],p[:,1],lw=.65,color=COLORS[int(m['mode'][i,k])],alpha=.8)
        if important is not None and important[k]:ax.plot(p[-1,0],p[-1,1],'o',ms=4,mfc='none',mec='black',mew=.6)
    hist=x['past'][i];ax.plot(hist[:,0],hist[:,1],color='black',lw=1,alpha=.6);ax.scatter([0],[0],c='black',s=5)
    if gt is not None:ax.plot(np.r_[0,gt[:,0]],np.r_[0,gt[:,1]],color='black',ls='--',lw=1.1)
    ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('local x (m)');ax.set_ylabel('local y (m)')
def main():
    g=guard(True);x=load(HERE/'inputs/X_hold.npz');m=load(HERE/'E1_mode_distribution/hold/coarse20/modes.npz');geom=load(HERE/'inputs/GEOM_hold.npz');cfg=read(HERE/'config/analysis_config.json');N=len(x['sample_id']);folder=HERE/'figures_final/E1'
    fig,axes=plt.subplots(2,3,figsize=(7.2047244,4.4094488),layout='constrained');ax=axes.ravel()
    for gi,group in enumerate(GROUPS[1:]):
        mask=x['group']==gi
        ax[0].hist(m['Pmax'][mask],bins=np.linspace(0,1,21),histtype='step',density=True,label=group,lw=1)
        ax[1].hist(m['Meff'][mask],bins=np.linspace(1,5,21),histtype='step',density=True,label=group,lw=1)
    ax[0].set(xlabel='Largest coarse-class fraction',ylabel='Density');ax[0].legend(fontsize=6)
    ax[1].set(xlabel='Effective coarse classes',ylabel='Density')
    means=[]
    for gi in range(3):means.append([(m['mode'][x['group']==gi]==k).mean() for k in range(5)])
    means=np.array(means);bottom=np.zeros(3)
    for k,n in enumerate(COARSE):ax[2].bar(range(3),means[:,k],bottom=bottom,color=COLORS[k],label=n);bottom+=means[:,k]
    ax[2].set(xticks=range(3),xticklabels=GROUPS[1:],ylabel='Mean candidate fraction');ax[2].tick_params(axis='x',labelsize=6);ax[2].legend(fontsize=5,loc='upper right')
    valid=np.isfinite(m['turn'])&~m['low'];ax[3].hexbin(np.rad2deg(m['turn'][valid]),m['delta_speed'][valid],gridsize=35,cmap='cividis',mincnt=1);ax[3].set(xlabel='Future turning angle (degrees)',ylabel='Late - early speed (m/s)')
    ax[3].text(.03,.97,f'All defined: {int(valid.sum())}\nlow/undefined: {int((~valid).sum())}',transform=ax[3].transAxes,va='top',fontsize=5)
    ax[4].hexbin(geom['distinct'],m['Meff'],gridsize=25,cmap='cividis',mincnt=1);ax[4].set(xlabel='Geometric distinct (original)',ylabel='Effective coarse classes')
    ax[5].hexbin(m['same_pair'],m['near_phi_pair'],gridsize=25,cmap='cividis',mincnt=1);ax[5].set(xlabel='Same coarse-class pair fraction',ylabel='Morphology-near pair fraction')
    label(axes);fig.suptitle(f'Native K20 morphology | hold n={N}, tracks={len(np.unique(x["track_key"]))}',fontsize=8);export(fig,folder,'E1_distribution')
    selected=[];fig,axes=plt.subplots(3,3,figsize=(7.2047244,7.2834646),layout='constrained')
    for gi,group in enumerate(GROUPS[1:]):
        ids=hashed(np.flatnonzero(x['group']==gi),x['sample_id'],'E1-example-v1',3)
        for col,i in enumerate(ids):draw_case(axes[gi,col],x,m,i);axes[gi,col].set_title(f'{group} | ID {x["sample_id"][i]}\nMeff={m["Meff"][i]:.2f}, Pmax={m["Pmax"][i]:.2f}',fontsize=6);selected.append(dict(stage='E1',kind=group,sample_id=int(x['sample_id'][i]),GT_selected=False,rule='hash smallest3 per registered group'))
    export(fig,folder,'E1_GT_free_cases');g.check()
    # E2 plots use GT only after the completed E1 lock.
    gt=load(HERE/'inputs/GT_hold.npz')['gt'];r=load(HERE/'E2_oracle_coverage/hold/coarse20/slot_removal_oracle.npz');q=load(HERE/'E2_oracle_coverage/hold/candidate_gt_metrics.npz');rows=read(HERE/'statistics/GROUP_STATISTICS.json');out=HERE/'figures_final/E2'
    fig,axes=plt.subplots(2,3,figsize=(7.2047244,4.5669291),layout='constrained');ax=axes.ravel()
    labels=['GT shape absent','present, missed','undefined','near boundary'];colors=['#D89A58','#4C78A8','#9E8ABD','#808080'];bottom=np.zeros(3)
    for k,n in enumerate(labels):
        counts=np.array([np.sum((x['group']==gi)&~r['native_good']&(r['failure_reason']==k)) for gi in range(3)]);den=np.array([np.sum((x['group']==gi)&~r['native_good']) for gi in range(3)]);v=counts/den;ax[0].bar(range(3),v,bottom=bottom,color=colors[k],label=n);bottom+=v
    ax[0].set(xticks=range(3),xticklabels=GROUPS[1:],ylabel='Fraction of native-failure windows');ax[0].legend(fontsize=5)
    fractions=np.array([(m['mode']==k).mean() for k in range(5)]);hits=np.array([((m['mode']==k)&q['good']).any(1).mean() for k in range(5)]);ax[1].bar(np.arange(5)-.17,fractions,.34,label='candidate share',color='#9DB5CE');ax[1].bar(np.arange(5)+.17,hits,.34,label='hit/all windows',color='#4C78A8');ax[1].set(xticks=range(5),xticklabels=COARSE,ylabel='Fraction (different denominators)');ax[1].tick_params(axis='x',labelrotation=45,labelsize=5);ax[1].legend(fontsize=5)
    for k,n in enumerate(labels):
        vals=q['J'].min(1)[~r['native_good']&(r['failure_reason']==k)];vals=np.sort(vals)
        if len(vals):ax[2].plot(vals,np.arange(1,len(vals)+1)/len(vals),color=colors[k],label=f'{n} n={len(vals)}')
    ax[2].set(xlabel='Native-failure J (all finite)',ylabel='Empirical CDF');ax[2].set_xscale('log');ax[2].legend(fontsize=5)
    for name,color in [('coarse20','#4C78A8'),('joint20','#D89A58'),('cluster_main','#4E9B81')]:
        rr=load(HERE/f'E2_oracle_coverage/hold/{name}/slot_removal_oracle.npz');ax[3].hist(rr['R'].sum(1),bins=np.arange(-.5,21.5),histtype='step',density=True,label=name,color=color)
    ax[3].set(xlabel='Single-GT single-delete\noracle-removable',ylabel='Fraction');ax[3].legend(fontsize=5)
    ax[4].hexbin(m['Meff'],r['R'].sum(1),gridsize=25,cmap='cividis',mincnt=1);ax[4].set(xlabel='Effective coarse classes',ylabel='Oracle-removable count')
    for variant in ('coarse20','joint20','coarse15','coarse30','floor_half','floor_double'):
        row=next(v for v in rows if v['population']=='hold' and v['variant']==variant and v['group']=='change');ax[5].errorbar(row['oracle_removable'],variant,xerr=[[row['oracle_removable']-row['oracle_removable_CI95_low']],[row['oracle_removable_CI95_high']-row['oracle_removable']]],fmt='o',color='#4C78A8',ms=3)
    ax[5].set(xlabel='Mean removable:\nwhole-track 95% CI');ax[5].tick_params(axis='y',labelsize=5)
    label(axes);fig.suptitle('Observed-GT contributions | descriptive, not a deletion policy',fontsize=8);export(fig,out,'E2_coverage_oracle')
    kinds=[('absent_shape_failure',~r['native_good']&(r['failure_reason']==0)),('present_shape_failure',~r['native_good']&(r['failure_reason']==1)),('same_shape_key_success',r['native_good']&(q['good'].sum(1)==1)&(m['mode_count'][np.arange(N),q['J'].argmin(1)]>=2)),('diverse_but_failure',~r['native_good']&(m['Meff']>=3))]
    fig,axes=plt.subplots(4,2,figsize=(7.2047244,8.6614173),layout='constrained')
    for row,(kind,mask) in enumerate(kinds):
        selected_ids=hashed(np.flatnonzero(mask),x['sample_id'],'E2-case-v1|'+kind,2)
        for col in range(2):
            a=axes[row,col]
            if col>=len(selected_ids):a.text(.5,.5,'No registered example',ha='center');a.set_axis_off();continue
            i=selected_ids[col];draw_case(a,x,m,i,gt[i],q['winJ'][i]|q['winA'][i]|q['winF'][i]);a.set_title(f'{kind} | ID {x["sample_id"][i]}',fontsize=6);selected.append(dict(stage='E2',kind=kind,sample_id=int(x['sample_id'][i]),GT_selected=True,rule='predefined diagnostic condition then hash-smallest2'))
    export(fig,out,'E2_GT_posthoc_cases');table(HERE/'figures_final/SELECTED_CASES.csv',selected)
    write(HERE/'figures_final/FIGURE_QA.json',dict(status='RENDERED_REQUIRES_VISUAL_INSPECTION',backend='python_saved_preference',archetype='quantitative grid + image plate',no_simulated_data=True,all_hold_points_hexbin_no_subsampling=True,undefined_counts_explicit=True,PNG_dpi=600,SVG_PDF_primary_editable_text=True,window_and_track_N_in_panels=True,source_data='E1_mode_distribution and E2_oracle_coverage NPZ/CSV; statistics GROUP_STATISTICS',descriptive_CI='20000 whole sequence|target track; no independent seed replication/no significance claims',not_final_journal_submission=True,case_selection=selected));resource(HERE/'figures_final/RESOURCE.json',g);print('PASS RENDER',flush=True)
if __name__=='__main__':main()


