from pathlib import Path
import sys,csv,textwrap
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from fine_features import ENTRY,ONSET,REASONS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
COLORS=['#4C78A8','#4E9B81','#D89A58','#9E8ABD','#808080','#B3B3B3']
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
MAIN='a20_history3_s2'
def export(fig,name):
    fig.savefig(OUT/(name+'.svg'))
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.png'),dpi=600)
    plt.close(fig)
def lookup(rows,pop,variant,group,metric):
    return next(r for r in rows if r['population']==pop and r['variant']==variant and r['group']==group and r['metric']==metric)
def main():
    g=guard(True);data=read(HERE/'statistics/bootstrap_ci.json');x=xload('hold');p=x['native'];gt=yload('hold')['gt'];mask=x['group']==2
    f=load(HERE/f'E3_gt_free/hold/{MAIN}.npz');gf=load(HERE/f'E3_gt_audit/hold/{MAIN}/gt_features.npz');r=load(HERE/f'E3_gt_audit/hold/{MAIN}/presence.npz');q=load(HERE/'E3_gt_audit/hold/native_quality.npz');err=load(HERE/'completion_review/error_reviewed/hold.npz')
    fail=mask&~q['native_good'];common=fail&r['common_valid']
    # Hero crosswalk: actual candidates with all categories and denominators.
    fig,axs=plt.subplots(1,3,figsize=(183/25.4,74/25.4),layout='constrained',gridspec_kw={'width_ratios':[1,1.4,1]})
    for ax,labels,names in [(axs[0],f['entry'],ENTRY),(axs[1],f['onset'],ONSET)]:
        counts=np.array([[( (f['L0']==i)&(labels==j)&mask[:,None]).sum() for j in range(len(names))] for i in range(5)])
        den=counts.sum(1);rate=np.divide(counts,den[:,None],out=np.zeros(counts.shape,float),where=den[:,None]>0)
        ax.imshow(rate,cmap='Blues',vmin=0,vmax=1,aspect='auto')
        ax.set_xticks(range(len(names)),names,rotation=65,ha='right',fontsize=5);ax.set_yticks(range(5),['low','undefined','straight','left','right'],fontsize=6)
        for i in range(5):
            for j in range(len(names)):ax.text(j,i,f'{counts[i,j]}\n{100*rate[i,j]:.0f}%',ha='center',va='center',fontsize=5,color='white' if rate[i,j]>.55 else 'black')
        ax.set_title('Old L0 → '+('entry' if len(names)==4 else 'turn onset'),fontsize=7)
    vals=[];lo=[];hi=[]
    for group in ('stable','transition','change'):
        z=lookup(data,'hold',MAIN,group,'candidate_pooled_turn_given_old_straight');vals.append(100*z['estimate']);lo.append(100*(z['estimate']-z['CI95_low']));hi.append(100*(z['CI95_high']-z['estimate']))
    axs[2].bar(range(3),vals,color=COLORS[:3]);axs[2].errorbar(range(3),vals,yerr=[lo,hi],fmt='none',color='black',capsize=2)
    axs[2].set_xticks(range(3),['stable','transition','change'],rotation=60,ha='right',fontsize=6);axs[2].set_ylabel('Entry-turn / old-straight candidates (%)');axs[2].set_title('Whole-track 95% CI',fontsize=7)
    fig.suptitle('GT-free crosswalk | hold change: 1023 windows, 74 tracks; 20 candidates/window',fontsize=8)
    export(fig,'fine_mode_crosswalk')
    # Funnel uses same common-valid failures and separately shows all683.
    fig,axs=plt.subplots(1,2,figsize=(183/25.4,78/25.4),layout='constrained')
    labels=['old L0','entry','entry + onset','entry + onset + speed'];values=[int((common&load(OLD/'E2_oracle_coverage/hold/coarse20/slot_removal_oracle.npz')['GT_mode_present']).sum())]+[int((common&r[k].any(1)).sum()) for k in ('entry_match','entry_onset_match','full_match')]
    axs[0].barh(range(4),values,color=COLORS[0]);axs[0].set_yticks(range(4),labels);axs[0].invert_yaxis();axs[0].set_xlabel(f'Present windows / same {common.sum()} comparable failures')
    for i,v in enumerate(values):axs[0].text(v,i,f' {v}',va='center',fontsize=6)
    cnt=np.bincount(r['reason'][fail],minlength=6);short=['undefined / boundary','entry absent','onset missing','speed missing','refined present, inaccurate','partial valid']
    axs[1].barh(range(6),cnt,color=COLORS);axs[1].set_yticks(range(6),short,fontsize=6);axs[1].invert_yaxis();axs[1].set_xlabel('Mutually exclusive count / all683 failures')
    for i,v in enumerate(cnt):axs[1].text(v,i,f' {v}',va='center',fontsize=6)
    fig.suptitle('GT-associated diagnosis | no threshold tuning; decision-chain order is not causation',fontsize=8);export(fig,'absence_funnel')
    # No cherry-picked scales: different metrics occupy separate axes and units.
    names=['FDE','ADE_first','ADE_middle','ADE_last','entry_heading_error','onset_time_diff','shape_distance','phase_progress_diff']
    fig,axs=plt.subplots(2,4,figsize=(183/25.4,105/25.4),layout='constrained')
    for ax,name in zip(axs.ravel(),names):
        sets=[];lab=[]
        for title,base,k in [('success kJ',mask&q['native_good'],err['k_J']),('failure kJ',fail,err['k_J']),('fine-match failure',fail&(r['reason']==4),err['k_mode'])]:
            i=np.flatnonzero(base&(k>=0));v=err[name][i,k[i]];v=v[np.isfinite(v)]
            sets.append(v);lab.append(title+'\nn='+str(len(v)))
        nonempty=[i for i,v in enumerate(sets) if len(v)]
        if nonempty:ax.boxplot([sets[i] for i in nonempty],positions=nonempty,showfliers=False,widths=.6,medianprops={'color':'#4C78A8'})
        ax.set_xticks(range(3),lab,rotation=65,ha='right',fontsize=5);ax.set_title(name.replace('_',' '),fontsize=6);ax.tick_params(labelsize=6)
        ax.set_ylabel('degrees' if 'heading' in name else ('seconds' if name=='onset_time_diff' else ('unitless' if name=='phase_progress_diff' else 'm')),fontsize=6)
    fig.suptitle('hold change | all data quantified; box=IQR, median; outlier marks omitted, not observations',fontsize=8);export(fig,'error_profiles')
    fig,axs=plt.subplots(1,2,figsize=(183/25.4,80/25.4),layout='constrained')
    ports=['Native','END_b02','END_b04','END_unbounded','TIME_grid'];v=[];lower=[];upper=[]
    for port in ports:
        z=lookup(data,'hold','oracle','change',port+'_coverage');v.append(100*z['estimate']);lower.append(100*(z['estimate']-z['CI95_low']));upper.append(100*(z['CI95_high']-z['estimate']))
    axs[0].bar(range(5),v,color=[COLORS[4],COLORS[0],COLORS[0],COLORS[2],COLORS[1]]);axs[0].errorbar(range(5),v,yerr=[lower,upper],fmt='none',capsize=2,color='black')
    axs[0].set_xticks(range(5),['Native','END ≤0.2m','END ≤0.4m','END unbounded','TIME finite grid'],rotation=55,ha='right',fontsize=6);axs[0].set_ylabel('Same-candidate coverage (%)')
    with (HERE/'E4_oracle/oracle_overlap.csv').open(encoding='utf-8') as fh:ov=list(csv.DictReader(fh))
    rr=next(z for z in ov if z['population']=='hold' and z['group']=='change' and z['endpoint_port']=='END_b04')
    counts=[int(rr[k]) for k in ('only_END','only_TIME','both','neither')]
    axs[1].bar(range(4),counts,color=COLORS[:4]);axs[1].set_xticks(range(4),['END0.4 only','TIME only','both','neither'],rotation=55,ha='right',fontsize=6);axs[1].set_ylabel('Original failure windows /683')
    for i,vv in enumerate(counts):axs[1].text(i,vv,str(vv),ha='center',va='bottom',fontsize=6)
    fig.suptitle('GT ORACLES, not model gains | END and TIME must not be added; motion reported separately',fontsize=8);export(fig,'oracle_gain_overlap')
    # Fixed-condition then hash examples, paired counterexamples; no favorable substitution.
    idx=np.arange(len(p));kj=err['k_J'];kt=idx,kj
    conditions=[('old-straight / GT-entry-turn',mask&(gf['L0'][:,0]==2)&np.isin(gf['entry'][:,0],[1,2]),mask&(gf['L0'][:,0]==2)&(gf['entry'][:,0]==0)),('refined-mode missing',fail&np.isin(r['reason'],[1,2,3]),fail&(r['reason']==4)),('same-mode timing mismatch',fail&(r['reason']==4)&(err['onset_time_diff'][kt]>=.8),fail&(r['reason']==4)&(err['onset_time_diff'][kt]==0)),('near-path / endpoint error',fail&(err['shape_distance'][kt]<=.2)&(err['FDE'][kt]>.4),fail&(err['shape_distance'][kt]>.2)&(err['FDE'][kt]>.4))]
    selected=[]
    fig,axs=plt.subplots(4,4,figsize=(183/25.4,205/25.4),layout='constrained')
    for col,(label,positive,negative) in enumerate(conditions):
        for row,base in [(0,positive),(2,negative)]:
            ids=np.flatnonzero(base)
            if len(ids):
                i=min(ids,key=lambda i:hashlib.sha256(f'fine-case-v1|{label}|{row}|{int(x["sample_id"][i])}'.encode()).hexdigest());selected.append(dict(condition=label,counterexample=row==2,sample_id=int(x['sample_id'][i]),selection='predefined condition then hash smallest'))
                ax=axs[row,col];sp=axs[row+1,col]
                for path in p[i]:ax.plot(np.r_[0,path[:,0]],np.r_[0,path[:,1]],color='#C8D3DF',lw=.5)
                ax.plot(x['past'][i,:,0],x['past'][i,:,1],color='#333333',lw=1);ax.plot(np.r_[0,gt[i,:,0]],np.r_[0,gt[i,:,1]],'--',color='#111111',lw=1)
                ax.plot(np.r_[0,p[i,kj[i],:,0]],np.r_[0,p[i,kj[i],:,1]],color=COLORS[0],lw=1);ax.set_aspect('equal',adjustable='datalim')
                title=('COUNTEREXAMPLE\n' if row==2 else '')+label+'\nID '+str(x['sample_id'][i]);ax.set_title('\n'.join(textwrap.wrap(title,27)),fontsize=5)
                ps=np.linalg.norm(np.diff(np.vstack((np.zeros(2),p[i,kj[i]])),axis=0),axis=-1)/.4;gs=np.linalg.norm(np.diff(np.vstack((np.zeros(2),gt[i])),axis=0),axis=-1)/.4
                sp.plot(np.arange(1,13)*.4,ps,color=COLORS[0],label='Native kJ');sp.plot(np.arange(1,13)*.4,gs,'--',color='#111111',label='GT')
                sp.set_xlabel('cached time (s)',fontsize=5);sp.set_ylabel('speed (m/s)',fontsize=5);ax.set_xlabel('local x (m)',fontsize=5);ax.set_ylabel('local y (m)',fontsize=5)
            else:
                axs[row,col].text(.5,.5,'NO_CASE',ha='center',va='center',transform=axs[row,col].transAxes);axs[row,col].set_title(label+(' counterexample' if row==2 else ''),fontsize=5);axs[row+1,col].axis('off')
            for a in (axs[row,col],axs[row+1,col]):a.tick_params(labelsize=5)
    fig.suptitle('GT-posthoc diagnostic cases + counterexamples | grey=K20, blue=oracle-selected kJ, dashed=GT',fontsize=7)
    export(fig,'diagnostic_cases');table(OUT/'CASE_SELECTION.csv',selected)
    write(OUT/'FIGURE_QA.json',dict(status='RENDERED_REQUIRES_ACTUAL_VISUAL_QA',backend='python',PNG_dpi=600,SVG_PDF_editable_text=True,no_subsampling_of_analysis=True,boxplot_outlier_marks_omitted_only=True,fixed_hash_cases=True,GT_oracles_not_prediction=True,all683_fixed_failure_den=True))
    resource(OUT/'RESOURCE.json',g)
if __name__=='__main__':main()
