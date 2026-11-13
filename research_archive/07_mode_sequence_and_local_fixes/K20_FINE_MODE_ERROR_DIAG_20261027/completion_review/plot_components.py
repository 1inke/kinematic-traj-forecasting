from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
def main():
    x=xload('hold');q=load(HERE/'E3_gt_audit/hold/native_quality.npz');r=load(HERE/'E3_gt_audit/hold/a20_history3_s2/presence.npz');e=load(HERE/'completion_review/error_reviewed/hold.npz');f=load(HERE/'E3_gt_free/hold/a20_history3_s2.npz');gf=load(HERE/'E3_gt_audit/hold/a20_history3_s2/gt_features.npz');mask=x['group']==2
    e['absolute_turn_difference_deg']=(f['abs_local_turn']-gf['abs_local_turn'])*180/np.pi
    specs=[('longitudinal_error','signed longitudinal end error (m)'),('lateral_error','signed lateral end error (m)'),('internal_heading_error','internal heading error (deg)'),('entry_heading_error','entry heading error (deg)'),('early_speed_diff','early speed difference (m/s)'),('late_speed_diff','late speed difference (m/s)'),('path_length_diff','path length difference (m)'),('absolute_turn_difference_deg','absolute-turn proxy difference (deg)')]
    fig,axs=plt.subplots(2,4,figsize=(183/25.4,108/25.4),layout='constrained');rows=[]
    for ax,(key,title) in zip(axs.ravel(),specs):
        sets=[];labels=[]
        for name,base,k in [('success kJ',mask&q['native_good'],e['k_J']),('fine-fail kJ',mask&~q['native_good']&(r['reason']==4),e['k_J']),('fine-fail kMode',mask&~q['native_good']&(r['reason']==4),e['k_mode'])]:
            ix=np.flatnonzero(base&(k>=0));v=e[key][ix,k[ix]];v=v[np.isfinite(v)];sets.append(v);labels.append(name+'\nn='+str(len(v)));rows.append(dict(metric=key,reference=name,N_valid=len(v),median=float(np.median(v)) if len(v) else None,q25=float(np.quantile(v,.25)) if len(v) else None,q75=float(np.quantile(v,.75)) if len(v) else None))
        active=[i for i,v in enumerate(sets) if len(v)]
        ax.boxplot([sets[i] for i in active],positions=active,showfliers=False,medianprops={'color':'#4C78A8'})
        ax.set_xticks(range(3),labels,rotation=60,ha='right',fontsize=5);ax.set_title(title,fontsize=6);ax.tick_params(labelsize=6);ax.axhline(0,color='#B3B3B3',lw=.5)
    fig.suptitle('hold change | matched241 fine-mode failures: kJ vs kMode; signed errors can cancel',fontsize=8)
    folder=HERE/'figures_v2';fig.savefig(folder/'error_components.svg');fig.savefig(folder/'error_components.pdf');fig.savefig(folder/'error_components.png',dpi=600);plt.close(fig)
    table(folder/'ERROR_COMPONENT_SOURCE.csv',rows)
if __name__=='__main__':main()
