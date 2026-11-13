from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,'axes.spines.top':False,'axes.spines.right':False})
def main():
    # Explicit E1-only statistical/figure extension; no GT files are opened.
    x=load(HERE/'inputs/X_hold.npz');m=load(HERE/'E1_mode_distribution/hold/coarse20/modes.npz');bs=load(HERE/'statistics/DRAWS_hold.npz');ix=np.searchsorted(bs['tracks'],x['track_key']);K=len(bs['tracks']);rows=[]
    fig,axes=plt.subplots(1,3,figsize=(7.2047244,2.6377953),layout='constrained');colors=['#808080','#9E8ABD','#4C78A8','#4E9B81','#D89A58']
    for gi,(group,ax) in enumerate(zip(GROUPS[1:],axes)):
        mask=x['group']==gi;den=np.array([np.sum((ix==k)&mask) for k in range(K)]);dd=den[bs['draws']].sum(1);vals=[];lows=[];highs=[]
        for label in range(5):
            f=(m['mode']==label).mean(1);num=np.array([f[(ix==k)&mask].sum() for k in range(K)]);v=np.divide(num[bs['draws']].sum(1),dd,out=np.full(20000,np.nan),where=dd>0);ok=np.isfinite(v);ci=np.quantile(v[ok],[.025,.975]);mean=float(f[mask].mean());vals.append(mean);lows.append(float(ci[0]));highs.append(float(ci[1]));rows.append(dict(group=group,mode=COARSE[label],windows=int(mask.sum()),tracks=len(np.unique(x['track_key'][mask])),mean_fraction=mean,CI95_low=float(ci[0]),CI95_high=float(ci[1]),invalid=int((~ok).sum())))
        ax.bar(range(5),vals,color=colors);ax.vlines(range(5),lows,highs,lw=.7,color='black');ax.hlines(lows,np.arange(5)-.08,np.arange(5)+.08,lw=.7,color='black');ax.hlines(highs,np.arange(5)-.08,np.arange(5)+.08,lw=.7,color='black');ax.set(xticks=range(5),xticklabels=COARSE,ylim=(0,1),ylabel='Mean candidate fraction',title=group);ax.tick_params(axis='x',labelrotation=45,labelsize=6)
    fig.suptitle('Native shape occupancy: 20,000 whole-track descriptive 95% CI',fontsize=8)
    out=HERE/'figures_final/E1'
    fig.savefig(out/'E1_class_track_CI.svg',bbox_inches='tight',facecolor='white')
    fig.savefig(out/'E1_class_track_CI.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(out/'E1_class_track_CI.png',bbox_inches='tight',dpi=600,facecolor='white')
    plt.close(fig);table(HERE/'completion_review/E1_CLASS_CI_v2.csv',rows);write(HERE/'completion_review/E1_CLASS_CI_ACCESS_v2.json',dict(status='PASS',parsed_data_files=['inputs/X_hold.npz','E1_mode_distribution/hold/coarse20/modes.npz','statistics/DRAWS_hold.npz'],no_GT_read=True,E1_done_remains_unchanged=True,source_extension='only missing per-class intervals and display; no category or threshold changes'))
    print('PASS E1 CLASS CI EXTENSION',flush=True)
if __name__=='__main__':main()


