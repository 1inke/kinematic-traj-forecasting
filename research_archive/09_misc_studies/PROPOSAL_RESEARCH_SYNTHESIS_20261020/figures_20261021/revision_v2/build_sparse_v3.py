"""Single attention schematic, preserving the actual score/message distinction.
No experimental data. Python SVG with editable text; PNG for internal QA only.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from build_v2 import Figure, INK, BLUE, PINK, LB, LP, GRAY

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial'],
                     'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
out=Path(__file__).resolve().parent
f=Figure('Sparse_temporal_attention_v3',640)
f.group(20,20,1560,594,'稀疏时空注意力',PINK)

# The content-matching operation remains; only the formula label is removed.
f.box(45,91,160,58,'目标时序摘要',size=18)
f.box(45,211,160,58,'邻居历史特征',size=18)
f.arrow((205,120),(270,120)); f.box(282,96,72,48,'WQ',size=19)
f.arrow((205,240),(270,240)); f.box(282,216,72,48,'WK',size=19)
f.arrow((354,120),(418,120),(418,165),(459,165))
f.arrow((354,240),(418,240),(418,195),(459,195))
f.box(459,148,151,64,'特征匹配',size=20)
f.arrow((610,180),(677,180))
f.box(689,148,180,64,'关系感知评分',LP,PINK,19)
f.arrow((869,180),(916,180));f.box(928,148,178,64,'Mask + Softmax',LP,PINK,18)
f.arrow((1106,180),(1145,180));f.grid(1158,150,5,6,True)
f.text(1197,240,'稀疏权重',16,color=GRAY)

# Explicit relation fork: score bias above, message relation below.
f.box(459,356,151,58,'运动关系',LP,PINK,19)
f.arrow((610,385),(779,385),(779,316))
f.box(704,264,150,52,'关系偏置',LP,PINK,18)
f.arrow((779,264),(779,212))
f.dot(779,385,3.2,INK)
f.arrow((779,385),(779,464))
f.box(704,464,150,58,'关系值映射',LP,PINK,18)

f.box(928,274,178,58,'图与有效性掩码',LP,PINK,17)
f.arrow((1017,274),(1017,212))

# Neighbor history also creates V. Fusion is visibly BEFORE weighting.
f.line([(231,240),(231,560)],INK,lw=1)
f.dot(231,240,3.2,INK)
f.arrow((231,560),(282,560));f.box(282,536,72,48,'WV',size=19)
f.arrow((354,560),(1017,560),(1017,462))
f.text(505,541,'V：邻居历史值',16,color=GRAY)
f.box(928,398,178,64,'消息融合 ⊕',size=19)
f.arrow((854,493),(899,493),(899,430),(928,430))

f.arrow((1236,180),(1280,180),(1280,353))
f.box(1205,353,150,64,'加权聚合',size=20)
f.arrow((1106,430),(1160,430),(1160,385),(1205,385))
f.arrow((1355,385),(1390,385));f.box(1400,361,64,48,'WO',size=18)
f.arrow((1464,385),(1504,385))
f.stack(1514,358,BLUE,n=3)
f.text(1532,449,'交互表征',16)
f.text(1400,498,'⊕：特征相加',15,color=GRAY)

f.fig.savefig(out/'Sparse_temporal_attention_v3.svg',facecolor='white')
f.fig.savefig(out/'Sparse_temporal_attention_v3.preview.png',dpi=300,facecolor='white')
plt.close(f.fig)
print('Sparse_temporal_attention_v3.svg')
