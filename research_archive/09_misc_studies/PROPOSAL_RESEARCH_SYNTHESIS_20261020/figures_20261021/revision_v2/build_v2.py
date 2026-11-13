"""Two model architecture schematics; no measurements or performance claims.
Style-only adaptation of the supplied architecture reference.
"""
from pathlib import Path
import sys
import json
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_figures as base
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial'],
                     'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
INK='#304252'; BLUE='#5D8EBC'; PINK='#B584A2'; GREEN='#648C62'
LB='#EFF5FB'; LP='#F4EFF8'; LG='#F1F7EF'; GRAY='#92A0AC'

class Figure(base.Canvas):
    def __init__(self,name,height):
        self.name,self.items=name,[]
        self.height=height
        self.fig=plt.figure(figsize=(16,height/100),dpi=300)
        self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set(xlim=(0,1600),ylim=(height,0));self.ax.axis('off')
        self.root=ET.Element('mxfile',host='app.diagrams.net')
        d=ET.SubElement(self.root,'diagram',name=name,id=name)
        m=ET.SubElement(d,'mxGraphModel',page='1',pageWidth='1600',pageHeight=str(height))
        self.cells=ET.SubElement(m,'root');ET.SubElement(self.cells,'mxCell',id='0');ET.SubElement(self.cells,'mxCell',id='1',parent='0')
    def box(self,x,y,w,h,label,fill=LB,stroke=BLUE,size=19):
        self.rect(x,y,w,h,fill,stroke,6);self.text(x+w/2,y+h/2,label,size,color=INK)
    def group(self,x,y,w,h,label,stroke=BLUE):
        self.rect(x,y,w,h,'#FCFDFE',stroke,9);self.text(x+18,y+25,label,21,bold=True,align='left')
    def arrow(self,*points,color=INK,dash=False):self.line(list(points),color,dash,True,1)
    def sum(self,x,y):
        self.dot(x,y,11,'white')
        self.cells[-1].set('style','ellipse;fillColor=#FFFFFF;strokeColor=#304252;')
        self.ax.add_patch(plt.Circle((x,y),10,fill=False,color=INK,lw=.8))
        self.line([(x-6,y),(x+6,y)],INK,lw=.7);self.line([(x,y-6),(x,y+6)],INK,lw=.7)
    def grid(self,x,y,n=5,m=6,mask=False):
        for i in range(n):
            for j in range(m):
                active=(j in [0,2,4]) if mask else ((i*3+j*2)%5<3)
                self.rect(x+j*13,y+i*13,11,11,'#92B2CF' if active else '#EDF1F5','#D6DFE8',0)
    def trajectories(self,x,y,w=130,h=105):
        # Illustrative paths only, not experimental observations.
        for k,c in enumerate([BLUE,PINK,GREEN,'#C79562','#8C83AD']):
            pts=[(x+w*t/12,y+h/2-(k-2)*h*.12*(t/12)**1.7-15*t/12) for t in range(13)]
            self.line(pts,c,lw=1.4)
        self.dot(x,y+h/2,4,INK)
    def save(self):
        self.fig.savefig(OUT/(self.name+'.svg'),facecolor='white')
        self.fig.savefig(OUT/(self.name+'.pdf'),facecolor='white')
        self.fig.savefig(OUT/(self.name+'.png'),dpi=300,facecolor='white')
        ET.ElementTree(self.root).write(OUT/(self.name+'.drawio'),encoding='utf-8',xml_declaration=True)
        data={'meta':{'figureType':'architecture','profile':'academic-paper','canvas':[1600,self.height]},'primitives':self.items}
        for ext in ['arch.json','spec.yaml']:(OUT/(self.name+'.'+ext)).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        plt.close(self.fig)

def overview():
    f=Figure('Fig1_model_framework_v2',925)
    f.text(25,28,'(a)  模型整体框架',23,bold=True,align='left')
    f.group(330,62,770,352,'运动与交互编码器')
    f.group(1120,62,455,352,'多模态非自回归解码器',GREEN)
    # Input histories: 3 observed agents, 8 observations each.
    for k,c in enumerate([BLUE,GREEN,PINK]):
        pts=[(32+t*11,218+k*27-t*(3+k*.5)) for t in range(8)]
        f.line(pts,c)
        for x,y in pts:f.dot(x,y,2.9,c)
    f.text(75,297,'历史轨迹',19,bold=True);f.text(75,326,'8 × N × 2',17,color=GRAY)
    f.arrow((120,245),(161,245))
    f.box(173,214,132,62,'局部坐标\n归一化',size=18)
    f.arrow((305,245),(350,245))
    f.box(363,214,137,62,'运动状态\n嵌入',size=19)
    f.text(431,303,'位置 · 速度 · 加速度',14,color=GRAY)
    f.arrow((500,245),(532,245));f.box(544,214,135,62,'时间编码',size=20)
    f.text(611,302,'逐行人时序表征',15,color=GRAY)
    f.arrow((679,245),(719,245))
    f.box(731,207,170,76,'稀疏时空注意力',LP,PINK,18)
    f.text(816,305,'见 (b)',16,color=PINK)
    f.arrow((901,245),(945,245))
    f.box(958,214,122,62,'门控融合',size=19)
    f.arrow((679,245),(697,245),(697,356),(1019,356),(1019,276))
    f.text(825,376,'时序残差',15,color=GRAY)
    # Graph construction from normalized observed trajectories, not latent future.
    f.arrow((305,245),(319,245),(319,143),(363,143))
    f.box(363,113,164,61,'运动关系编码',size=18)
    f.arrow((527,143),(564,143))
    f.box(577,113,173,61,'自适应有向图',size=18)
    f.arrow((750,143),(816,143),(816,207))
    f.text(895,143,'关系 / 掩码',15,color=GRAY)
    f.arrow((1080,245),(1142,245))
    f.box(1154,207,192,76,'并行候选生成',LG,GREEN,20)
    f.text(1250,303,'解码细节见图2',15,color=GRAY)
    f.box(1169,113,162,53,'K 个模式查询',LG,GREEN,18)
    f.arrow((1250,166),(1250,207))
    f.arrow((1346,245),(1384,245));f.trajectories(1395,191,147,108)
    f.text(1470,312,'K × 12 × 2',18);f.text(1470,341,'候选轨迹',17)
    f.box(1154,333,192,48,'独立评分分支',LG,GREEN,18)
    f.arrow((1142,245),(1142,357),(1154,357))
    f.arrow((1346,357),(1401,357));f.text(1468,372,'K 个评分',17)
    # Lower mechanism panel.
    f.text(25,455,'(b)  稀疏时空注意力',23,bold=True,align='left')
    f.group(25,487,1550,386,'目标查询 × 邻居历史',PINK)
    f.box(52,548,160,53,'目标时序摘要',size=17)
    f.box(52,645,160,53,'邻居历史特征',size=17)
    f.box(52,752,160,53,'运动关系',LP,PINK,17)
    f.arrow((212,574),(259,574));f.box(271,551,69,46,'WQ',size=18)
    f.arrow((212,671),(236,671),(236,648),(271,648));f.box(271,625,69,46,'WK',size=18)
    f.arrow((236,671),(236,721),(271,721));f.box(271,698,69,46,'WV',size=18)
    f.arrow((340,574),(399,574),(399,615),(434,615))
    f.arrow((340,648),(399,648),(399,615))
    f.box(446,580,134,70,'QK^T / √d',size=20)
    f.arrow((580,615),(624,615));f.sum(636,615)
    f.arrow((212,778),(446,778));f.box(446,754,134,48,'关系偏置',LP,PINK,17)
    f.arrow((580,778),(636,778),(636,626))
    f.arrow((647,615),(694,615));f.box(706,580,183,70,'Mask + Softmax',LP,PINK,18)
    f.box(706,695,183,45,'有向图 / 有效掩码',LP,PINK,16)
    f.arrow((797,695),(797,650))
    f.arrow((889,615),(929,615));f.grid(944,581,5,6,True)
    f.text(983,677,'稀疏权重 A',16,color=GRAY)
    f.arrow((1025,615),(1065,615));f.box(1077,580,145,70,'加权聚合',size=20)
    f.box(706,754,183,48,'关系值映射',LP,PINK,17)
    f.arrow((430,778),(430,818),(675,818),(675,778),(706,778))
    f.arrow((340,721),(365,721),(365,850),(1009,850))
    f.sum(1020,850)
    f.arrow((889,778),(1020,778),(1020,839))
    f.arrow((1031,850),(1149,850),(1149,650))
    f.text(1196,809,'V + 关系值',15,color=GRAY)
    f.arrow((1222,615),(1258,615));f.box(1270,589,73,52,'WO',size=19)
    f.arrow((1343,615),(1382,615));f.box(1394,580,149,70,'交互表征',size=19)
    f.text(1467,692,'→ 门控融合',16,color=GRAY)
    f.text(800,903,'N：有效行人数     K = 20：候选数     A：在所选邻居及其历史时间维上归一化',15,color=GRAY)
    f.save()

def decoder():
    f=Figure('Fig2_decoder_architecture_v2',815)
    f.text(25,28,'(a)  集合感知非自回归解码器',23,bold=True,align='left')
    f.group(25,65,1550,325,'共享上下文 → 候选特征 → 完整路径',GREEN)
    f.box(52,133,157,56,'编码特征 H',size=19)
    f.box(52,269,157,56,'历史运动状态',size=18)
    f.arrow((209,161),(238,161),(238,223),(274,223))
    f.arrow((209,297),(238,297),(238,223))
    f.box(286,191,155,64,'上下文融合',size=19)
    f.text(363,291,'末时刻 + 时序池化',15,color=GRAY)
    f.arrow((441,223),(484,223));f.sum(497,223)
    f.box(443,111,161,50,'K 个模式查询',LG,GREEN,18)
    f.arrow((523,161),(523,180),(497,180),(497,212))
    f.arrow((508,223),(549,223));f.box(561,191,133,64,'模式 MLP',size=19)
    f.arrow((694,223),(732,223));f.box(744,183,185,80,'候选集合交互',LP,PINK,20)
    f.text(836,301,'见 (b)',16,color=PINK)
    f.arrow((929,223),(968,223));f.box(980,183,175,80,'三锚点路径头',LG,GREEN,20)
    f.text(1067,301,'t = 4, 8, 12',17,color=GRAY)
    f.arrow((1155,223),(1193,223));f.box(1205,191,139,64,'插值 / 细化',LG,GREEN,18)
    f.arrow((1344,223),(1380,223));f.trajectories(1393,166,150,111)
    f.text(1466,312,'K × 12 × 2',19)
    f.text(1466,343,'并行完整路径',16,color=GRAY)
    f.arrow((694,223),(711,223),(711,346),(980,346))
    f.box(980,321,175,50,'独立评分头',LG,GREEN,18)
    f.arrow((1137,263),(1137,321))
    f.arrow((1155,346),(1193,346));f.text(1266,346,'K 个评分',18)
    # Detailed block uses no hardcoded transformer recipe that has not been selected.
    f.text(25,432,'(b)  候选集合交互',23,bold=True,align='left')
    f.group(25,466,1550,285,'候选维信息交换',PINK)
    f.stack(76,558);f.text(113,655,'Z0',20);f.text(113,690,'K × d',16,color=GRAY)
    f.arrow((144,585),(191,585));f.box(203,551,164,68,'共享特征映射',size=18)
    f.arrow((367,585),(411,585))
    f.rect(423,528,281,130,LP,PINK,7)
    points=[(450,566),(509,548),(575,557),(666,564),(476,629),(556,628),(640,626)]
    for i,j in [(0,1),(1,2),(2,3),(0,4),(1,5),(2,6),(3,6),(4,5),(5,6)]:f.line([points[i],points[j]],PINK,lw=1)
    for p in points:f.dot(*p,7,PINK)
    f.text(563,693,'候选信息混合',18)
    f.arrow((704,585),(747,585));f.box(759,551,174,68,'候选更新映射',LP,PINK,18)
    f.arrow((933,585),(986,585));f.sum(998,585)
    f.arrow((173,585),(173,716),(998,716),(998,596))
    f.text(569,737,'候选特征残差',15,color=GRAY)
    f.arrow((1009,585),(1057,585));f.stack(1080,559,PINK)
    f.text(1117,655,'Z1',20);f.text(1117,690,'K × d',16,color=GRAY)
    f.arrow((1157,585),(1204,585));f.box(1216,539,315,93,'三锚点路径头\n全部候选同时生成',LG,GREEN,20)
    f.text(1373,686,'共享参数，无逐时刻反馈',16,color=GRAY)
    f.text(800,789,'H：编码特征     Z0 / Z1：交互前 / 后的候选特征     ⊕：逐元素相加     K = 20',15,color=GRAY)
    f.save()

if __name__=='__main__':
    overview();decoder();print('Exported v2 architecture figures.')
