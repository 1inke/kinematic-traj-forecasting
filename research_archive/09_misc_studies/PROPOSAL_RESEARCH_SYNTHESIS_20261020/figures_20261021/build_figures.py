"""Proposal schematics; all trajectory glyphs are illustrative, not observations.

Evidence contract: depict proposed architecture, never measured improvement.
Python rendering, editable SVG text, vector PDF, and matching draw.io primitives.
"""
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'Arial'], 'font.size': 10, 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'axes.unicode_minus': False})
C = dict(ink='#293444', muted='#64748B', blue='#6096BA', pink='#BA768D',
         teal='#579A92', violet='#9185B5', line='#B5C6DB', pale='#F5F9FD',
         rose='#FBF3F6', green='#F0F8F6', white='#FFFFFF')

class Canvas:
    def __init__(self, name, title, subtitle):
        self.name, self.items = name, []
        self.fig = plt.figure(figsize=(16, 9.4), dpi=300)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0,1600), ylim=(940,0)); self.ax.axis('off')
        self.root = ET.Element('mxfile', host='app.diagrams.net')
        diagram = ET.SubElement(self.root,'diagram',name=title,id=name)
        model = ET.SubElement(diagram,'mxGraphModel',page='1',pageWidth='1600',pageHeight='940')
        self.cells = ET.SubElement(model,'root')
        ET.SubElement(self.cells,'mxCell',id='0')
        ET.SubElement(self.cells,'mxCell',id='1',parent='0')
        self.text(38,35,title,25,bold=True,align='left')
        self.text(38,72,subtitle,15,color=C['muted'],align='left')
    def cell(self, kind, **attrs):
        ident = 'n'+str(len(self.items)+2)
        self.items.append(dict(id=ident,kind=kind,**attrs))
        return ident
    def rect(self,x,y,w,h,fill=None,stroke=None,rounding=10):
        fill,stroke=fill or C['white'],stroke or C['line']
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounding}',
                                       facecolor=fill,edgecolor=stroke,linewidth=.85))
        i=self.cell('rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke)
        cell=ET.SubElement(self.cells,'mxCell',id=i,parent='1',vertex='1',value='',
            style=f'rounded=1;arcSize=8;fillColor={fill};strokeColor={stroke};')
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
    def text(self,x,y,s,size=18,color=None,bold=False,align='center'):
        color=color or C['ink']
        self.ax.text(x,y,s,fontsize=size*.72,color=color,weight='bold' if bold else 'normal',
                     ha=align,va='center',linespacing=1.6)
        w=max(max(map(len,s.split('\n')))*size*1.05,20); h=(s.count('\n')+1)*size*1.6
        tx=x if align=='left' else x-w/2
        i=self.cell('text',x=tx,y=y-h/2,w=w,h=h,label=s,size=size)
        cell=ET.SubElement(self.cells,'mxCell',id=i,parent='1',vertex='1',value=s,
            style=f'text;html=0;whiteSpace=wrap;align={align};verticalAlign=middle;fillColor=none;strokeColor=none;fontFamily=Microsoft YaHei;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};')
        ET.SubElement(cell,'mxGeometry',x=str(tx),y=str(y-h/2),width=str(w),height=str(h),attrib={'as':'geometry'})
    def line(self,points,color=None,dash=False,arrow=False,lw=1.15):
        color=color or C['ink']
        xs,ys=zip(*points)
        self.ax.plot(xs,ys,color=color,lw=lw,ls=(0,(4,4)) if dash else '-',solid_capstyle='round')
        if arrow:
            self.ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=9,
                color=color,linewidth=lw))
        i=self.cell('edge',points=points,color=color,dashed=dash,arrow=arrow)
        cell=ET.SubElement(self.cells,'mxCell',id=i,parent='1',edge='1',value='',
            style=f'edgeStyle=none;rounded=0;strokeColor={color};strokeWidth={lw};endArrow={"block" if arrow else "none"};dashed={int(dash)};')
        g=ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
        for p,role in [(points[0],'sourcePoint'),(points[-1],'targetPoint')]:
            ET.SubElement(g,'mxPoint',x=str(p[0]),y=str(p[1]),attrib={'as':role})
        if len(points)>2:
            ar=ET.SubElement(g,'Array',attrib={'as':'points'})
            for p in points[1:-1]: ET.SubElement(ar,'mxPoint',x=str(p[0]),y=str(p[1]))
    def dot(self,x,y,r=6,color=None):
        color=color or C['blue']; self.ax.add_patch(Circle((x,y),r,facecolor=color,edgecolor='white',lw=.5))
        i=self.cell('ellipse',x=x-r,y=y-r,w=2*r,h=2*r,fill=color)
        cell=ET.SubElement(self.cells,'mxCell',id=i,parent='1',vertex='1',value='',style=f'ellipse;fillColor={color};strokeColor=#FFFFFF;')
        ET.SubElement(cell,'mxGeometry',x=str(x-r),y=str(y-r),width=str(2*r),height=str(2*r),attrib={'as':'geometry'})
    def panel(self,x,y,w,h,num,title,fill=None):
        self.rect(x,y,w,h,fill)
        self.dot(x+26,y+28,12,C['ink']); self.text(x+26,y+28,num,14,color='white',bold=True)
        self.text(x+50,y+28,title,19,bold=True,align='left')
    def stack(self,x,y,color=None,n=4):
        color=color or C['blue']
        for k in range(n): self.rect(x+k*13,y-k*3,9,52,fill='#EAF2F8' if color==C['blue'] else '#F4E4EB',stroke=color,rounding=0)
    def label(self,x,y,title,detail):
        self.text(x,y,title,18,bold=True); self.text(x,y+28,detail,14,color=C['muted'])
    def paths(self,x,y,w=130,h=90,multi=True,color=None):
        self.line([(x,y+h),(x+w,y+h)],'#DAE2EB',lw=.7)
        self.line([(x,y+h),(x,y)],'#DAE2EB',lw=.7)
        for k in range(5 if multi else 1):
            pts=[(x+w*t/12,y+h-(h*.65*t/12+(k-2)*h*.11*(t/12)**2)) for t in range(13)]
            self.line(pts,color or [C['blue'],C['teal'],C['pink'],C['violet'],C['muted']][k],lw=1.3)
        self.dot(x,y+h,4,C['ink'])
    def legend(self):
        self.rect(32,844,1536,64)
        self.line([(60,875),(110,875)],arrow=True); self.text(125,875,'推理数据流',15,align='left')
        self.line([(290,875),(340,875)],C['pink'],dash=True,arrow=True); self.text(355,875,'仅训练监督',15,align='left')
        for x,color,label in [(565,C['blue'],'观测／共享表征'),(865,C['pink'],'拟研究生成模块'),(1165,C['teal'],'输出／验证')]:
            self.rect(x,865,18,20,fill=color,stroke=color,rounding=0); self.text(x+30,875,label,15,align='left')
        self.text(800,927,'技术路线示意，非实验结果；轨迹图形仅作说明，不代表已验证的覆盖或加速收益。',13,color=C['muted'])
    def save(self):
        self.legend()
        self.fig.savefig(OUT/(self.name+'.svg'),facecolor='white')
        self.fig.savefig(OUT/(self.name+'.pdf'),facecolor='white')
        self.fig.savefig(OUT/(self.name+'.png'),dpi=300,facecolor='white')
        ET.ElementTree(self.root).write(OUT/(self.name+'.drawio'),encoding='utf-8',xml_declaration=True)
        spec={'meta':{'profile':'academic-paper','figureType':'architecture','theme':'academic-color',
                      'canvas':[1600,940],'renderer':'build_figures.py','evidence':'proposed, not validated'},'primitives':self.items}
        # JSON is also valid YAML 1.2; primitive sidecars preserve the exact editable layout.
        (OUT/(self.name+'.spec.yaml')).write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
        (OUT/(self.name+'.arch.json')).write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
        plt.close(self.fig)

def fig1():
    f=Canvas('Fig1_sparse_motion_framework','图1  运动相关稀疏交互与轻量化预测框架',
             '拟创新点一：在有限交互预算内保留关键运动信息；与固定预算并行生成共同构成技术路线。')
    f.rect(32,110,1152,704,C['pale']); f.text(60,139,'目标行人及其局部交互',17,align='left')
    f.panel(52,168,1112,278,'1','历史运动状态与有向关系建模')
    # Three observed tracks, explicitly past only.
    for k,c in enumerate([C['ink'],C['blue'],C['teal']]):
        pts=[(91+t*11,274+k*25-t*(3+k*.8)) for t in range(8)]
        f.line(pts,c)
        for px,py in pts: f.dot(px,py,3,c)
    f.label(133,365,'过去8点','目标＋有效邻居')
    f.line([(193,282),(247,282)],arrow=True)
    f.stack(273,254); f.label(306,365,'运动状态编码','位置 · 速度 · 加速度')
    f.line([(352,282),(411,282)],arrow=True)
    f.rect(432,245,196,87,C['pale']); f.text(530,274,'相对位置 / 速度',17); f.text(530,305,'接近关系与方向性',16)
    f.label(530,365,'运动关系建模','只读取可见历史')
    f.line([(648,282),(701,282)],arrow=True)
    center=(774,285); neighbors=[(727,235),(817,227),(837,288),(795,333),(720,332)]
    for k,p in enumerate(neighbors):
        f.line([p,center],C['blue'] if k<3 else '#BFC8D2',dash=k>=3,arrow=k<3)
        f.dot(*p,8,C['blue'] if k<3 else '#CCD4DE')
    f.dot(*center,11,C['pink']); f.label(776,365,'有向稀疏邻居','保留相关关系＋自环')
    f.line([(856,282),(908,282)],arrow=True)
    f.rect(930,243,192,91,C['green']); f.text(1026,269,'有限邻居预算',18,bold=True); f.text(1026,305,'有效性 mask / 索引',15)
    f.label(1026,365,'选中交互 token','稀疏执行须单独验证')
    f.text(610,418,'关系稀疏 ≠ 实测加速；与距离近邻、稠密交互进行同预算对照。',15,color=C['muted'])
    f.line([(1026,446),(1026,469),(73,469),(73,549),(151,549),(151,559)],C['blue'],arrow=True)
    f.panel(52,493,1112,297,'2','共享表征与非自回归完整路径预测')
    f.stack(127,564); f.label(160,686,'历史与邻居 token','保留时序信息')
    f.line([(203,591),(262,591)],arrow=True)
    f.rect(281,551,225,87,C['pale']); f.text(393,577,'时空交互聚合',19,bold=True); f.text(393,612,'共享的轻量运动表征',16)
    f.label(393,686,'上下文 h','同一表征服务全部候选')
    f.line([(526,591),(584,591)],arrow=True)
    f.stack(611,566,C['pink']); f.label(647,686,'K = 20 模式查询','读取共享上下文')
    f.line([(691,591),(745,591)],arrow=True)
    f.rect(763,548,173,91,C['rose']); f.text(849,577,'集合感知',19,bold=True); f.text(849,610,'并行解码（见图2）',15)
    f.line([(952,591),(990,591)],arrow=True)
    f.paths(1005,550,120,85); f.label(1063,686,'20条 × 未来12点','完整时域同时输出')
    f.text(610,753,'不依赖逐个未来时间步反馈；候选生成与最终评分分开评价。',15,color=C['muted'])
    f.panel(1210,110,358,704,'3','预算约束与验证',C['white'])
    for y,title,lines,fill in [(177,'信息效率','运动相关邻居 vs 距离邻居\n稀疏 vs 稠密交互\n同参数量／同输入对照',C['pale']),
                              (370,'输出效率','固定 K = 20\n完整路径覆盖与成功保留\n运动学与有效候选检查',C['green']),
                              (563,'部署成本','整模参数量 / MAC\n实际延迟 P50 / P95\n峰值内存与图构造开销',C['rose'])]:
        f.rect(1230,y,318,174,fill); f.text(1389,y+31,title,19,bold=True); f.text(1389,y+106,lines,17)
    f.text(1389,781,'目标：信息有效，而非单纯变稀疏',14,color=C['muted'])
    f.save()

def fig2():
    f=Canvas('Fig2_complementary_parallel_decoder','图2  固定候选预算的集合感知互补非自回归生成',
             '拟创新点二：在生成过程中协调候选分工；以新增有效覆盖和原有成功保留验证互补性。')
    f.rect(32,110,1152,704,C['pale'])
    f.panel(52,130,1112,343,'1','推理：共享上下文 → 候选通信 → 并行完整路径')
    f.rect(80,222,166,68,C['pale']); f.text(163,247,'共享运动上下文 h',16,bold=True); f.text(163,274,'来自图1编码端',14)
    f.stack(133,327,C['pink']); f.text(163,405,'20个模式查询 Q',16)
    f.line([(246,256),(282,256),(282,302),(314,302)],arrow=True)
    f.line([(195,351),(282,351),(282,302)],arrow=False)
    f.stack(338,277); f.label(370,399,'条件候选特征','Z0 = Condition(Q, h)')
    f.line([(415,302),(461,302)],arrow=True)
    f.rect(478,207,244,161,C['rose'])
    dots=[(516,252),(568,242),(628,248),(681,261),(543,316),(610,326),(665,313)]
    for i,j in [(0,1),(1,2),(2,3),(0,4),(1,5),(2,5),(3,6),(4,5),(5,6)]:f.line([dots[i],dots[j]],'#C9A3B2',lw=.85)
    for p in dots:f.dot(*p,8,C['pink'])
    f.text(600,345,'轻量集合信息交换',16,bold=True)
    f.label(600,399,'SetMix + 残差更新','候选间通信，不是时间自回归')
    f.line([(738,302),(780,302)],arrow=True)
    f.stack(802,277,C['pink']); f.label(836,399,'共享并行解码器','Z1 → 全部未来点')
    f.line([(881,302),(939,302)],arrow=True)
    f.paths(966,243,155,109); f.label(1043,399,'固定20条完整路径','候选数不因互补而增加')
    f.text(607,448,'推理输入仅来自观测；真实未来不进入编码、候选通信或生成。',15,color=C['muted'])
    f.panel(52,493,1112,297,'2','训练与评价：有效覆盖，而非仅拉大轨迹间距离')
    f.paths(91,558,103,73,multi=False,color=C['ink']); f.label(143,672,'真实未来','只作训练监督／评价')
    f.line([(211,597),(264,597)],C['pink'],dash=True,arrow=True)
    for x,title,sub in [(281,'完整路径质量','同一候选同时满足\nADE 与 FDE 标准'),(551,'补全未覆盖样例','rescue：原失败 → 成功\n不是只追求分散度'),(821,'保留既有成功','lost：原成功 → 失败\n同时检查运动学')]:
        f.rect(x,549,248,129,C['rose'] if x==281 else C['green'])
        f.text(x+124,577,title,18,bold=True); f.text(x+124,632,sub,16)
    f.text(607,720,'候选责任与保留约束须逐因素验证；蒸馏可选，不是互补机制的必要条件。',16)
    f.text(607,758,'集合评价与评分分离：有无好路径 ≠ 是否选中；不将旧 logits 当作新路径概率。',15,color=C['muted'])
    f.panel(1210,110,358,704,'3','独立验证与边界')
    f.rect(1230,175,318,181,C['pale']); f.text(1389,205,'2 × 2 因素对照',19,bold=True)
    f.text(1329,245,'普通解码',14); f.text(1463,245,'集合感知',14)
    f.text(1257,280,'稠密',14); f.text(1257,320,'稀疏',14)
    for x in [1329,1463]:
        for y in [280,320]:f.rect(x-40,y-13,80,26,fill='#EAF1F7',rounding=2);f.text(x,y,'同 K',14)
    f.rect(1230,375,318,185,C['green']); f.text(1389,405,'核心证据',19,bold=True)
    f.text(1389,478,'覆盖增加且丢失受控\n五项运动学指标\n同容量与同计算预算',17)
    f.rect(1230,579,318,175,C['rose']); f.text(1389,609,'不能替代的检查',19,bold=True)
    f.text(1389,683,'K24 不作等预算收益\n不默认低分候选冗余\n不以分散度代表互补',17)
    f.text(1389,783,'当前是研究假设，非已完成创新',14,color=C['muted'])
    f.save()

if __name__ == '__main__':
    fig1(); fig2()
    print('Generated two SVG/PDF/PNG/drawio bundles.')
