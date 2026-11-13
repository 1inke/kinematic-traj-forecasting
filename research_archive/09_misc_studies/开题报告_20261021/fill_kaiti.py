# -*- coding: utf-8 -*-
"""将开题报告 md 内容填入《基础.docx》模板，输出填写版 docx。"""
import os, re, shutil
import docx
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC_MD = r'<LOCAL_DOWNLOADS>\开题报告.md'
TPL = r'<LOCAL_DESKTOP>\开题\基础.docx'
OUTDIR = r'<WORKSPACE_ROOT>\qoder_work\开题报告_20261021'
OUT = os.path.join(OUTDIR, '开题报告_填写版_20261021.docx')
CLEAN_MD = os.path.join(OUTDIR, '开题报告_整理版_20261021.md')

os.makedirs(OUTDIR, exist_ok=True)
shutil.copy(TPL, OUT)

raw = open(SRC_MD, encoding='utf-8').read()
lines = raw.split('\n')

def find(pred, start=0):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    raise ValueError('marker not found: ' + getattr(pred, '__name__', str(pred)))

# ---------------- 数学公式 -> 纯文本 ----------------
SUB = {'i': 'ᵢ', 'j': 'ⱼ', 'k': 'ₖ', 'n': 'ₙ',
       '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅'}
SUP = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '(': '⁽', ')': '⁾'}

def _sub(m):
    return ''.join(SUB.get(c, c) for c in m.group(1))

def _sup(m):
    inner = m.group(1)
    if all(c in SUP for c in inner):
        return ''.join(SUP[c] for c in inner)
    return '^(' + inner + ')'

def conv_math(s):
    s = s.replace('\\qquad', '，').replace('\\quad', ' ')
    s = s.replace('\\widehat{\\mathcal{Y}}', 'Ŷ')
    s = s.replace('\\mathcal{N}', 'N').replace('\\mathcal{Y}}', 'Y}')
    s = re.sub(r'\\mathcal\s*\{?\s*C\s*\}?', 'C', s)
    s = s.replace('\\mathbb{R}', 'R')
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'\\(?:text|mathrm|operatorname)\{([^{}]*)\}', r'\1', s)
    pairs = [('\\le', '≤'), ('\\ge', '≥'), ('\\times', '×'), ('\\land', '∧'),
             ('\\rightarrow', '→'), ('\\to', '→'), ('\\in', '∈'), ('\\sim', '~'),
             ('\\ldots', '…'), ('\\Delta', 'Δ'), ('\\theta', 'θ'), ('\\psi', 'ψ'),
             ('\\phi', 'φ'), ('\\exists', '∃'), ('\\sum', 'Σ')]
    for a, b in pairs:
        s = s.replace(a, b)
    s = s.replace('\\frac1n', '(1/n)').replace('\\mathbf1', '1')
    s = s.replace('\\big', '').replace('\\%', '%')
    s = s.replace('\\{', '{').replace('\\}', '}')
    s = re.sub(r'_\{([^{}]+)\}', _sub, s)
    s = re.sub(r'_([A-Za-z0-9])', lambda m: SUB.get(m.group(1), '_' + m.group(1)), s)
    s = re.sub(r'\^\{([^{}]+)\}', _sup, s)
    s = re.sub(r'([≤≥∧×])\s+', r'\1', s)
    s = re.sub(r'\s+([≤≥∧×])', r'\1', s)
    return s.strip()

def conv_inline(text):
    """转换行内 $...$ 公式，并按 ** 拆分为 (text, bold) 段。"""
    def _dol(m):
        return conv_math(m.group(1))
    text = re.sub(r'\$\$(.+?)\$\$', _dol, text, flags=re.S)
    text = re.sub(r'\$(.+?)\$', _dol, text)
    segs = []
    for k, part in enumerate(text.split('**')):
        if part:
            segs.append((part, k % 2 == 1))
    return segs

DISPLAY = [
    ('h_i = E_\\theta',
     'hᵢ = Eθ(Xᵢ, sᵢ, {Xⱼ, sⱼ, rᵢⱼ | j ∈ Nᵢ})'),
    ('Z_i^{(0)} = \\operatorname{Condition}',
     'Zᵢ⁽⁰⁾ = Condition(Q, hᵢ)，Zᵢ⁽¹⁾ = Zᵢ⁽⁰⁾ + SetMix_ψ(Zᵢ⁽⁰⁾, hᵢ)，Ŷᵢ = Decode_φ(Zᵢ⁽¹⁾) ∈ R^(K×Tf×2)'),
    ('mathrm{Cov}',
     'Cov(C) = (1/n)·Σᵢ 1[∃k ∈ {1,…,20}：ADEᵢₖ ≤ 0.2m ∧ FDEᵢₖ ≤ 0.4m]'),
]

# ---------------- Word 写入辅助 ----------------
def set_run_font(r, ea='宋体', latin='Times New Roman', size=12, bold=False):
    r.font.name = latin
    r.font.size = Pt(size)
    r.font.bold = bold
    rPr = r._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), ea)

def set_indent2(p):
    pPr = p._p.get_or_add_pPr()
    ind = pPr.find(qn('w:ind'))
    if ind is None:
        ind = OxmlElement('w:ind')
        pPr.append(ind)
    ind.set(qn('w:firstLineChars'), '200')
    ind.set(qn('w:firstLine'), '480')

def add_para(cell, segs, indent=True, align=None, size=12, line_pt=18,
             bold_all=False, ea='宋体', latin='Times New Roman', space_before=0):
    p = cell.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = Pt(line_pt)
    if space_before:
        pf.space_before = Pt(space_before)
    if indent:
        set_indent2(p)
    if align == 'center':
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if isinstance(segs, str):
        segs = [(segs, bold_all)]
    for txt, b in segs:
        r = p.add_run(txt)
        set_run_font(r, ea=ea, latin=latin, size=size, bold=b or bold_all)
    return p

def clear_cell(cell):
    for p in list(cell.paragraphs):
        p._element.getparent().remove(p._element)
    for t in list(cell.tables):
        t._tbl.getparent().remove(t._tbl)

def finalize_cell(cell):
    tc = cell._tc
    if len(tc) and tc[-1].tag == qn('w:tbl'):
        cell.add_paragraph()

def set_borders(tbl):
    tblPr = tbl._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement('w:' + edge)
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '4')
        el.set(qn('w:color'), '000000')
        borders.append(el)
    tblPr.append(borders)

def parse_md_table(block):
    rows = []
    for ln in block:
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{2,}:?', c or '---') for c in cells):
            continue
        rows.append(cells)
    return rows

def add_nested_table(doc, cell, rows):
    nt = cell.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        nt.style = doc.styles['Table Grid']
    except Exception:
        set_borders(nt)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = nt.rows[ri].cells[ci]
            segs = conv_inline(val)
            bold_row = any(b for _, b in segs)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = Pt(14)
            for txt, b in segs:
                r = p.add_run(txt)
                set_run_font(r, size=10.5, bold=b)
    return nt

# ---------------- 章节内容写入 ----------------
def process_section(doc, sec_lines, cell):
    i, in_code = 0, False
    while i < len(sec_lines):
        ln = sec_lines[i].rstrip()
        s = ln.strip()
        if s.startswith('```'):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            add_para(cell, [(ln.rstrip(), False)], indent=False, size=9,
                     line_pt=11, latin='Consolas')
            i += 1
            continue
        if not s or s == '---':
            i += 1
            continue
        if s.startswith('|'):
            block = []
            while i < len(sec_lines) and sec_lines[i].strip().startswith('|'):
                block.append(sec_lines[i].strip())
                i += 1
            add_nested_table(doc, cell, parse_md_table(block))
            continue
        if s.startswith('$$'):
            txt = None
            for key, rep in DISPLAY:
                if key in s:
                    txt = rep
                    break
            if txt is None:
                txt = conv_math(s.strip('$'))
            add_para(cell, [(txt, False)], indent=False, align='center')
            i += 1
            continue
        if s.startswith('#'):
            txt = s.lstrip('#').strip()
            add_para(cell, conv_inline(txt), indent=False, bold_all=True,
                     space_before=6)
            i += 1
            continue
        if re.match(r'^\*\*(图|表)\s*\d', s):
            add_para(cell, [(s.replace('*', ''), True)], indent=False, align='center')
            i += 1
            continue
        if s.startswith('> '):
            i += 1
            continue
        if s.startswith('* '):
            add_para(cell, conv_inline(s[2:]), indent=True)
            i += 1
            continue
        if re.match(r'^\[\d+\]', s):   # 参考文献
            add_para(cell, conv_inline(s), indent=False)
            i += 1
            continue
        add_para(cell, conv_inline(s), indent=True)
        i += 1
    finalize_cell(cell)

# ---------------- 抽取 md 各章节 ----------------
title_cn = lines[find(lambda s: '论文题目（中文）' in s)].split('：', 1)[1].strip()
title_en = lines[find(lambda s: '论文题目（英文）' in s)].split('：', 1)[1].strip()
i_abs = find(lambda s: s.strip().startswith('### 摘要'))
i_kw = find(lambda s: '关键词（中文）' in s)
abs_paras = [l.strip() for l in lines[i_abs + 1:i_kw]
             if l.strip() and not l.strip().startswith('*') and l.strip() != '---']
kw_cn = lines[i_kw].split('：', 1)[1].strip()
kw_en = lines[find(lambda s: '关键词（英文）' in s)].split('：', 1)[1].strip()

sec2 = lines[find(lambda s: s.strip().startswith('### 1. 学位论文研究意义')):
             find(lambda s: s.strip().startswith('## 六、论文工作计划'))]
sec3 = lines[find(lambda s: s.strip().startswith('### 1、研究内容')):
             find(lambda s: s.strip().startswith('# 陕西师范大学硕士研究生学位论文开题报告（模块四'))]
sec4 = lines[find(lambda s: s.strip().startswith('### 1. 拟采取的主要理论')):
             find(lambda s: s.strip().startswith('## 五、研究特色'))]
sec5 = lines[find(lambda s: s.strip().startswith('### 1. 面向有限交互预算的运动相关稀疏交互建模机制')):
             find(lambda s: s.strip().startswith('目前，我们已经完成'))]

i_plan = find(lambda s: s.strip().startswith('### 1. 论文工作进度安排'))
sched_block = []
j = i_plan + 1
while j < len(lines):
    if lines[j].strip().startswith('|'):
        sched_block.append(lines[j].strip())
    elif sched_block:
        break
    j += 1
sched_rows = parse_md_table(sched_block)[1:]   # 去掉表头
exp_lines = [l.strip() for l in lines[find(lambda s: s.strip().startswith('### 2. 预期研究成果')) + 1:]
             if l.strip() and l.strip() != '---']

# ---------------- 打开模板并填充 ----------------
doc = docx.Document(OUT)

def cell_fmt_from(ref_cell):
    """从已有单元格取 (latin, ea, size_pt)。"""
    for p in ref_cell.paragraphs:
        for r in p.runs:
            ea = None
            if r._element.rPr is not None and r._element.rPr.rFonts is not None:
                ea = r._element.rPr.rFonts.get(qn('w:eastAsia'))
            size = r.font.size.pt if r.font.size else 12
            return (r.font.name or 'Times New Roman', ea or '宋体', size)
    return ('Times New Roman', '宋体', 12)

def set_cell_text(cell, text, fmt=None, align=None):
    fmt = fmt or ('Times New Roman', '宋体', 12)
    for p in cell.paragraphs[1:]:
        p._element.getparent().remove(p._element)
    p = cell.paragraphs[0]
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    if align:
        p.alignment = align
    r = p.add_run(text)
    set_run_font(r, latin=fmt[0], ea=fmt[1], size=fmt[2])

# 封面：填表日期（其余留空由用户填写）
cover = doc.tables[1]
fmt_cover = cell_fmt_from(cover.rows[0].cells[1])
set_cell_text(cover.rows[6].cells[1], '2026年10月21日', fmt_cover,
              WD_ALIGN_PARAGRAPH.CENTER)

# 表2：题目/开题次数/摘要/关键词
t2 = doc.tables[2]
set_cell_text(t2.rows[0].cells[2], title_cn)
set_cell_text(t2.rows[1].cells[2], title_en)
fmt_cnt = cell_fmt_from(t2.rows[2].cells[2])
set_cell_text(t2.rows[2].cells[2], '☑ 第一次      □ 第二次      □ 第三次', fmt_cnt,
              WD_ALIGN_PARAGRAPH.CENTER)
abs_cell = t2.rows[3].cells[2]
clear_cell(abs_cell)
for para in abs_paras:
    add_para(abs_cell, conv_inline(para), indent=True)
finalize_cell(abs_cell)
set_cell_text(t2.rows[4].cells[2], kw_cn)
set_cell_text(t2.rows[5].cells[2], kw_en)

# 二、选题依据
c3 = doc.tables[3].rows[1].cells[0]
clear_cell(c3)
process_section(doc, sec2, c3)

# 三、研究内容及目标
c4 = doc.tables[4].rows[1].cells[0]
clear_cell(c4)
process_section(doc, sec3, c4)

# 四、研究方案及可行性分析
c5 = doc.tables[5].rows[1].cells[0]
clear_cell(c5)
process_section(doc, sec4, c5)

# 五、研究特色与创新之处
c6 = doc.tables[6].rows[0].cells[0]
clear_cell(c6)
process_section(doc, sec5, c6)

# 六、论文工作计划、预期成果
t7 = doc.tables[7]
for k, row in enumerate(sched_rows):
    tr = t7.rows[2 + k]
    vals = [row[0], row[1], row[2] if len(row) > 2 else '']
    for ci, val in enumerate(vals):
        cell = tr.cells[ci]
        for p in cell.paragraphs[1:]:
            p._element.getparent().remove(p._element)
        p = cell.paragraphs[0]
        for r in list(p.runs):
            r._element.getparent().remove(r._element)
        p.paragraph_format.line_spacing = Pt(14)
        if ci != 1:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        segs = conv_inline(val)
        for txt, b in segs:
            r = p.add_run(txt)
            set_run_font(r, size=10.5, bold=b)
# 删除多余的空行（模板 8 个空行，计划 7 条）
extra = 8 - len(sched_rows)
for _ in range(extra):
    tr = t7.rows[2 + len(sched_rows)]._tr
    t7._tbl.remove(tr)
# 预期研究成果
exp_cell = t7.rows[-1].cells[0]
clear_cell(exp_cell)
for l in exp_lines:
    add_para(exp_cell, conv_inline(l), indent=False, size=10.5, line_pt=16)
finalize_cell(exp_cell)

doc.save(OUT)

# ---------------- 输出整理版 md ----------------
def span(start_marker, end_marker=None):
    a = find(lambda s: s.strip().startswith(start_marker))
    b = find(lambda s: s.strip().startswith(end_marker)) if end_marker else len(lines)
    return lines[a:b]

part1 = span('## 一、学位论文摘要', '## 三、研究内容及目标')
part2 = span('## 二、选题依据', '## 六、论文工作计划')
part3 = span('## 三、研究内容及目标', '# 陕西师范大学硕士研究生学位论文开题报告（模块四')
part4 = span('## 四、研究方案及可行性分析', '## 五、研究特色与创新之处')
part5 = span('## 五、研究特色与创新之处', '目前，我们已经完成')
part6 = span('## 六、论文工作计划')
clean = ['# 陕西师范大学硕士研究生学位论文开题报告（整理版）', '']
for part in (part1, part2, part3, part4, part5, part6):
    clean.extend(part)
    clean.append('')
open(CLEAN_MD, 'w', encoding='utf-8').write('\n'.join(clean))

# ---------------- 校验 ----------------
abs_text = ''.join(abs_paras)
cn_chars = len(re.findall(r'[一-鿿]', abs_text))
print('摘要中文字数:', cn_chars, '(总字符', len(abs_text), ')')
leftover = []
doc2 = docx.Document(OUT)
def scan_tables(tbls):
    for t in tbls:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    if '$' in p.text or '\\' in p.text:
                        leftover.append(p.text[:60])
                scan_tables(c.tables)
scan_tables(doc2.tables)
for p in doc2.paragraphs:
    if '$' in p.text or '\\' in p.text:
        leftover.append(p.text[:60])
print('残留公式符号段落数:', len(leftover))
for x in leftover[:10]:
    print('  LEFT:', x.encode('unicode_escape').decode())
print('DONE ->', OUT)
