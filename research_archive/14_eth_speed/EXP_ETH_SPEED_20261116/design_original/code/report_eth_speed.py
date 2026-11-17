"""Render existing JSON results; no model/data loading or threshold changes."""
from __future__ import annotations

import csv
from pathlib import Path

from eth_speed_math import gain
from speed_io import read


def render(root):
    root = Path(root); result = read(root/'RESULTS.json'); decision = read(root/'DECISION.json')
    lines = ['# ETH/HOTEL 零训练诊断结果', '',
             f"训练建议：`{decision['status']}`。这是探索筛选，不能据此认定速度是唯一原因。", '',
             '| 场景 | 条件 | 样本数 | minADE | minFDE | 覆盖率 | ADE 改善 | FDE 改善 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    rows = []
    for fold, item in result['folds'].items():
        base = item['cases']['identity']['overall']
        for case, blocks in item['cases'].items():
            b = blocks['overall']; a = gain(base['minADE'], b['minADE']); f = gain(base['minFDE'], b['minFDE'])
            astr = f'{a:.2%}' if a is not None else '—'; fstr = f'{f:.2%}' if f is not None else '—'
            lines.append(f"| {fold} | {case} | {b['n']} | {b['minADE']:.6f} | {b['minFDE']:.6f} | {b['coverage']:.2%} | {astr} | {fstr} |")
            def append(family, label, block):
                rows.append(dict(fold=fold, case=case, family=family, label=label, **block))
            append('overall', 'all', b)
            for family in ('speed', 'maneuver', 'original_group'):
                for label, block in blocks[family].items(): append(family, label, block)
            for speed, members in blocks['speed_by_maneuver'].items():
                for maneuver, block in members.items(): append('speed_by_maneuver', speed+'/'+maneuver, block)
    lines += ['', '## 固定条件检查', '']
    for name, passed in decision['checks'].items(): lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines += ['', '速度分界只由相应折的 95% 训练子集确定；机动类型使用未来标签，仅用于评价分组。',
              '整体改善不等于每种机动都改善；小于 20 窗的分组只作描述。',
              '场景整体缩放同时改变邻居距离，因此不能单独归因于速度。',
              '本地视频、提取脚本和标签链的一致性不构成上游物理标定的独立验证。',
              'ETH/HOTEL 均已被用于开发分析，本结果不作为未查看过的独立测试。',
              'DECISION.json 只给出是否值得做一次增强训练的建议；没有自动启动训练。', '']
    (root/'RESULTS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (root/'STRATIFIED_RESULTS.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
