# -*- coding: utf-8 -*-
"""
aggregate_final.py
读取 _eval_all_s{0,1,2}.csv，算出论文所需的全部 mean±std。

用法：python multiseed/aggregate_final.py
"""
import os, csv, io, sys, statistics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

R2 = r'C:\Users\KUST\xlw\results-2'
SEEDS = [0, 1, 2]
MODELS = ['Baseline','DWT','AMAR','ADF-F','ADF-S','ADF-N','ADF-R','SE','CBAM','ECA','CA']
COLS = ['BDD_mAP50','BDD_mAP5095','Fog','Night','Rain','Snow','SSDI']

def load_csv(seed):
    p = os.path.join(R2, f'_eval_all_s{seed}.csv')
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name = row['Model']
            vals = {}
            for col in COLS:
                v = row.get(col, '')
                vals[col] = float(v) if v else None
            out[name] = vals
    return out

def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return (None, None)
    if len(vals) == 1: return (vals[0], 0.0)
    return (statistics.mean(vals), statistics.stdev(vals))

data = {s: load_csv(s) for s in SEEDS}
avail = sum(1 for s in SEEDS if data[s])
print(f'找到 {avail}/{len(SEEDS)} 个 seed 的评测文件\n')

# ============ 表1/表2: BDD 域内 ============
print('='*90)
print('表1/表2 — BDD100K 域内（独立评测）mean±std')
print('='*90)
print(f'{"Model":12s} {"mAP50 (mean±std)":22s} {"mAP50-95 (mean±std)":22s} {"n":>3s}')
print('-'*65)
for name in MODELS:
    m50s, m95s = [], []
    for s in SEEDS:
        d = data.get(s, {}).get(name, {})
        if d.get('BDD_mAP50') is not None:
            m50s.append(d['BDD_mAP50'])
        if d.get('BDD_mAP5095') is not None:
            m95s.append(d['BDD_mAP5095'])
    m50, sd50 = ms(m50s); m95, sd95 = ms(m95s)
    f50 = f'{m50:.4f}±{sd50:.4f}' if m50 else '   --'
    f95 = f'{m95:.4f}±{sd95:.4f}' if m95 else '   --'
    print(f'{name:12s} {f50:22s} {f95:22s} {len(m50s):>3d}')

# ============ 表3: ACDC 跨域 ============
print('\n' + '='*90)
print('表3 — ACDC 跨域（独立评测）mean±std')
print('='*90)
print(f'{"Model":12s} {"Fog":16s} {"Night":16s} {"Rain":16s} {"Snow":16s} {"SSDI":16s}')
print('-'*92)
for name in MODELS:
    cells = []
    for col in ['Fog','Night','Rain','Snow','SSDI']:
        vals = [data.get(s,{}).get(name,{}).get(col) for s in SEEDS]
        m, sd = ms(vals)
        cells.append(f'{m:.4f}±{sd:.4f}' if m else '   --')
    print(f'{name:12s} ' + ' '.join(f'{c:16s}' for c in cells))

# ============ WADI ============
print('\n' + '='*90)
print('WADI 计算（各场景最优取均值）')
print('='*90)
scenarios = ['Fog','Night','Rain','Snow']
for s in SEEDS:
    d = data.get(s, {})
    if not d: continue
    best = {}
    for sc in scenarios:
        vals = [(name, d[name][sc]) for name in MODELS if name in d and d[name].get(sc)]
        if vals:
            bn, bv = max(vals, key=lambda x: x[1])
            best[sc] = (bn, bv)
    if len(best) == 4:
        wadi = sum(v for _,v in best.values()) / 4
        print(f'\n  Seed {s}:')
        for sc in scenarios:
            print(f'    {sc:5s} best = {best[sc][0]:8s} {best[sc][1]:.4f}')
        print(f'    WADI = {wadi:.4f}')

# ============ 跨域排名反转 ============
print('\n' + '='*90)
print('排名反转检查（域内 rank vs SSDI rank）')
print('='*90)
for s in SEEDS:
    d = data.get(s, {})
    if not d: continue
    bdd_rank = sorted(d.keys(), key=lambda n: -(d[n].get('BDD_mAP50') or 0))
    ssdi_rank = sorted(d.keys(), key=lambda n: -(d[n].get('SSDI') or 0))
    print(f'\n  Seed {s}:')
    print(f'    BDD 域内 top3: {bdd_rank[:3]}')
    print(f'    SSDI 跨域 top3: {ssdi_rank[:3]}')
    # AMAR 在 BDD 排第几？CA 在 SSDI 排第几？
    if 'AMAR' in bdd_rank:
        print(f'    AMAR 域内排名: 第{bdd_rank.index("AMAR")+1}')
    if 'CA' in ssdi_rank:
        print(f'    CA  跨域排名: 第{ssdi_rank.index("CA")+1}')

print('\nDone.')
