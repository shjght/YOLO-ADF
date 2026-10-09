# -*- coding: utf-8 -*-
"""run_all.py 更新版：三阶段（训练 → BDD评测 → ACDC评测）
Phase 1: 训练（11 模型 × seed{1,2}）
Phase 2: BDD 独立评测（seed{0,1,2}，口径与汇总表一致）
Phase 3: ACDC 跨域评测（seed{0,1,2}）
"""
import os, sys, io, time, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

GPUS = [0, 1, 2, 3]
TRAIN_SEEDS = [1, 2]
EVAL_SEEDS  = [0, 1, 2]
POLL_SEC = 30

XLW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = r'C:\Users\KUST\xlw\results-2'
LOGS = os.path.join(XLW, 'multiseed', 'logs'); os.makedirs(LOGS, exist_ok=True)

SCRIPTS = [
    ('train.py',                      '01_baseline_consistent'),
    ('train_dwt.py',                  '04_dwt_consistent'),
    ('train_exp08_final_fixed.py',    '08_scale_adaptive_lightweight'),
    ('train_scale_adaptive_dwt.py',   '09_scale_adaptive_dwt_lightweight'),
    ('train_exp11.py',                '11_gated_fusion'),
    ('train_exp12.py',                '12_multiscale_gated_fusion'),
    ('train_exp13.py',                '13_sequential_fusion'),
    ('train_comp_se.py',              'comp01_se'),
    ('train_comp_cbam.py',            'comp02_cbam'),
    ('train_comp_eca.py',             'comp03_eca'),
    ('train_comp_ca.py',              'comp04_ca'),
]

def done_best(expdir, seed):
    return os.path.exists(os.path.join(RESULTS, expdir + (f'_s{seed}' if seed else ''), 'weights', 'best.pt'))

def run_pool(jobs):
    queue = list(jobs); running = {}; qi = 0; results = []
    def launch(gpu):
        nonlocal qi
        while qi < len(queue):
            j = queue[qi]; qi += 1
            if j['skip']():
                print(f"[skip ] {j['tag']}  (已完成)"); results.append((j['tag'],'skipped')); continue
            env = dict(os.environ); env.update(j.get('env', {})); env['XLW_GPU'] = str(gpu)
            lf = open(os.path.join(LOGS, j['tag'].replace('/','_') + '.log'), 'w')
            cmd = [sys.executable, j['cmd']]
            proc = subprocess.Popen(cmd, cwd=XLW, env=env, stdout=lf, stderr=subprocess.STDOUT)
            running[gpu] = (proc, lf, j); print(f"[start] {j['tag']}  -> GPU{gpu}")
            return True
        return False
    for g in GPUS: launch(g)
    while running:
        time.sleep(POLL_SEC)
        for g in list(running):
            proc, lf, j = running[g]
            if proc.poll() is not None:
                lf.close(); rc = proc.returncode
                print(f"[{'done ' if rc==0 else 'FAIL '}] {j['tag']}  GPU{g}  rc={rc}")
                results.append((j['tag'], 'ok' if rc==0 else f'FAIL rc={rc}'))
                del running[g]; launch(g)
    return results

def main():
    print('='*70)
    print(f'GPUS = {GPUS}  TRAIN_SEEDS={TRAIN_SEEDS}  EVAL_SEEDS={EVAL_SEEDS}')
    print('='*70)

    # Phase 1: 训练
    train_jobs = []
    for script, expdir in SCRIPTS:
        for s in TRAIN_SEEDS:
            train_jobs.append(dict(
                tag=f'train/{expdir}_s{s}', cmd=os.path.join('multiseed', script),
                env={'XLW_SEED': str(s)},
                skip=lambda s=s, e=expdir: done_best(e, s),
            ))
    print(f'\n### Phase 1: 训练 {len(train_jobs)} 个任务 ###')
    r1 = run_pool(train_jobs)

    # Phase 2: BDD 独立评测
    bdd_jobs = []
    for s in EVAL_SEEDS:
        csvp = os.path.join(RESULTS, f'_bdd_eval_s{s}.csv')
        bdd_jobs.append(dict(
            tag=f'eval/bdd_s{s}', cmd=os.path.join('multiseed', 'eval_bdd_multiseed.py'),
            env={'XLW_SEED': str(s)},
            skip=lambda p=csvp: os.path.exists(p),
        ))
    print(f'\n### Phase 2: BDD 独立评测 {len(bdd_jobs)} 个任务 ###')
    r2 = run_pool(bdd_jobs)

    # Phase 3: ACDC 跨域评测
    eval_jobs = []
    for s in EVAL_SEEDS:
        csvp = os.path.join(RESULTS, f'_acdc_eval_s{s}.csv')
        eval_jobs.append(dict(
            tag=f'eval/acdc_s{s}', cmd=os.path.join('multiseed', 'eval_acdc_multiseed.py'),
            env={'XLW_SEED': str(s)},
            skip=lambda p=csvp: os.path.exists(p),
        ))
    print(f'\n### Phase 3: ACDC 跨域评测 {len(eval_jobs)} 个任务 ###')
    r3 = run_pool(eval_jobs)

    # 汇总
    print('\n' + '='*70); print('全部完成。结果：'); print('='*70)
    for tag, st in r1 + r2 + r3:
        print(f'  {st:12s} {tag}')
    nok = sum(1 for _,st in (r1+r2+r3) if st=='ok')
    nfail = sum(1 for _,st in (r1+r2+r3) if st.startswith('FAIL'))
    print(f'\n成功 {nok}，失败 {nfail}，跳过 {len(r1+r2+r3)-nok-nfail}。')
    print('最后运行:  python multiseed/aggregate.py   出 mean±std（BDD + ACDC）')

if __name__ == '__main__':
    main()
