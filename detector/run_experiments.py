# run_experiments.py
# -*- coding: utf-8 -*-
"""
快速启动脚本
方便地运行指定的实验或所有实验
"""

import subprocess
import sys
from train_config import EXPERIMENTS, print_experiment_info


def run_single_experiment(exp_id: str):
    """运行单个实验"""
    print(f"\n🚀 运行实验：{exp_id}\n")
    
    cmd = [sys.executable, 'train_unified.py', '--exp', exp_id]
    subprocess.run(cmd, check=False)


def run_all_experiments():
    """运行所有实验"""
    print(f"\n🚀 运行所有 {len(EXPERIMENTS)} 个实验\n")
    
    cmd = [sys.executable, 'train_unified.py', '--exp', 'all']
    subprocess.run(cmd, check=False)


def run_experiments_by_gpu():
    """按 GPU 分组运行实验（同一 GPU 的实验顺序执行）"""
    from collections import defaultdict
    
    # 按 GPU 分组
    gpu_groups = defaultdict(list)
    for exp_id, exp_config in EXPERIMENTS.items():
        gpu = exp_config['device']
        gpu_groups[gpu].append(exp_id)
    
    print("\n" + "=" * 100)
    print("📊 按 GPU 分组的实验")
    print("=" * 100)
    
    for gpu, exp_ids in sorted(gpu_groups.items()):
        print(f"\n🔹 GPU {gpu}:")
        for exp_id in exp_ids:
            print(f"   - {exp_id}: {EXPERIMENTS[exp_id]['description']}")
    
    print("\n" + "=" * 100)
    print("🚀 开始按 GPU 分组运行实验")
    print("=" * 100 + "\n")
    
    for gpu, exp_ids in sorted(gpu_groups.items()):
        print(f"\n{'=' * 100}")
        print(f"GPU {gpu} 的实验队列：{exp_ids}")
        print(f"{'=' * 100}\n")
        
        for exp_id in exp_ids:
            run_single_experiment(exp_id)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='快速启动脚本')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'all', 'by-gpu'],
        default='single',
        help='运行模式：single（单个）, all（所有）, by-gpu（按 GPU 分组）',
    )
    parser.add_argument(
        '--exp',
        type=str,
        default='exp01',
        help='实验 ID（仅在 --mode single 时使用）',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用的实验',
    )
    
    args = parser.parse_args()
    
    # 列出所有实验
    if args.list:
        print("\n" + "=" * 100)
        print("📋 可用的实验列表")
        print("=" * 100)
        for exp_id, exp_config in EXPERIMENTS.items():
            print(f"\n{exp_id}:")
            print(f"  描述：{exp_config['description']}")
            print(f"  GPU：{exp_config['device']}")
            print(f"  模型：{exp_config['model_yaml']}")
        print("\n" + "=" * 100 + "\n")
        sys.exit(0)
    
    # 运行实验
    if args.mode == 'single':
        run_single_experiment(args.exp)
    elif args.mode == 'all':
        run_all_experiments()
    elif args.mode == 'by-gpu':
        run_experiments_by_gpu()
