# train_unified.py - 完整版（添加参数自动修正）

import os
import sys
import re
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')

import gc
import argparse
from pathlib import Path
from datetime import datetime
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn

from ultralytics import YOLO

print("\n" + "="*100)
print("🔧 注册自定义模块")
print("="*100)

from register_modules import (
    register_custom_modules, 
    verify_registration,
    print_registration_summary
)

register_custom_modules()

if not verify_registration():
    print("⚠️  警告：部分模块注册失败，继续尝试...\n")

print_registration_summary()

print("\n" + "="*100)
print("🔧 安装 Ultralytics Hook")
print("="*100)

try:
    from patch_ultralytics_v2 import install_parse_model_hook
    install_parse_model_hook()
    print("✅ Hook 安装成功\n")
except Exception as e:
    print(f"⚠️  Hook 安装失败：{e}，继续尝试...\n")

# ============================================================================
# 配置
# ============================================================================

print("\n" + "="*100)
print("📋 加载训练配置")
print("="*100 + "\n")

DATA_YAML = r'C:\Users\KUST\xlw\bdd100k.yaml'

EPOCHS = 350
LEARNING_RATE = 2e-5

RESULTS_DIR = r'C:\Users\KUST\xlw\results-1'

BATCH_SIZES = {
    'exp01': 32,
    'exp02': 32,
    'exp03': 32,
    'exp04': 32,
    'exp05': 32,
    'exp06': 20,
    'exp07': 20,
}

EXPERIMENTS = {
    'exp01': {'name': '01_baseline_yolo11m', 'yaml': 'yolo11m.yaml'},
    'exp02': {'name': '02_transformer_yolo11m', 'yaml': 'yolo11m_pure_transformer.yaml'},
    'exp03': {'name': '03_dwt_transformer_yolo11m', 'yaml': 'yolo11m_dwt_transformer.yaml'},
    'exp04': {'name': '04_transformer_layer6_yolo11m', 'yaml': 'yolo11m_transformer_layer6.yaml'},
    'exp05': {'name': '05_dwt_transformer_layer6_yolo11m', 'yaml': 'yolo11m_dwt_transformer_layer6.yaml'},
    'exp06': {'name': '06_window_transformer_layer8_yolo11m', 'yaml': 'yolo11m_window_transformer_layer8.yaml'},
    'exp07': {'name': '07_dwt_window_transformer_layer8_yolo11m', 'yaml': 'yolo11m_dwt_window_transformer_layer8.yaml'},
}

print("✅ 配置加载完成\n")


# ============================================================================
# 参数自动修正函数
# ============================================================================

def fix_yaml_parameters(yaml_path, exp_id):
    """
    自动修正 YAML 中的参数
    """
    
    print("\n" + "="*100)
    print("🔧 自动修正 YAML 参数")
    print("="*100 + "\n")
    
    # 读取 YAML 文件
    # 新代码
    try:
        with open(yaml_path, 'r', encoding='gbk') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()

    
    # 定义参数修正规则
    # YOLO 格式：[c2, arg1, arg2, arg3, ...]
    # c2 是输出通道数，args 是其他参数
    fixes = {
        'exp04': {
            'old': r'- \[-1, 1, TransformerBlock, \[1024, 1024, 8, 8, 0\.0\]\]',
            'new': '- [-1, 1, TransformerBlock, [512, 8, 4, 512, 0.0]]',
            'description': 'layer6 Transformer 参数调整（512→512通道）'
            # [c2=512, num_heads=8, num_layers=4, hidden_dim=512, dropout=0.0]
        },
        'exp05': {
            'old': r'- \[-1, 1, DWTTransformerBlock, \[1024, 1024, 8, 4, 0\.0\]\]',
            'new': '- [-1, 1, DWTTransformerBlock, [512, 8, 4, 512, 0.0]]',
            'description': 'layer6 DWT+Transformer 参数调整（512→512通道）'
            # [c2=512, num_heads=8, num_layers=4, hidden_dim=512, dropout=0.0]
        },
        'exp06': {
            'old': r'- \[-1, 1, WindowTransformerBlock, \[1024, 1024, 8, 8, 0\.0\]\]',
            'new': '- [-1, 1, WindowTransformerBlock, [1024, 8, 8, 0, 4.0, 0.0]]',
            'description': 'layer8 Window Transformer 参数格式调整'
            # [c2=1024, num_heads=8, window_size=8, shift_size=0, mlp_ratio=4.0, dropout=0.0]
        },
        'exp07': {
            'old': r'- \[-1, 1, DWTWindowTransformerBlock, \[1024, 1024, 8, 8, 0\.0\]\]',
            'new': '- [-1, 1, DWTWindowTransformerBlock, [1024, 8, 8, 0, 4.0, 0.0]]',
            'description': 'layer8 DWT+Window Transformer 参数格式调整'
            # [c2=1024, num_heads=8, window_size=8, shift_size=0, mlp_ratio=4.0, dropout=0.0]
        }
    }
    
    if exp_id not in fixes:
        print(f"✅ 实验 {exp_id} 无需修正参数\n")
        return yaml_path
    
    fix_rule = fixes[exp_id]
    
    # 检查是否需要修正
    if re.search(fix_rule['old'], content):
        print(f"⚠️  检测到需要修正的参数")
        print(f"📝 {fix_rule['description']}\n")
        
        # 执行修正
        modified_content = re.sub(fix_rule['old'], fix_rule['new'], content)
        
        # 保存到临时文件
        temp_yaml_path = yaml_path.replace('.yaml', '_fixed.yaml')
        with open(temp_yaml_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"✅ 参数已修正")
        print(f"📝 修改内容：")
        print(f"   旧参数：{fix_rule['old']}")
        print(f"   新参数：{fix_rule['new']}")
        print(f"📁 临时文件：{temp_yaml_path}\n")
        
        return temp_yaml_path
    else:
        print(f"✅ 参数已正确，无需修正\n")
        return yaml_path



# ============================================================================
# 训练函数
# ============================================================================

def train_model(config_path, exp_id, exp_name, gpu_id=0, batch_size=None):
    """训练模型"""

    # ✅ 添加这一行：自动修正参数
    config_path = fix_yaml_parameters(config_path, exp_id)
    
    # ✅ 添加这些行 - 强制设置 GPU
    gpu_id = int(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)
    
    if batch_size is None:
        batch_size = BATCH_SIZES.get(exp_id, 32)
    
    print("\n" + "="*100)
    print(f"🚀 实验：{exp_name}")
    print("="*100)
    
    print("\n📋 实验配置：")
    print(f"  - 实验 ID：{exp_id}")
    print(f"  - 模型配置：{config_path}")
    print(f"  - 使用 GPU：{gpu_id}")
    print(f"  - 训练轮数：{EPOCHS}")
    print(f"  - 批大小：{batch_size} ⚡ 优化")
    print(f"  - 学习率：{LEARNING_RATE}")
    print(f"  - 优化器：AdamW")
    print(f"  - 保存位置：{RESULTS_DIR}\n")
    
    # ========== 加载模型 ==========
    print("\n" + "="*100)
    print("📝 加载模型")
    print("="*100 + "\n")
    
    try:
        print(f"⏳ 加载模型：{config_path}\n")
        model = YOLO(config_path)
        
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print(f"✅ 模型加载成功！")
        print(f"   - 总参数量：{total_params/1e6:.2f}M")
        print(f"   - 可训练参数：{trainable_params/1e6:.2f}M\n")
    
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        import traceback
        traceback.print_exc()
        return False

    # ========== 清理显存 ==========
    print("🧹 清理显存...")
    with torch.cuda.device(0):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    print("✅ 显存已清理\n")
    
    # ========== 开始训练 ==========
    print("\n" + "="*100)
    print("🚀 开始训练")
    print("="*100 + "\n")
    
    start_time = datetime.now()
    print(f"⏰ 训练开始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=416,
            batch=batch_size,
            device=0,
            workers=2,
        
            optimizer='AdamW',
            lr0=LEARNING_RATE,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
        
            warmup_epochs=10,
            warmup_momentum=0.8,
            warmup_bias_lr=0.00005,
        
            box=7.5,
            cls=0.5,
            dfl=1.5,
        
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            erasing=0.0,
            close_mosaic=10,
        
            amp=True,
            half=False,

           # clip_grad=1.0,
            nbs=64,
        
            save=True,
            save_period=10,
        
            project=RESULTS_DIR,
            name=exp_name,
            exist_ok=True,
            patience=30,
            plots=False,
            val=True,
            cache=False,
            verbose=True,
            deterministic=True,
            seed=0,
            single_cls=False,
            rect=False,
            cos_lr=False,
            fraction=1.0,
        )

        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*100)
        print(f"✅ 训练完成！")
        print("="*100)
        print(f"⏰ 训练结束时间：{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  训练耗时：{duration}")
        print(f"📁 结果保存位置：{RESULTS_DIR}/{exp_name}\n")
        
        # ========== 验证权重保存 ==========
        print("\n" + "="*100)
        print("✅ 验证权重保存")
        print("="*100 + "\n")
        
        weights_dir = Path(RESULTS_DIR) / exp_name / 'weights'
        
        if weights_dir.exists():
            print(f"📁 权重保存位置：{weights_dir}\n")
            
            best_pt = weights_dir / 'best.pt'
            last_pt = weights_dir / 'last.pt'
            
            if best_pt.exists():
                size_mb = best_pt.stat().st_size / 1e6
                print(f"✅ best.pt 已保存（{size_mb:.2f}MB）")
            else:
                print(f"❌ best.pt 未找到")
            
            if last_pt.exists():
                size_mb = last_pt.stat().st_size / 1e6
                print(f"✅ last.pt 已保存（{size_mb:.2f}MB）")
            else:
                print(f"❌ last.pt 未找到")
            
            print()
        else:
            print(f"❌ 权重目录不存在：{weights_dir}\n")
        
        return True
    
    except Exception as e:
        print(f"\n❌ 训练失败：{e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一训练脚本（自动参数修正）')
    parser.add_argument('--exp', type=str, default='exp01', help='实验 ID')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--batch', type=int, default=None, help='覆盖批大小')
    
    args = parser.parse_args()
    
    if args.exp not in EXPERIMENTS:
        print(f"❌ 未知的实验 ID：{args.exp}")
        print(f"可用的实验：{list(EXPERIMENTS.keys())}")
        return
    
    exp_config = EXPERIMENTS[args.exp]
    config_path = exp_config['yaml']
    exp_name = exp_config['name']
    
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在：{config_path}")
        return
    
    if args.batch is not None:
        print(f"⚠️  覆盖批大小：{args.batch}\n")
        batch_size = args.batch
    else:
        batch_size = None
    
    success = train_model(config_path, args.exp, exp_name, gpu_id=args.gpu, batch_size=batch_size)
    
    if success:
        print("\n" + "="*100)
        print("✅ 训练成功完成！")
        print("="*100 + "\n")
    else:
        print("\n" + "="*100)
        print("❌ 训练失败！")
        print("="*100 + "\n")



if __name__ == '__main__':
    main()
