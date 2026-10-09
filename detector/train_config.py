# train_config.py
"""
统一的训练配置文件
所有 7 个实验都使用这个配置
"""

# ============================================================================
# 基础训练配置（所有实验共用）
# ============================================================================

BASE_CONFIG = {
    # ========== 数据配置 ==========
    'data': 'bdd100k.yaml',
    'epochs': 300,
    'imgsz': 416,
    'batch': 48,
    'workers': 2,
    'cache': False,
    
    # ========== 优化器配置 ==========
    'optimizer': 'AdamW',
    'lr0': 0.00005,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'nbs': 64,  # ✅ 关键参数
    
    # ========== 预热配置 ==========
    'warmup_epochs': 10.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.00005,
    
    # ========== 损失函数权重 ==========
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    
    # ========== 数据增强 ==========
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    
    # ========== 其他配置 ==========
    'patience': 20,
    'amp': True,
    'save': True,
    'save_period': 10,
    'save_json': True,  # ✅ 关键参数
    'plots': True,
    'verbose': True,
    'seed': 0,
    'deterministic': True,
    'single_cls': False,
    'rect': False,
    'val': False,
    'fraction': 1.0,
    'exist_ok': True,
}

# ============================================================================
# 实验配置（每个实验的特定配置）
# ============================================================================

EXPERIMENTS = {
    # ========== 第 1 组：Baseline ==========
    'exp01': {
        'name': '01_baseline_yolo11m',
        'model_yaml': 'yolo11m.pt',
        'device': 0,
        'description': 'Baseline YOLOv11m（无增强）',
    },
    
    # ========== 第 2 组：Pure Transformer ==========
    'exp02': {
        'name': '02_transformer_yolo11m',
        'model_yaml': 'yolo11m_pure_transformer.yaml',
        'device': 1,
        'description': 'Pure Transformer @ P5/32',
    },
    
    # ========== 第 3 组：DWT + Transformer（融合） ==========
    'exp03': {
        'name': '03_dwt_transformer_yolo11m',
        'model_yaml': 'yolo11m_dwt_transformer.yaml',
        'device': 2,
        'description': 'DWT + Transformer（融合）@ P5/32',
    },
    
    # ========== 第 4 组：Transformer @ Layer 6 ==========
    'exp04': {
        'name': '04_transformer_layer6_yolo11m',
        'model_yaml': 'yolo11m_transformer_layer6.yaml',  # ✅ 正确的文件名
        'device': 0,
        'description': 'Transformer @ P4/16（26×26）',
    },
    
    # ========== 第 5 组：DWT + Transformer @ Layer 6 ==========
    'exp05': {
        'name': '05_dwt_transformer_layer6_yolo11m',
        'model_yaml': 'yolo11m_dwt_transformer_layer6.yaml',  # ✅ 正确的文件名
        'device': 1,
        'description': 'DWT + Transformer @ P4/16（26×26）',
    },
    
    # ========== 第 6 组：Window Transformer @ Layer 8（改进版） ==========
    'exp06': {
        'name': '06_window_transformer_layer8_yolo11m',
        'model_yaml': 'yolo11m_window_transformer_layer8.yaml',
        'device': 2,
        'description': 'Window Transformer（改进版）@ P5/32（13×13）',
    },
    
    # ========== 第 7 组：DWT + Window Transformer（交互融合）✅ 修复版 ==========
    'exp07': {
        'name': '07_dwt_window_transformer_layer8_yolo11m',
        'model_yaml': 'yolo11m_dwt_window_transformer_layer8.yaml',
        'device': 3,
        'description': 'DWT + Window Transformer（交互融合）@ P5/32 ✅ 修复版',
    },
}

# ========== 项目路径配置 ==========
PROJECT_PATH = r'C:\Users\KUST\xlw\results-1'  # ✅ 改为新路径
DATASET_PATH = r'C:\Users\KUST\xlw\bdd100k.yaml'


def get_experiment_config(exp_id: str) -> dict:
    """
    获取指定实验的完整配置
    
    Args:
        exp_id: 实验 ID（如 'exp01', 'exp02' 等）
    
    Returns:
        完整的训练配置字典
    """
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_id}. Available: {list(EXPERIMENTS.keys())}")
    
    exp_config = EXPERIMENTS[exp_id].copy()
    
    # 合并基础配置和实验特定配置
    config = {
        **BASE_CONFIG,
        'data': DATASET_PATH,
        'project': PROJECT_PATH,
        **exp_config,
    }
    
    return config


def print_experiment_info(exp_id: str):
    """打印实验信息"""
    config = get_experiment_config(exp_id)
    
    print("=" * 100)
    print(f"🚀 {config['description']}")
    print("=" * 100)
    print(f"📋 实验配置：")
    print(f"  - 实验 ID：{exp_id}")
    print(f"  - 实验名称：{config['name']}")
    print(f"  - 模型配置：{config['model_yaml']}")
    print(f"  - 使用 GPU：{config['device']}")
    print(f"  - 训练轮数：{config['epochs']}")
    print(f"  - 批大小：{config['batch']}")
    print(f"  - 学习率：{config['lr0']}")
    print(f"  - 优化器：{config['optimizer']}")
    print(f"  - 保存位置：{config['project']}")
    print("=" * 100)


if __name__ == '__main__':
    # 测试配置
    for exp_id in EXPERIMENTS.keys():
        print_experiment_info(exp_id)
        print()
