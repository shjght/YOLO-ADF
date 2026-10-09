# -*- coding: utf-8 -*-
# train_comp_se.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'          # 根据空闲卡调整

import torch
import torch.nn as nn
import sys
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# ================== SE 模块定义 ==================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # Squeeze: H×W → 1×1
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()                           # 输出0~1的权重
        )

    def forward(self, x):
        weight = self.fc(x)                       # [B, C, 1, 1]
        return x * weight                         # Scale: 加权


# ================== 注册模块 ==================
import ultralytics.nn.tasks as tasks
tasks.SEBlock = SEBlock
sys.modules['__main__'].SEBlock = SEBlock

if __name__ == '__main__':
    DATA_YAML = r'C:\Users\KUST\xlw\bdd100k.yaml'
    RESULTS_DIR = r'C:\Users\KUST\xlw\results-2'
    EXP_NAME = 'comp01_se'

    # 1. 加载基线模型，获取缩放后的配置
    base_model = YOLO('yolo11m.yaml', verbose=False)
    cfg = base_model.model.yaml
    nc = 10

    # 2. 自动计算第8层的实际输入通道数
    with torch.no_grad():
        dummy = torch.randn(1, 3, 416, 416)
        feat = base_model.model.model[:8](dummy)
        actual_channels = feat.shape[1]
    print(f"Actual input channels for layer 8: {actual_channels}")

    # 3. 替换backbone第8层为SEBlock
    backbone = cfg['backbone']
    backbone[8] = [-1, 1, 'SEBlock', [actual_channels, 16]]  # reduction=16

    # 4. 防止二次缩放
    cfg['width_multiple'] = 1.0
    cfg['depth_multiple'] = 1.0

    # 5. 重建模型
    model = DetectionModel(cfg, ch=3, nc=nc, verbose=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.info()

    # 6. 包装为YOLO对象并训练
    final_model = YOLO('yolo11m.yaml', verbose=False)
    final_model.model = model
    final_model.model.nc = nc
    final_model.model.yaml = cfg

    final_model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=416,
        batch=32,
        device=0,
        workers=2,
        amp=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-3,
        warmup_epochs=20,           # 与基线保持一致，不需要延长预热
        warmup_momentum=0.8,
        warmup_bias_lr=0.00005,
        box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.2,
        copy_paste=0.0, erasing=0.0,
        close_mosaic=15,
        nbs=64,
        save=True, save_period=10,
        project=RESULTS_DIR, name=EXP_NAME,
        exist_ok=True,
        patience=50,
        plots=False, val=True,
        cache=False, verbose=True,
        deterministic=True, seed=0,
        cos_lr=True,
        fraction=1.0,
    )