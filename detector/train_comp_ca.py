# -*- coding: utf-8 -*-
# train_comp_ca.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'          # 根据空闲卡调整

import torch
import torch.nn as nn
import sys
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# ================== CA 模块定义 ==================
class CABlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 水平方向和垂直方向的平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # 共享的 1×1 卷积进行通道降维
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
        self.bn = nn.BatchNorm2d(channels // reduction)
        # 分别投影回原始通道数
        self.conv_h = nn.Conv2d(channels // reduction, channels, 1)
        self.conv_w = nn.Conv2d(channels // reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 方向池化
        x_h = self.pool_h(x)                 # [B, C, H, 1]
        x_w = self.pool_w(x).transpose(-2, -1)  # [B, C, 1, W] -> [B, C, W, 1]
        # 2. 拼接 + 共享变换
        y = torch.cat([x_h, x_w], dim=2)     # [B, C, H+W, 1]
        y = self.bn(self.conv1(y))
        y = nn.functional.relu(y)
        # 3. 分离回两个方向
        x_h, x_w = torch.split(y, [H, W], dim=2)
        # 4. 生成注意力并加权
        a_h = self.sigmoid(self.conv_h(x_h))  # [B, C, H, 1]
        a_w = self.sigmoid(self.conv_w(x_w.transpose(-2, -1)))  # [B, C, 1, W]
        return x * a_h * a_w


# ================== 注册模块 ==================
import ultralytics.nn.tasks as tasks
tasks.CABlock = CABlock
sys.modules['__main__'].CABlock = CABlock

if __name__ == '__main__':
    DATA_YAML = r'C:\Users\KUST\xlw\bdd100k.yaml'
    RESULTS_DIR = r'C:\Users\KUST\xlw\results-2'
    EXP_NAME = 'comp04_ca'

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

    # 3. 替换 backbone 第8层为 CABlock
    backbone = cfg['backbone']
    backbone[8] = [-1, 1, 'CABlock', [actual_channels, 16]]  # reduction=16

    # 4. 防止二次缩放
    cfg['width_multiple'] = 1.0
    cfg['depth_multiple'] = 1.0

    # 5. 重建模型
    model = DetectionModel(cfg, ch=3, nc=nc, verbose=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.info()

    # 6. 包装为 YOLO 对象并训练
    final_model = YOLO('yolo11m.yaml', verbose=False)
    final_model.model = model
    final_model.model.nc = nc
    final_model.model.yaml = cfg

    final_model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=416,
        batch=32,
        device=2,
        workers=2,
        amp=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-3,
        warmup_epochs=20,           # 与基线保持一致
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