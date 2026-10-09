# -*- coding: utf-8 -*-
# train_exp08_final_fixed.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 根据空闲卡调整

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# ================== 轻量模块定义（数值稳定增强） ==================
class LightweightMultiScaleAttention(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(32, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # 多尺度增强（depthwise残差）
        q = q + self.dw3(q) + self.dw5(q)
        k = k + self.dw3(k) + self.dw5(k)
        v = v + self.dw3(v) + self.dw5(v)

        # 重塑并计算注意力
        q = q.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)  # [B, heads, HW, dim]
        k = k.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)

        # 数值稳定性：对 Q, K 做 L2 归一化
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(-2, -1).contiguous().reshape(B, C, H, W)
        out = self.proj(out)
        return self.norm(out + x)          # 残差连接

class ScaleAdaptiveAttention(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.small = LightweightMultiScaleAttention(channels, num_heads, dropout)
        self.medium = LightweightMultiScaleAttention(channels, num_heads, dropout)
        self.large = LightweightMultiScaleAttention(channels, num_heads, dropout)
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, 1),
            nn.BatchNorm2d(channels//4), nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels//8, 1),
            nn.BatchNorm2d(channels//8), nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, num_scales, 1)
        )
        self.temperature = temperature
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, epoch=None):
        s = self.small(x); m = self.medium(x); l = self.large(x)
        w = self.weight_net(x)
        if epoch is not None:
            t = self.temperature * (1.0 - 0.5 * epoch/300)
        else:
            t = self.temperature
        w = (w / t).softmax(1)
        fused = w[:,0:1]*s + w[:,1:2]*m + w[:,2:3]*l
        return self.dropout(self.norm(fused + x))

# 注册自定义模块
import ultralytics.nn.tasks as tasks
tasks.ScaleAdaptiveAttention = ScaleAdaptiveAttention
tasks.LightweightMultiScaleAttention = LightweightMultiScaleAttention
sys.modules['__main__'].ScaleAdaptiveAttention = ScaleAdaptiveAttention
sys.modules['__main__'].LightweightMultiScaleAttention = LightweightMultiScaleAttention

if __name__ == '__main__':
    DATA_YAML = r'C:\Users\KUST\xlw\bdd100k.yaml'
    RESULTS_DIR = r'C:\Users\KUST\xlw\results-2'
    EXP_NAME = '08_scale_adaptive_lightweight'

    # 1. 加载基线模型，获取实际缩放后的配置
    base_model = YOLO('yolo11m.yaml', verbose=False)
    cfg = base_model.model.yaml
    nc = 10

    # 2. 前向计算第8层的实际输入通道数
    with torch.no_grad():
        dummy = torch.randn(1, 3, 416, 416)
        feat = base_model.model.model[:8](dummy)  # 索引 0..7
        actual_channels = feat.shape[1]
    print(f"Actual input channels for layer 8: {actual_channels}")

    # 3. 修改 backbone 配置：将第 8 层替换为自定义模块
    backbone = cfg['backbone']
    backbone[8] = [-1, 1, 'ScaleAdaptiveAttention', [actual_channels, 8, 3, 4.0, 0.0]]

    # 4. 防止重建时再次缩放
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
        resume=True, 
        epochs=300,
        imgsz=416,
        batch=32,
        device=0,
        workers=2,               # 先用 0 避免 Windows 多进程死锁
        amp=True,
        optimizer='AdamW',
        lr0=0.001,               # 学习率与基线完全一致
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-3,
        warmup_epochs=50,        # 延长预热，帮助注意力模块稳定初期训练
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