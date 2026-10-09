# train_exp08_final_pt.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import sys
import torch.nn as nn
from ultralytics import YOLO

# ===================== 轻量模块定义（放在本文件中，避免循环导入） =====================
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
        q = q + self.dw3(q) + self.dw5(q)
        k = k + self.dw3(k) + self.dw5(k)
        v = v + self.dw3(v) + self.dw5(v)
        q = q.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(-2, -1).contiguous().reshape(B, C, H, W)
        return self.norm(self.proj(out) + x)

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

# 注册自定义模块，以便加载 .pt 时能找到
sys.modules['__main__'].ScaleAdaptiveAttention = ScaleAdaptiveAttention
sys.modules['__main__'].LightweightMultiScaleAttention = LightweightMultiScaleAttention

# ===================== 训练入口 =====================
if __name__ == '__main__':
    DATA_YAML = r'C:\Users\KUST\xlw\bdd100k.yaml'
    MODEL_PATH = 'yolo11m_scale_adaptive_built.pt'   # 你已生成的包含模块的模型
    RESULTS_DIR = r'C:\Users\KUST\xlw\results-2'
    EXP_NAME = '08_scale_adaptive_lightweight'

    model = YOLO(MODEL_PATH)
    model.info()   # 应显示 22.0M 参数，第8层为 ScaleAdaptiveAttention

    model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=416,
        batch=32,
        device=0,
        workers=4,
        amp=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-3,
        warmup_epochs=20,# train_exp08_yaml.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import sys
from ultralytics import YOLO

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
        q = q + self.dw3(q) + self.dw5(q)
        k = k + self.dw3(k) + self.dw5(k)
        v = v + self.dw3(v) + self.dw5(v)
        q = q.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        k = k.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, H*W).transpose(-2, -1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(-2, -1).contiguous().reshape(B, C, H, W)
        return self.norm(self.proj(out) + x)

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

# 注册
import ultralytics.nn.tasks as tasks
tasks.ScaleAdaptiveAttention = ScaleAdaptiveAttention
tasks.LightweightMultiScaleAttention = LightweightMultiScaleAttention
sys.modules['__main__'].ScaleAdaptiveAttention = ScaleAdaptiveAttention
sys.modules['__main__'].LightweightMultiScaleAttention = LightweightMultiScaleAttention

if __name__ == '__main__':
    DATA_YAML = r'C:\Users\KUST\xlw\bdd100k.yaml'
    MODEL_YAML = 'yolo11m_scale_adaptive_attention.yaml'
    RESULTS_DIR = r'C:\Users\KUST\xlw\results-2'
    EXP_NAME = '08_scale_adaptive_lightweight'

    model = YOLO(MODEL_YAML, verbose=True)
    model.info()

    model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=416,
        batch=32,
        device=0,
        workers=4,
        amp=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-3,
        warmup_epochs=20,
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