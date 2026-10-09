# -*- coding: utf-8 -*-
# eval_exp15_only.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from ultralytics import YOLO

# ================== 所需的自定义模块定义 ==================
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
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(-2, -1).contiguous().reshape(B, C, H, W)
        out = self.proj(out)
        return self.norm(out + x)

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
        s, m, l = self.small(x), self.medium(x), self.large(x)
        w = self.weight_net(x)
        t = self.temperature * (1.0 - 0.5 * epoch/300) if epoch is not None else self.temperature
        w = (w / t).softmax(1)
        return self.dropout(self.norm(w[:,0:1]*s + w[:,1:2]*m + w[:,2:3]*l + x))

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class ECASABlock(nn.Module):
    def __init__(self, channels, num_heads=8, num_scales=3, temperature=4.0, dropout=0.0):
        super().__init__()
        self.eca = ECABlock(channels)
        self.attn = ScaleAdaptiveAttention(channels, num_heads, num_scales, temperature, dropout)
        self.norm = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, epoch=None):
        x = self.eca(x)
        out = self.attn(x, epoch)
        return self.dropout(self.norm(out + x))

# ================== 注册所有用到的自定义类 ==================
sys.modules['__main__'].LightweightMultiScaleAttention = LightweightMultiScaleAttention
sys.modules['__main__'].ScaleAdaptiveAttention = ScaleAdaptiveAttention
sys.modules['__main__'].ECABlock = ECABlock
sys.modules['__main__'].ECASABlock = ECASABlock

# ================== 模型路径 ==================
MODEL_PATH = r"C:\Users\KUST\xlw\results-2\15_ecasa\weights\best.pt"

# ================== 数据集配置 ==================
BDD100K_YAML = r'C:\Users\KUST\xlw\bdd100k.yaml'

ACDC_SCENARIOS = {
    "Fog":   r"C:\Users\KUST\xlw\acdc\acdc_fog.yaml",
    "Night": r"C:\Users\KUST\xlw\acdc\acdc_night.yaml",
    "Rain":  r"C:\Users\KUST\xlw\acdc\acdc_rain.yaml",
    "Snow":  r"C:\Users\KUST\xlw\acdc\acdc_snow.yaml",
}

# ACDC仅评估7个共有类别（排除train, traffic light, traffic sign）
EVAL_CLASSES = [0, 1, 2, 3, 4, 5, 8]

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    print("=" * 80)
    print("EVALUATION FOR exp15 (ECASA, layer8)")
    print("=" * 80)

    # ---------- BDD100K ----------
    print("\n--- BDD100K ---")
    model = YOLO(MODEL_PATH)
    metrics_bdd = model.val(data=BDD100K_YAML, split='val', workers=0, verbose=False)
    mAP50_bdd = metrics_bdd.box.map50
    mAP50_95_bdd = metrics_bdd.box.map
    print(f"BDD100K mAP50: {mAP50_bdd:.4f}, mAP50-95: {mAP50_95_bdd:.4f}")

    # ---------- ACDC ----------
    print("\n--- ACDC Adverse Weather ---")
    acdc_results = {}
    for scenario_name, yaml_path in ACDC_SCENARIOS.items():
        metrics = model.val(data=yaml_path, split='val', workers=0, classes=EVAL_CLASSES, verbose=False)
        mAP50 = metrics.box.map50
        acdc_results[scenario_name] = mAP50
        print(f"  {scenario_name:6s}: mAP50 = {mAP50:.4f}")

    # 计算 SSDI
    ssdi = sum(acdc_results.values()) / len(acdc_results)
    print(f"\n  SSDI (average): {ssdi:.4f}")

    # ---------- 汇总表 ----------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"BDD100K:  mAP50={mAP50_bdd:.4f}  mAP50-95={mAP50_95_bdd:.4f}")
    for s, v in acdc_results.items():
        print(f"ACDC {s:6s}: mAP50={v:.4f}")
    print(f"ACDC SSDI (4-scene avg): {ssdi:.4f}")
    print("=" * 80)

if __name__ == '__main__':
    main()